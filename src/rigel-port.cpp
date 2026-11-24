// rigel-port.cpp -- the Windows side of rigel-port.h (nothing here on other platforms).
#include "rigel-port.h"

#if defined(_WIN32)

#include <psapi.h>
#include <tlhelp32.h>

#include <cstdio>
#include <cstring>

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// POSIX calls on a bad descriptor (a second close(), a read of a closed fd) fail with EBADF; the CRT instead calls
// its invalid-parameter handler, which ends the process. The runtime was written for Linux and leans on the POSIX
// behaviour in places, so make the CRT return the error instead.
static void rigel_invalid_parameter(const wchar_t *, const wchar_t *, const wchar_t *, unsigned int, uintptr_t) {}
static const bool rigel_crt_posix_errors = [] {
    _set_invalid_parameter_handler(rigel_invalid_parameter);
    return true;
}();

// A crash (access violation and the like) prints its code, address and stack as module+offset to stderr before the
// process ends, so a log says where it happened (llvm-symbolizer turns the offsets into lines with a debug build).
static LONG WINAPI rigel_crash_report(EXCEPTION_POINTERS * ep) {
    const EXCEPTION_RECORD * er = ep->ExceptionRecord;
    fprintf(stderr, "\nrigel: crash 0x%08lX at %p", (unsigned long) er->ExceptionCode, er->ExceptionAddress);
    if (er->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && er->NumberParameters >= 2) {
        fprintf(stderr, " (%s %p)", er->ExceptionInformation[0] == 1 ? "writing" : "reading", (void *) er->ExceptionInformation[1]);
    }
    fprintf(stderr, "\n");
    void * frames[48];
    const USHORT n = CaptureStackBackTrace(0, 48, frames, nullptr);
    for (USHORT i = 0; i < n; i++) {
        HMODULE mod = nullptr;
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR) frames[i], &mod) && mod) {
            GetModuleFileNameA(mod, name, sizeof name);
        }
        const char * base = strrchr(name, '\\');
        fprintf(stderr, "  #%u %s+0x%llx\n", (unsigned) i, base ? base + 1 : name,
                (unsigned long long) ((const char *) frames[i] - (const char *) mod));
    }
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
static const bool rigel_crash_handler = [] {
    AddVectoredExceptionHandler(0, [](EXCEPTION_POINTERS * ep) -> LONG {
        const DWORD c = ep->ExceptionRecord->ExceptionCode;
        // only faults that end the process; C++ exceptions and debugger events pass through untouched
        if (c == EXCEPTION_ACCESS_VIOLATION || c == EXCEPTION_ILLEGAL_INSTRUCTION || c == EXCEPTION_STACK_OVERFLOW ||
            c == EXCEPTION_INT_DIVIDE_BY_ZERO || c == EXCEPTION_ARRAY_BOUNDS_EXCEEDED) {
            rigel_crash_report(ep);
        }
        return EXCEPTION_CONTINUE_SEARCH;
    });
    return true;
}();

// RIGEL_SAMPLE_MS=N: every N ms, write each thread's stack (module+offset, innermost first) to stderr as
// "rigel-sample tid: frames". A profiler for a machine without one: llvm-symbolizer names the offsets.
static void rigel_sample_once(DWORD self_pid, DWORD self_tid) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 te = {};
    te.dwSize = sizeof te;
    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
        if (te.th32OwnerProcessID != self_pid || te.th32ThreadID == self_tid) continue;
        HANDLE th = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID);
        if (!th) continue;
        if (SuspendThread(th) == (DWORD) -1) { CloseHandle(th); continue; }
        CONTEXT ctx = {};
        ctx.ContextFlags = CONTEXT_FULL;
        char line[2048];
        int len = snprintf(line, sizeof line, "rigel-sample %lu:", (unsigned long) te.th32ThreadID);
        if (GetThreadContext(th, &ctx)) {
            for (int depth = 0; depth < 24 && ctx.Rip; depth++) {
                HMODULE mod = nullptr;
                char name[MAX_PATH] = "?";
                if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                       (LPCSTR) ctx.Rip, &mod) && mod) {
                    GetModuleFileNameA(mod, name, sizeof name);
                }
                const char * base = strrchr(name, '\\');
                if (len < (int) sizeof line - 80) {
                    len += snprintf(line + len, sizeof line - len, " %s+0x%llx", base ? base + 1 : name,
                                    (unsigned long long) (ctx.Rip - (DWORD64) mod));
                }
                DWORD64 image = 0;
                PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(ctx.Rip, &image, nullptr);
                if (!fn) { // a leaf function: the return address is on top of the stack
                    ctx.Rip = *(DWORD64 *) ctx.Rsp;
                    ctx.Rsp += 8;
                    continue;
                }
                void * handler = nullptr;
                DWORD64 frame = 0;
                RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, ctx.Rip, fn, &ctx, &handler, &frame, nullptr);
            }
        }
        ResumeThread(th);
        CloseHandle(th);
        fprintf(stderr, "%s\n", line);
    }
    CloseHandle(snap);
}
static const bool rigel_sampler = [] {
    const char * e = getenv("RIGEL_SAMPLE_MS");
    const int ms = e ? atoi(e) : 0;
    if (ms <= 0) return false;
    std::thread([ms] {
        const DWORD pid = GetCurrentProcessId(), tid = GetCurrentThreadId();
        for (;;) {
            Sleep((DWORD) ms);
            rigel_sample_once(pid, tid);
        }
    }).detach();
    return true;
}();

static std::wstring rigel_widen(const char * s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    if (n <= 0) return {};
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
    w.resize((size_t) n - 1);
    return w;
}

int rigel_open(const char * path, int flags) {
    const std::wstring w = rigel_widen(path);
    if (w.empty()) { errno = ENOENT; return -1; }
    DWORD attrs = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
    attrs |= (flags & O_DIRECT) ? FILE_FLAG_NO_BUFFERING : FILE_FLAG_RANDOM_ACCESS;
    HANDLE h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, attrs, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        errno = GetLastError() == ERROR_FILE_NOT_FOUND || GetLastError() == ERROR_PATH_NOT_FOUND ? ENOENT : EACCES;
        return -1;
    }
    const int fd = _open_osfhandle((intptr_t) h, _O_RDONLY | _O_BINARY);
    if (fd < 0) { CloseHandle(h); errno = EMFILE; return -1; }
    return fd;
}

static HANDLE rigel_handle(int fd) {
    if (fd < 0) return INVALID_HANDLE_VALUE;
    return (HANDLE) _get_osfhandle(fd);
}

// One positioned read; works on overlapped and ordinary handles alike. Bytes read, or -errno.
static int64_t rigel_read_at(int fd, void * buf, size_t n, int64_t off) {
    HANDLE h = rigel_handle(fd);
    if (h == INVALID_HANDLE_VALUE) return -EBADF;
    if (off < 0) return -EINVAL;
    thread_local HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    int64_t total = 0;
    char * p = (char *) buf;
    while (n > 0) {
        const DWORD want = (DWORD) (n > (size_t) 0x40000000 ? 0x40000000 : n); // 1 GiB per call, sector aligned
        OVERLAPPED ov = {};
        ov.Offset     = (DWORD) ((uint64_t) off & 0xffffffffu);
        ov.OffsetHigh = (DWORD) ((uint64_t) off >> 32);
        ov.hEvent     = ev;
        ResetEvent(ev);
        DWORD got = 0;
        BOOL ok = ReadFile(h, p, want, &got, &ov);
        if (!ok && GetLastError() == ERROR_IO_PENDING) ok = GetOverlappedResult(h, &ov, &got, TRUE);
        if (!ok) {
            const DWORD e = GetLastError();
            if (e == ERROR_HANDLE_EOF) break;
            if (total > 0) break;
            return e == ERROR_INVALID_PARAMETER ? -EINVAL : -EIO;
        }
        total += got;
        if (got < want) break; // end of file
        p += got; off += got; n -= got;
    }
    return total;
}

ssize_t rigel_pread(int fd, void * buf, size_t n, int64_t off) {
    const int64_t r = rigel_read_at(fd, buf, n, off);
    if (r < 0) { errno = (int) -r; return -1; }
    return (ssize_t) r;
}

int64_t rigel_file_size(int fd) {
    LARGE_INTEGER s;
    HANDLE h = rigel_handle(fd);
    if (h == INVALID_HANDLE_VALUE || !GetFileSizeEx(h, &s)) return -1;
    return (int64_t) s.QuadPart;
}

void * rigel_mmap(void *, size_t len, int prot, int flags, int fd, int64_t off) {
    if (flags & MAP_ANONYMOUS) {
        void * p = VirtualAlloc(nullptr, len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        return p ? p : MAP_FAILED;
    }
    if (prot & PROT_WRITE) return MAP_FAILED; // only read-only file views are needed
    HANDLE h = rigel_handle(fd);
    if (h == INVALID_HANDLE_VALUE) return MAP_FAILED;
    HANDLE map = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!map) return MAP_FAILED;
    void * p = MapViewOfFile(map, FILE_MAP_READ, (DWORD) ((uint64_t) off >> 32), (DWORD) ((uint64_t) off & 0xffffffffu), len);
    CloseHandle(map); // the view keeps the mapping alive
    return p ? p : MAP_FAILED;
}

int rigel_munmap(void * addr, size_t) {
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery(addr, &mbi, sizeof mbi)) return -1;
    if (mbi.Type == MEM_MAPPED) return UnmapViewOfFile(addr) ? 0 : -1;
    return VirtualFree(addr, 0, MEM_RELEASE) ? 0 : -1;
}

int getrusage(int, struct rusage * u) {
    PROCESS_MEMORY_COUNTERS pmc = {};
    pmc.cb = sizeof pmc;
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return -1;
    u->ru_majflt = 0;
    u->ru_minflt = (long) pmc.PageFaultCount;
    u->ru_maxrss = (long) (pmc.PeakWorkingSetSize / 1024);
    return 0;
}

// ---- the io_uring stand-in ----
//
// Prepared entries wait in `sq` until io_uring_submit() hands them to the worker threads; each worker does one
// positioned read at a time, so up to `workers.size()` reads are in flight together. Completions queue in `cq` in
// the order they finish. A std::deque keeps the address of every element it holds while others are added, so the
// pointer io_uring_wait_cqe() returns stays good until io_uring_cqe_seen().

struct rigel_ring {
    unsigned entries = 0;
    std::vector<io_uring_sqe> sq;
    unsigned sq_n = 0;

    std::mutex m;
    std::condition_variable work_cv;
    std::condition_variable cq_cv;
    std::deque<io_uring_sqe> work;
    std::deque<io_uring_cqe> cq;
    unsigned inflight = 0;
    bool stop = false;
    std::vector<std::thread> workers;

    void run() {
        for (;;) {
            io_uring_sqe s;
            {
                std::unique_lock<std::mutex> lock(m);
                work_cv.wait(lock, [&] { return stop || !work.empty(); });
                if (stop && work.empty()) return;
                s = work.front();
                work.pop_front();
                inflight++;
            }
            const int64_t r = rigel_read_at(s.fd, (void *) (uintptr_t) s.addr, s.len, (int64_t) s.off);
            {
                std::lock_guard<std::mutex> lock(m);
                inflight--;
                cq.push_back({s.user_data, (int32_t) r, 0});
            }
            cq_cv.notify_all();
        }
    }
};

int io_uring_queue_init(unsigned entries, struct io_uring * ring, unsigned) {
    if (!ring || entries == 0) return -EINVAL;
    auto * r = new rigel_ring();
    r->entries = entries;
    r->sq.resize(entries);
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned n  = std::max(4u, std::min({entries, 16u, hw ? hw : 8u}));
    for (unsigned i = 0; i < n; i++) r->workers.emplace_back([r] { r->run(); });
    ring->impl = r;
    return 0;
}

void io_uring_queue_exit(struct io_uring * ring) {
    if (!ring || !ring->impl) return;
    rigel_ring * r = ring->impl;
    {
        std::lock_guard<std::mutex> lock(r->m);
        r->work.clear(); // queued reads are dropped; reads already started finish
        r->stop = true;
    }
    r->work_cv.notify_all();
    for (auto & t : r->workers) t.join();
    delete r;
    ring->impl = nullptr;
}

struct io_uring_sqe * io_uring_get_sqe(struct io_uring * ring) {
    rigel_ring * r = ring->impl;
    if (!r || r->sq_n >= r->entries) return nullptr;
    io_uring_sqe * s = &r->sq[r->sq_n++];
    *s = {};
    return s;
}

unsigned io_uring_sq_ready(const struct io_uring * ring) { return ring->impl ? ring->impl->sq_n : 0; }
unsigned io_uring_sq_space_left(const struct io_uring * ring) {
    return ring->impl ? ring->impl->entries - ring->impl->sq_n : 0;
}

int io_uring_submit(struct io_uring * ring) {
    rigel_ring * r = ring->impl;
    if (!r) return -EINVAL;
    const unsigned n = r->sq_n;
    bool completed = false;
    {
        std::lock_guard<std::mutex> lock(r->m);
        for (unsigned i = 0; i < n; i++) {
            const io_uring_sqe & s = r->sq[i];
            if (s.opcode == RIGEL_OP_READ) {
                r->work.push_back(s);
            } else if (s.opcode == RIGEL_OP_CANCEL) {
                int res = -ENOENT;
                for (auto it = r->work.begin(); it != r->work.end(); ++it) {
                    if (it->user_data == s.addr) {
                        r->cq.push_back({it->user_data, -ECANCELED, 0});
                        r->work.erase(it);
                        res = 0;
                        break;
                    }
                }
                if (res != 0 && r->inflight > 0) res = -EALREADY;
                r->cq.push_back({s.user_data, res, 0});
                completed = true;
            } else {
                r->cq.push_back({s.user_data, 0, 0});
                completed = true;
            }
        }
        r->sq_n = 0;
    }
    r->work_cv.notify_all();
    if (completed) r->cq_cv.notify_all();
    return (int) n;
}

int io_uring_submit_and_wait(struct io_uring * ring, unsigned wait_nr) {
    const int n = io_uring_submit(ring);
    if (n < 0) return n;
    rigel_ring * r = ring->impl;
    std::unique_lock<std::mutex> lock(r->m);
    r->cq_cv.wait(lock, [&] { return r->cq.size() >= wait_nr || (r->work.empty() && r->inflight == 0); });
    return n;
}

int io_uring_wait_cqe(struct io_uring * ring, struct io_uring_cqe ** cqe) {
    rigel_ring * r = ring->impl;
    if (!r) return -EINVAL;
    std::unique_lock<std::mutex> lock(r->m);
    r->cq_cv.wait(lock, [&] { return !r->cq.empty() || (r->work.empty() && r->inflight == 0); });
    if (r->cq.empty()) { *cqe = nullptr; return -EAGAIN; } // nothing was submitted: waiting would never end
    *cqe = &r->cq.front();
    return 0;
}

int io_uring_peek_cqe(struct io_uring * ring, struct io_uring_cqe ** cqe) {
    rigel_ring * r = ring->impl;
    if (!r) return -EINVAL;
    std::lock_guard<std::mutex> lock(r->m);
    if (r->cq.empty()) { *cqe = nullptr; return -EAGAIN; }
    *cqe = &r->cq.front();
    return 0;
}

void io_uring_cqe_seen(struct io_uring * ring, struct io_uring_cqe *) { io_uring_cq_advance(ring, 1); }

void io_uring_cq_advance(struct io_uring * ring, unsigned n) {
    rigel_ring * r = ring->impl;
    if (!r) return;
    std::lock_guard<std::mutex> lock(r->m);
    while (n-- > 0 && !r->cq.empty()) r->cq.pop_front();
}

struct io_uring_cqe * rigel_cq_at(struct io_uring * ring, unsigned i) {
    rigel_ring * r = ring->impl;
    if (!r) return nullptr;
    std::lock_guard<std::mutex> lock(r->m);
    return i < r->cq.size() ? &r->cq[i] : nullptr;
}

#endif // _WIN32
