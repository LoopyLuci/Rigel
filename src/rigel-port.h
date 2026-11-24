// rigel-port.h -- the POSIX pieces the paged MoE runtime uses, for Linux and Windows.
//
// Rigel is a custom build of Betelgeuse for Windows and AMD GPUs (HIP). On Linux this header includes the system
// headers the runtime was written against (liburing, O_DIRECT, mmap). On Windows it supplies the same small surface:
//   - files opened with rigel_open(): overlapped handles (O_DIRECT adds FILE_FLAG_NO_BUFFERING), wrapped in CRT
//     descriptors so close() and dup() keep working;
//   - pread(), posix_fadvise(), mmap()/munmap() (anonymous memory and read-only file views), sysconf(), getrusage(),
//     clock_gettime(), posix_memalign() (free with rigel_aligned_free);
//   - the subset of liburing the runtime calls, run by a pool of threads that issue positioned overlapped reads.
#pragma once

#include <cstddef>
#include <cstdint>

#if !defined(_WIN32)

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <liburing.h>
#if defined(__linux__)
#include <linux/ioprio.h>
#endif
#include <cstdlib>

#define RIGEL_HAS_URING 1

static inline int rigel_open(const char * path, int flags) { return open(path, flags); }
static inline void rigel_aligned_free(void * p) { free(p); }
// The size of an open file (st_size is 64-bit here).
static inline int64_t rigel_file_size(int fd) {
    struct stat st;
    return fstat(fd, &st) == 0 ? (int64_t) st.st_size : -1;
}

#else // _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <malloc.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstdlib>
#include <ctime>

#define RIGEL_HAS_URING 1

#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef SSIZE_T ssize_t;
#endif

#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#endif
// Not a CRT flag: rigel_open() turns it into FILE_FLAG_NO_BUFFERING (reads must then be sector aligned).
#define O_DIRECT 0x04000000

int     rigel_open(const char * path, int flags);
ssize_t rigel_pread(int fd, void * buf, size_t n, int64_t off);
int64_t rigel_file_size(int fd);
#define pread(fd, buf, n, off) rigel_pread((fd), (buf), (n), (int64_t) (off))
// close() and dup() are the CRT's own (io.h): they close and duplicate the handle behind the descriptor.

#define POSIX_FADV_NORMAL     0
#define POSIX_FADV_RANDOM     1
#define POSIX_FADV_SEQUENTIAL 2
#define POSIX_FADV_WILLNEED   3
#define POSIX_FADV_DONTNEED   4
static inline int posix_fadvise(int, int64_t, int64_t, int) { return 0; }

#define PROT_NONE     0
#define PROT_READ     1
#define PROT_WRITE    2
#define MAP_SHARED    1
#define MAP_PRIVATE   2
#define MAP_ANONYMOUS 0x20
#define MAP_FAILED    ((void *) -1)
// Anonymous memory (committed read-write pages) or a read-only view of a file opened with rigel_open().
void * rigel_mmap(void * addr, size_t len, int prot, int flags, int fd, int64_t off);
int    rigel_munmap(void * addr, size_t len);
#define mmap(addr, len, prot, flags, fd, off) rigel_mmap((addr), (len), (prot), (flags), (fd), (int64_t) (off))
#define munmap(addr, len) rigel_munmap((addr), (len))

static inline int posix_memalign(void ** p, size_t align, size_t n) {
    *p = _aligned_malloc(n, align);
    return *p ? 0 : ENOMEM;
}
static inline void rigel_aligned_free(void * p) { _aligned_free(p); }

#define _SC_PAGESIZE 30
static inline long sysconf(int name) {
    if (name != _SC_PAGESIZE) return -1;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (long) si.dwPageSize;
}

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
static inline int rigel_clock_gettime(int, struct timespec * t) {
    static LARGE_INTEGER freq = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    t->tv_sec  = (time_t) (c.QuadPart / freq.QuadPart);
    t->tv_nsec = (long) ((c.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
    return 0;
}
#define clock_gettime(clk, t) rigel_clock_gettime((clk), (t))

#define RUSAGE_SELF 0
struct rusage {
    long ru_majflt;
    long ru_minflt;
    long ru_maxrss;
};
// Page faults on Windows are not split into major and minor: all of them are reported as minor.
int getrusage(int who, struct rusage * u);

// Page-cache hints. Windows has no cheap residency query or readahead for a byte range of a file: mincore() fails
// (so nothing counts as resident) and readahead() does nothing, so the page-cache warmer is idle on Windows. Expert
// reads themselves go through the ring and the pinned host cache as on Linux.
static inline int mincore(void *, size_t, unsigned char *) { return -1; }
static inline ssize_t readahead(int, int64_t, size_t) { return 0; }

// The I/O priority of a read: accepted and ignored.
#define IOPRIO_CLASS_IDLE 3
#define IOPRIO_PRIO_VALUE(cls, data) (((cls) << 13) | (data))

// ---- liburing, the part the runtime uses ----

#define IOSQE_ASYNC (1U << 4)

enum : uint8_t { RIGEL_OP_NOP = 0, RIGEL_OP_READ = 1, RIGEL_OP_CANCEL = 2 };

struct io_uring_sqe {
    uint8_t  opcode;
    uint8_t  flags;
    uint16_t ioprio;
    int32_t  fd;
    uint64_t off;
    uint64_t addr;
    uint32_t len;
    uint64_t user_data;
};

struct io_uring_cqe {
    uint64_t user_data;
    int32_t  res;
    uint32_t flags;
};

struct rigel_ring;
struct io_uring {
    rigel_ring * impl = nullptr;
};

int  io_uring_queue_init(unsigned entries, struct io_uring * ring, unsigned flags);
void io_uring_queue_exit(struct io_uring * ring);
struct io_uring_sqe * io_uring_get_sqe(struct io_uring * ring);
int  io_uring_submit(struct io_uring * ring);
int  io_uring_submit_and_wait(struct io_uring * ring, unsigned wait_nr);
int  io_uring_wait_cqe(struct io_uring * ring, struct io_uring_cqe ** cqe);
int  io_uring_peek_cqe(struct io_uring * ring, struct io_uring_cqe ** cqe);
void io_uring_cqe_seen(struct io_uring * ring, struct io_uring_cqe * cqe);
void io_uring_cq_advance(struct io_uring * ring, unsigned n);
unsigned io_uring_sq_ready(const struct io_uring * ring);
unsigned io_uring_sq_space_left(const struct io_uring * ring);
// The i-th completion that has not been seen yet, or null.
struct io_uring_cqe * rigel_cq_at(struct io_uring * ring, unsigned i);

#define io_uring_for_each_cqe(ring, head, cqe) \
    for ((head) = 0; ((cqe) = rigel_cq_at((ring), (head))) != nullptr; (head)++)

static inline void io_uring_prep_read(struct io_uring_sqe * sqe, int fd, void * buf, unsigned n, uint64_t off) {
    sqe->opcode = RIGEL_OP_READ;
    sqe->fd     = fd;
    sqe->addr   = (uint64_t) (uintptr_t) buf;
    sqe->len    = n;
    sqe->off    = off;
}
static inline void io_uring_prep_cancel64(struct io_uring_sqe * sqe, uint64_t user_data, int) {
    sqe->opcode = RIGEL_OP_CANCEL;
    sqe->fd     = -1;
    sqe->addr   = user_data;
}
static inline void io_uring_sqe_set_data(struct io_uring_sqe * sqe, void * data) {
    sqe->user_data = (uint64_t) (uintptr_t) data;
}
static inline void * io_uring_cqe_get_data(const struct io_uring_cqe * cqe) {
    return (void *) (uintptr_t) cqe->user_data;
}

#endif // _WIN32
