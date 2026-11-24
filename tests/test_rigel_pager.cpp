// test_rigel_pager.cpp -- Rigel (Windows + HIP): the expert pager reads the same bytes as a plain file read.
//
// The Windows counterpart of test_moe_pager.cpp: every probe reads one expert's gate, up and down slices through the
// pager (blocking fetch, then prefetch + commit through the io_uring stand-in in rigel-port.cpp) into GPU memory,
// copies them back, and compares them with an ordinary buffered read of the same range of the GGUF.
//
// usage: test-rigel-pager <model.gguf> <n_layer> <n_expert> <n_used>
#include "moe_pager.h"
#include "rigel-gpu.h"
#include "rigel-port.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static bool read_reference(const char * path, const moe_expert_slice & s, char * dst, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg((std::streamoff) (s.off + s.pre));
    f.read(dst, (std::streamsize) n);
    return (size_t) f.gcount() == n;
}

static bool check(moe_pager & pager, const char * path, int il, int e, bool async) {
    size_t want[3];
    for (int r = 0; r < 3; r++) want[r] = pager.expert_role_bytes(il, r);
    const size_t total = want[0] + want[1] + want[2];
    char * d = nullptr;
    if (cudaMalloc((void **) &d, total) != cudaSuccess) return false;
    cudaMemset(d, 0, total);
    void * dst[3] = {d, d + want[0], d + want[0] + want[1]};
    bool ok = async ? pager.commit(il, e, dst[0], want[0], dst[1], want[1], dst[2], want[2])
                    : pager.fetch(il, e, dst[0], want[0], dst[1], want[1], dst[2], want[2]);
    if (!ok) {
        fprintf(stderr, "FAIL: %s(L%d, E%d)\n", async ? "commit" : "fetch", il, e);
        cudaFree(d);
        return false;
    }
    const moe_expert_slice * sl = pager.slices(il, e);
    for (int r = 0; r < 3 && ok; r++) {
        std::vector<char> ref(want[r]), got(want[r]);
        if (!read_reference(path, sl[r], ref.data(), want[r])) {
            fprintf(stderr, "FAIL: reference read L%d E%d role %d\n", il, e, r);
            ok = false;
            break;
        }
        cudaMemcpy(got.data(), dst[r], want[r], cudaMemcpyDeviceToHost);
        ok = memcmp(ref.data(), got.data(), want[r]) == 0;
    }
    printf("  %s(L%2d, E%3d): %zu bytes  %s\n", async ? "async" : "fetch", il, e, total, ok ? "EXACT MATCH" : "*** MISMATCH ***");
    cudaFree(d);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.gguf> <n_layer> <n_expert> <n_used>\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char * path = argv[1];
    const int NL = atoi(argv[2]), NE = atoi(argv[3]), NU = atoi(argv[4]);
    moe_pager pager;
    std::string err;
    if (!pager.init(path, NL, NE, NU, err)) {
        fprintf(stderr, "FAIL: pager.init: %s\n", err.c_str());
        return 1;
    }
    printf("pager: layers=%d experts=%d used=%d bytes/expert=%zu\n", pager.n_layer(), pager.n_expert(), pager.n_used(),
           pager.bytes_per_expert());
    int bad = 0;
    const int probes[][2] = {{0, 0}, {0, 1}, {NL / 3, NE / 2}, {NL - 1, NE - 1}, {NL / 2, 7}};
    for (auto & p : probes) bad += !check(pager, path, p[0], p[1], false);
    const int aprobes[][2] = {{1, 19 % NE}, {NL / 4, 44 % NE}, {NL / 2, 99 % NE}, {NL - 2, 3}};
    for (auto & p : aprobes) {
        if (!pager.prefetch(p[0], p[1])) {
            fprintf(stderr, "FAIL: prefetch(L%d, E%d) refused\n", p[0], p[1]);
            bad++;
        }
    }
    for (auto & p : aprobes) bad += !check(pager, path, p[0], p[1], true);
    printf(bad ? "FAILED: %d probe(s)\n" : "OK: every probe matched\n", bad);
    return bad ? 1 : 0;
}
