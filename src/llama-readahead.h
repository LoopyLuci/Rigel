#pragma once

// Layer paging for a dense model whose weights are memory-mapped and bigger than the memory that is free. Two parts, which
// work together and are each off by default:
//
// Read-ahead (--layer-readahead N): Rigel's take on AirLLM's "read the next layer while this one computes". Every miss on a
// memory-mapped weight is a small, synchronous read, and the layers are walked in the same order for every token. This reads
// the layers that come next in large, ordered requests on a worker thread (PrefetchVirtualMemory on Windows,
// madvise(WILLNEED) elsewhere) while the current layer computes.
//
// Resident set (--layer-resident-mib N): the walk over all the layers for every token, with a cache smaller than the model,
// makes least-recently-used eviction throw out every page just before it is needed again, so the whole model is read from disk
// for every token. Locking the first layers in memory (VirtualLock / mlock) up to a budget, and streaming only the rest, cuts
// the bytes read per token from the model's size to the model's size minus the locked part. AirLLM, which re-reads every layer
// for every token, cannot do this.
//
// Neither changes the compute or where the weights live. A model that fits in memory should turn both off (read-ahead costs
// a hint per layer, the resident set locks memory that nothing needed locked).

#include "llama-mmap.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

class llama_layer_readahead {
public:
    // `tensors`: every tensor of the model by name; the ones named blk.<il>.* that sit in host memory are the layer's.
    // `ahead`: layers to read ahead (0 = none). `resident_mib`: weights to lock in memory, in MiB (> 0), or what is free minus
    // a margin (< 0), or none (0).
    llama_layer_readahead(const std::vector<std::pair<std::string, ggml_tensor *>> & tensors, int n_layer, int ahead, int resident_mib = 0)
        : n_layer(n_layer), ahead(ahead), ranges(n_layer > 0 ? n_layer : 0), pinned(n_layer > 0 ? n_layer : 0, false) {
        for (const auto & [name, t] : tensors) {
            if (!t || !t->data || !t->buffer || name.compare(0, 4, "blk.") != 0) {
                continue;
            }
            const int il = atoi(name.c_str() + 4);
            if (il < 0 || il >= n_layer || !ggml_backend_buffer_is_host(t->buffer)) {
                continue;
            }
            ranges[il].push_back({ t->data, ggml_nbytes(t) });
            host_bytes += ggml_nbytes(t);
        }
        if (resident_mib != 0 && host_bytes > 0) {
            pin(resident_budget(resident_mib));
        }
        if (ahead > 0) {
            worker = std::thread([this] { run(); });
        }
    }

    ~llama_layer_readahead() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        llama_unlock_ranges(locked);
    }

    llama_layer_readahead(const llama_layer_readahead &) = delete;
    llama_layer_readahead & operator=(const llama_layer_readahead &) = delete;

    // Bytes of layer weights in host memory (what could be read ahead or locked), and layers that have any.
    uint64_t bytes() const { return host_bytes; }
    int layers_with_host_weights() const {
        int n = 0;
        for (const auto & r : ranges) {
            n += r.empty() ? 0 : 1;
        }
        return n;
    }
    uint64_t requests() const { return n_requests.load(); }

    // What the resident set holds: bytes locked in memory (whole pages) and the layers locked completely.
    uint64_t resident_bytes() const { return locked_bytes; }
    int resident_layers() const { return (int) std::count(pinned.begin(), pinned.end(), true); }

    // Layer `il` has been computed: ask for the layers after it. The next token starts again at layer 0, so the layers past
    // the last one wrap round to the first. A layer that is locked in memory is not asked for.
    void layer_done(int il) {
        if (il < 0 || il >= n_layer || ahead <= 0) {
            return;
        }
        std::lock_guard<std::mutex> lk(mu);
        if (il < last_il) {
            pass++; // a new walk over the layers began
        }
        last_il = il;
        const int64_t here = (int64_t) pass * n_layer + il;
        const int64_t want = here + ahead;
        for (int64_t p = std::max<int64_t>(queued_upto, here) + 1; p <= want; p++) {
            const int target = (int) (p % n_layer);
            if (!pinned[target]) {
                todo.push_back(target);
            }
        }
        queued_upto = std::max(queued_upto, want);
        // a worker that cannot keep up must not fall behind forever: the oldest requests are the stalest
        while (todo.size() > (size_t) (2 * ahead + 2)) {
            todo.pop_front();
        }
        cv.notify_one();
    }

private:
    // The budget in bytes. A request in MiB is capped at 85% of the machine's memory; "what is free" (< 0) leaves the larger of
    // 2 GiB and a tenth of the machine's memory to everything else.
    uint64_t resident_budget(int resident_mib) const {
        uint64_t total = 0;
        const uint64_t avail = llama_available_memory(&total);
        uint64_t budget = 0;
        if (resident_mib > 0) {
            budget = (uint64_t) resident_mib << 20;
            if (total > 0) {
                budget = std::min<uint64_t>(budget, total / 100 * 85);
            }
        } else if (total > 0) {
            const uint64_t margin = std::max<uint64_t>(2ull << 30, total / 10);
            budget = avail > margin ? avail - margin : 0;
        }
        return std::min<uint64_t>(budget, host_bytes);
    }

    // Lock whole layers, first layer first, while they fit in the budget. A layer that does not fit is skipped, since a
    // later one may be smaller; the first lock that fails (no memory, or the process limit) ends it.
    void pin(uint64_t budget) {
        for (int il = 0; il < n_layer; il++) {
            if (ranges[il].empty()) {
                continue;
            }
            llama_memory_ranges mr;
            mr.reserve(ranges[il].size());
            uint64_t layer_bytes = 0;
            for (const auto & r : ranges[il]) {
                mr.push_back({ r.first, r.second });
                layer_bytes += r.second;
            }
            if (locked_bytes + layer_bytes > budget) {
                continue;
            }
            bool all = false;
            llama_memory_ranges done = llama_lock_ranges(std::move(mr), &all);
            for (const auto & d : done) {
                locked_bytes += d.size;
                locked.push_back(d);
            }
            if (!all) {
                break;
            }
            pinned[il] = true;
        }
    }

    void run() {
        for (;;) {
            int il;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [this] { return stop || !todo.empty(); });
                if (stop) {
                    return;
                }
                il = todo.front();
                todo.pop_front();
            }
            llama_memory_ranges mr;
            mr.reserve(ranges[il].size());
            for (const auto & r : ranges[il]) {
                mr.push_back({ r.first, r.second });
            }
            llama_prefetch(std::move(mr));
            n_requests++;
        }
    }

    int n_layer;
    int ahead;
    // per layer: [address, bytes] of each tensor in host memory
    std::vector<std::vector<std::pair<const void *, size_t>>> ranges;
    uint64_t host_bytes = 0;

    // the resident set: layers locked in memory, and the page ranges that hold them
    std::vector<bool>   pinned;
    llama_memory_ranges locked;
    uint64_t            locked_bytes = 0;

    std::mutex              mu;
    std::condition_variable cv;
    std::deque<int>         todo;
    int                     pass        = 0;
    int                     last_il     = -1;
    int64_t                 queued_upto = -1;
    bool                    stop        = false;
    std::atomic<uint64_t>   n_requests{0};
    std::thread             worker;
};
