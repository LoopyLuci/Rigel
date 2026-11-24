#pragma once

// Layer read-ahead: Rigel's take on AirLLM's "read the next layer while this one computes".
//
// A dense model whose weights are memory-mapped and bigger than the memory that is free is read from disk by page faults:
// every miss is a small, random, synchronous read, and the layers are walked in the same order for every token, so
// whatever the operating system keeps is always the layers about to be evicted. This reads the layers that come next in
// large, ordered requests on a worker thread (PrefetchVirtualMemory on Windows, madvise(WILLNEED) elsewhere) while the
// current layer computes. It changes nothing about the compute and nothing about where the weights live; a model that
// fits in memory should not turn it on (it only costs a hint per layer).
//
// Off by default (--layer-readahead N, N layers ahead).

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
    llama_layer_readahead(const std::vector<std::pair<std::string, ggml_tensor *>> & tensors, int n_layer, int ahead)
        : n_layer(n_layer), ahead(ahead), ranges(n_layer > 0 ? n_layer : 0) {
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
        worker = std::thread([this] { run(); });
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
    }

    llama_layer_readahead(const llama_layer_readahead &) = delete;
    llama_layer_readahead & operator=(const llama_layer_readahead &) = delete;

    // Bytes of layer weights in host memory (what could be read ahead), and layers that have any.
    uint64_t bytes() const { return host_bytes; }
    int layers_with_host_weights() const {
        int n = 0;
        for (const auto & r : ranges) {
            n += r.empty() ? 0 : 1;
        }
        return n;
    }
    uint64_t requests() const { return n_requests.load(); }

    // Layer `il` has been computed: ask for the layers after it. The next token starts again at layer 0, so the layers past
    // the last one wrap round to the first.
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
            todo.push_back((int) (p % n_layer));
        }
        queued_upto = std::max(queued_upto, want);
        // a worker that cannot keep up must not fall behind forever: the oldest requests are the stalest
        while (todo.size() > (size_t) (2 * ahead + 2)) {
            todo.pop_front();
        }
        cv.notify_one();
    }

private:
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
