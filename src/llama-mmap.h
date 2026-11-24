#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <cstdio>

// staging buffer size for direct I/O reads, 64MB works well for NVMe drives
#define LLAMA_DIRECT_IO_BUFFER_SIZE (64 * 1024 * 1024)

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    llama_file(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file(FILE * file);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    // list of [first, last) byte ranges within a file
    using ranges = std::vector<std::pair<size_t, size_t>>;

    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false,
               const ranges & lazy_ranges = {});
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);
    void drop_cache() const;
    void advise_random() const;

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_memory_range {
    const void * addr;
    size_t size;
};

using llama_memory_ranges = std::vector<llama_memory_range>;

// Prefetch the host pages covering these memory ranges.
void llama_prefetch(llama_memory_ranges mr);

// Lock the host pages covering these ranges into physical memory (VirtualLock / mlock) so they are never evicted. Stops at the
// first range that cannot be locked (`all_locked` tells whether every one was). Returns the page-aligned ranges that are locked;
// give them to llama_unlock_ranges.
llama_memory_ranges llama_lock_ranges(llama_memory_ranges mr, bool * all_locked = nullptr);
void llama_unlock_ranges(const llama_memory_ranges & locked);

// Physical memory that is free or can be reclaimed right now, in bytes (0 when unknown); `total` gets the machine's.
uint64_t llama_available_memory(uint64_t * total = nullptr);

size_t llama_path_max();
