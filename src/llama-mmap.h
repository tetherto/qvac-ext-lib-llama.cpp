#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cstdio>

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    virtual ~llama_file() = default;

    virtual size_t tell() const = 0;
    virtual size_t size() const = 0;
    virtual int file_id() const = 0;

    virtual void seek(size_t offset, int whence) const = 0;

    virtual void read_raw(void * ptr, size_t len) = 0;
    virtual void read_raw_unsafe(void * ptr, size_t len) { read_raw(ptr, len); }
    virtual void read_aligned_chunk(void * dest, size_t size) { read_raw(dest, size); }
    virtual uint32_t read_u32() = 0;

    virtual void write_raw(const void * ptr, size_t len) const = 0;
    virtual void write_u32(uint32_t val) const = 0;

    virtual size_t read_alignment() const { return 1; }
    virtual bool has_direct_io() const { return false; }
};

struct llama_file_disk : public llama_file {
    llama_file_disk(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file_disk(FILE * file);
    ~llama_file_disk() override;

    size_t tell() const override;
    size_t size() const override;
    int file_id() const override;

    void seek(size_t offset, int whence) const override;

    void read_raw(void * ptr, size_t len) override;
    void read_raw_unsafe(void * ptr, size_t len) override;
    void read_aligned_chunk(void * dest, size_t size) override;
    uint32_t read_u32() override;

    void write_raw(const void * ptr, size_t len) const override;
    void write_u32(uint32_t val) const override;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false);
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

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

size_t llama_path_max();
