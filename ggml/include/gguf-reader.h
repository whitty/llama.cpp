#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define GGML_API_CLASS __declspec(dllexport)
#        else
#            define GGML_API_CLASS __declspec(dllimport)
#        endif
#    else
#        ifdef GGML_BUILD
#            define GGML_API_CLASS __attribute__ ((visibility ("default")))
#        else
#            define GGML_API_CLASS
#        endif
#    endif
#else
#    define GGML_API_CLASS
#endif

#ifdef _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#else
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#endif

#define GGUF_MAX_STRING_LENGTH  (1024*1024*1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

struct GGML_API_CLASS gguf_reader_impl {
    gguf_reader_impl(gguf_reader_callback_t callback,
                     void * userdata,
                     size_t max_chunk_read,
                     uint64_t data_offset = 0,
                     uint64_t nbytes_remain = 0);

    virtual ~gguf_reader_impl() = default;

    virtual size_t read_raw(void * dst, size_t size) const;

    virtual uint64_t remaining() const {
        return nbytes_remain;
    }

    virtual bool seek(uint64_t absolute_offset) const;

    virtual uint64_t tell() const {
        return data_offset;
    }

private:
    gguf_reader_callback_t callback = nullptr;
    void * userdata = nullptr;
    size_t max_chunk_read = 0;
    mutable uint64_t data_offset = 0;
    mutable uint64_t nbytes_remain = 0;
};

struct GGML_API_CLASS gguf_reader_impl_factory {
    virtual ~gguf_reader_impl_factory() = default;

    virtual std::unique_ptr<gguf_reader_impl> build_for(gguf_reader_callback_t callback,
                                                        void * userdata,
                                                        size_t max_chunk_read,
                                                        uint64_t data_offset = 0,
                                                        uint64_t nbytes_remain = 0,
                                                        const std::filesystem::path& file_path = {}) = 0;
};

struct GGML_API_CLASS gguf_reader
{
    gguf_reader(gguf_reader_callback_t callback,
                void * userdata,
                size_t max_chunk_read,
                uint64_t data_offset = 0,
                uint64_t nbytes_remain = 0,
                const std::filesystem::path& file_path = {});

    // helper for remaining bytes in a file
    static uint64_t file_remain(FILE * file);

    template <typename T>
    bool read(T & dst) const {
        const size_t size = sizeof(dst);
        if (size > impl->remaining()) {
            return false;
        }
        return read_raw(&dst, size) == size;
    }

    template <typename T>
    bool read(std::vector<T> & dst, const size_t n) const {
        if (n > GGUF_MAX_ARRAY_ELEMENTS) {
            return false;
        }
        if constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            if (n > SIZE_MAX / sizeof(uint64_t)) {
                return false;
            }
            if (impl->remaining() < n * sizeof(uint64_t)) {
                return false;
            }
        } else {
            if (n > SIZE_MAX / sizeof(T)) {
                return false;
            }
            if (impl->remaining() < n * sizeof(T)) {
                return false;
            }
        }
        dst.resize(n);
        for (size_t i = 0; i < dst.size(); ++i) {
            if constexpr (std::is_same<T, bool>::value) {
                bool tmp;
                if (!read(tmp)) {
                    return false;
                }
                dst[i] = tmp;
            } else {
                if (!read(dst[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool read(bool & dst) const;
    bool read(enum ggml_type & dst) const;
    bool read(enum gguf_type & dst) const;
    bool read(std::string & dst) const;
    bool read(void * dst, const size_t size) const;

    bool seek(uint64_t absolute_offset) const;

    uint64_t tell() const {
        return impl->tell();
    }

private:
    virtual size_t read_raw(void * dst, size_t size) const;

    std::unique_ptr<gguf_reader_impl> impl;
};
