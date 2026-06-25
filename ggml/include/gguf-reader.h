#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "gguf.h"

#include <cinttypes>
#include <cstddef>
#include <cstdio>
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

struct GGML_API_CLASS gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            void * userdata,
            size_t max_chunk_read,
            uint64_t data_offset = 0,
            uint64_t nbytes_remain = 0)
        : callback(callback),
          userdata(userdata),
          max_chunk_read(max_chunk_read),
          data_offset(data_offset),
          nbytes_remain(nbytes_remain) {
        GGML_ASSERT(max_chunk_read > 0);
    }

    // helper for remaining bytes in a file
    static uint64_t file_remain(FILE * file) {
        const int64_t cur = gguf_ftell(file);
        if (cur < 0) {
            return 0;
        }
        if (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            return 0;
        }
        const int64_t end = gguf_ftell(file);
        if (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            return 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        return static_cast<uint64_t>(end - cur);
    }

    template <typename T>
    bool read(T & dst) const {
        const size_t size = sizeof(dst);
        if (size > nbytes_remain) {
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
            if (nbytes_remain < n * sizeof(uint64_t)) {
                return false;
            }
        } else {
            if (n > SIZE_MAX / sizeof(T)) {
                return false;
            }
            if (nbytes_remain < n * sizeof(T)) {
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

    bool read(bool & dst) const {
        int8_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = tmp != 0;
        return true;
    }

    bool read(enum ggml_type & dst) const {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = ggml_type(tmp);
        return true;
    }

    bool read(enum gguf_type & dst) const {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = gguf_type(tmp);
        return true;
    }

    bool read(std::string & dst) const {
        uint64_t size = 0;
        if (!read(size)) {
            return false;
        }
        if (size > GGUF_MAX_STRING_LENGTH) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_LENGTH);
            return false;
        }
        if (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            return false;
        }
        dst.resize(static_cast<size_t>(size));
        return read_raw(dst.data(), static_cast<size_t>(size)) == size;
    }

    bool read(void * dst, const size_t size) const {
        if (size > nbytes_remain) {
            return false;
        }
        return read_raw(dst, size) == size;
    }

    uint64_t tell() const {
        return data_offset;
    }

    bool seek(uint64_t absolute_offset) const {
        const uint64_t end_offset = uint64_t(data_offset) + nbytes_remain;
        if (absolute_offset > end_offset) {
            return false;
        }

        data_offset = absolute_offset;
        nbytes_remain = end_offset - absolute_offset;

        return true;
    }

private:
    size_t read_raw(void * dst, size_t size) const {
        if (callback == nullptr || size == 0) {
            return 0;
        }

        uint8_t * data = static_cast<uint8_t *>(dst);
        size_t total_nread = 0;
        bool reached_eof = false;

        while (total_nread < size) {
            const size_t chunk_size = std::min(max_chunk_read, size - total_nread);
            if (data_offset + total_nread < data_offset) {
                break;
            }
            const size_t nread = callback(userdata, static_cast<void *>(data + total_nread), data_offset + total_nread, chunk_size);
            total_nread += nread;
            if (nread != chunk_size) {
                reached_eof = true;
                break;
            }
        }

        data_offset += total_nread;
        GGML_ASSERT(total_nread <= nbytes_remain);
        nbytes_remain -= total_nread;

        if (reached_eof) {
            nbytes_remain = 0;
        }

        return total_nread;
    }

    gguf_reader_callback_t callback = nullptr;
    void * userdata = nullptr;
    size_t max_chunk_read = 0;
    mutable uint64_t data_offset = 0;
    mutable uint64_t nbytes_remain = 0;
};
