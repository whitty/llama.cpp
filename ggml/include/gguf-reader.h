#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstddef>
#include <ios>
#include <string>
#include <vector>
#include <filesystem>

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

struct GGML_API_CLASS gguf_reader_impl {
    using position_t = int64_t;

    virtual ~gguf_reader_impl() = default;

    virtual bool close() = 0;
    virtual bool open(const std::filesystem::path & file) = 0;
    virtual position_t position() = 0;
    virtual bool position_set(position_t position) = 0;
    virtual size_t read(void * buf, size_t len) = 0;
};

struct GGML_API_CLASS gguf_reader_impl_factory {
    virtual ~gguf_reader_impl_factory() = default;

    virtual std::unique_ptr<gguf_reader_impl> build_for(const std::filesystem::path & file) = 0;
};

struct GGML_API_CLASS gguf_reader {
    using position_t = gguf_reader_impl::position_t;

    // Create a reader using the currently-configured default implementation (see
    // gguf_set_default_reader_impl())
    gguf_reader(const std::filesystem::path & path);

    // Create a reader using the implementation given by impl
    gguf_reader(const std::filesystem::path & path, std::unique_ptr<gguf_reader_impl> && impl)
        : m_path(path), m_impl(std::move(impl)) {
    }

    ~gguf_reader() {
        m_impl->close();
    }

    bool open() {
        return m_impl->open(m_path);
    }

    position_t position() {
        return m_impl->position();
    }

    bool position_set(position_t position) {
        return m_impl->position_set(position);
    }

    bool read(bool & dst);
    bool read(enum ggml_type & dst);
    bool read(enum gguf_type & dst);
    bool read(std::string & dst);

    size_t read(void * dst, size_t size) {
        return m_impl->read(dst, size);
    }

    template <typename T>
    bool read(T & dst) {
        return m_impl->read(&dst, sizeof(dst)) == sizeof(dst);
    }

    template <typename T>
    bool read(std::vector<T> & dst, const size_t n) {
        dst.resize(n);

        for (size_t i = 0U; i < dst.size(); ++i) {
            if constexpr (std::is_same<T, bool>::value) {
                bool tmp;

                if (!read(tmp)) {
                    return false;
                }

                dst[i] = tmp;
            } else if (!read(dst[i])) {
                return false;
            }
        }

        return true;
    }

private:
    std::filesystem::path m_path;
    std::unique_ptr<gguf_reader_impl> m_impl;
};
