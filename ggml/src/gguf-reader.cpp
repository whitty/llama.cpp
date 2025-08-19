#include "gguf-reader.h"

#include "gguf.h"
#include "gguf-reader-file.h"

#include <cstddef>
#include <memory>

struct default_impl_factory : public gguf_reader_impl_factory {
    std::unique_ptr<gguf_reader_impl> build_for(const std::filesystem::path &) override {
        return std::make_unique<gguf_reader_file>();
    }
};

static default_impl_factory default_factory;

static gguf_reader_impl_factory * impl_factory = &default_factory;

gguf_reader::gguf_reader(const std::filesystem::path & path)
    : m_path(path), m_impl(impl_factory->build_for(path)) {
}

bool gguf_reader::read(std::string & dst) {
    uint64_t size;

    if (!read(size)) {
        return false;
    }

    dst.resize(size);

    return m_impl->read(dst.data(), dst.length()) == dst.length();
}

bool gguf_reader::read(bool & dst) {
    int8_t tmp;

    if (!read(tmp)) {
        return false;
    }

    dst = tmp != 0;

    return true;
}

bool gguf_reader::read(enum ggml_type & dst) {
    int32_t tmp;

    if (!read(tmp)) {
        return false;
    }

    dst = ggml_type(tmp);

    return true;
}

bool gguf_reader::read(enum gguf_type & dst) {
    int32_t tmp;

    if (!read(tmp)) {
        return false;
    }

    dst = gguf_type(tmp);

    return true;
}

void gguf_set_default_reader_impl(struct gguf_reader_impl_factory * factory) {
    impl_factory = factory ? factory : &default_factory;
}
