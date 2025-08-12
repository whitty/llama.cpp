#pragma once

#include "ggml.h"

#include "gguf-reader.h"

#include <cstddef>
#include <fstream>

struct GGML_API_CLASS gguf_reader_file final : public gguf_reader_impl {
    gguf_reader_file() = default;

    ~gguf_reader_file() override {
        close();
    }

    bool close() override;
    bool open(const std::filesystem::path & path) override;
    position_t position() override;
    bool position_set(position_t position) override;
    size_t read(void * dst, size_t size) override;

private:
    std::ifstream m_file;
};
