#include "gguf-reader-file.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <string>
#include <type_traits>

bool gguf_reader_file::close() {
    if (file) {
        fclose(file);

        file = nullptr;
    }

    return true;
}

bool gguf_reader_file::open(const std::filesystem::path & path) {
    close();

    file = ggml_fopen(path.generic_string().c_str(), "rb");

    if (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s'\n", __func__, path.c_str());

        return false;
    }

    return true;
}

gguf_reader_file::position_t gguf_reader_file::position() {
    if (!file) {
        return -1;
    }

    return ftell(file);
}

bool gguf_reader_file::position_set(position_t position) {
    if (!file || position < 0) {
        return false;
    }

    return fseek(file, position, SEEK_SET) == 0;
}

size_t gguf_reader_file::read(void * dst, const size_t size) {
    if (!file) {
        return 0U;
    }

    return fread(reinterpret_cast<char *>(dst), 1U, size, file);
}
