#include "gguf-reader-file.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <ios>
#include <string>
#include <type_traits>

bool gguf_reader_file::close() {
    if (m_file.is_open()) {
        m_file.close();
    }

    return true;
}

bool gguf_reader_file::open(const std::filesystem::path & path) {
    close();

    m_file.open(path, std::ios_base::in | std::ios_base::binary);

    if (!(m_file.is_open() && m_file.good())) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s'\n", __func__, path.c_str());

        return false;
    }

    return true;
}

gguf_reader_file::position_t gguf_reader_file::position() {
    std::streampos position = m_file.tellg();

    if (position == std::streampos(-1)) {
        return -1;
    }

    return position;
}

bool gguf_reader_file::position_set(position_t position) {
    if (!m_file.is_open()) {
        return false;
    }

    m_file.seekg(position, std::ios_base::beg);

    return m_file.good();
}

size_t gguf_reader_file::read(void * dst, const size_t size) {
    if (!m_file.is_open()) {
        return 0U;
    }

    size_t bytesRead = 0U;

    while (m_file.good() && (bytesRead < size)) {
        // std::ifstream::read takes length as a std::streamsize, which may have a lower maximum
        // value than the size_t we take, so split long reads up if needed
        auto bytesToRead = std::min(std::make_unsigned_t<std::streamsize>(std::numeric_limits<std::streamsize>::max()),
                                    size - bytesRead);

        m_file.read(reinterpret_cast<char *>(dst), bytesToRead);

        bytesRead += static_cast<size_t>(m_file.gcount());
    }

    return bytesRead;
}
