#include "gguf-reader.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "gguf.h"

#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

gguf_reader_impl::gguf_reader_impl(gguf_reader_callback_t callback,
                                   void * userdata,
                                   size_t max_chunk_read,
                                   uint64_t data_offset,
                                   uint64_t nbytes_remain)
:   callback(callback),
    userdata(userdata),
    max_chunk_read(max_chunk_read),
    data_offset(data_offset),
    nbytes_remain(nbytes_remain)
{
}

size_t gguf_reader_impl::read_raw(void * dst, size_t size) {
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

bool gguf_reader_impl::seek(uint64_t absolute_offset) {
    const uint64_t end_offset = uint64_t(data_offset) + nbytes_remain;
    if (absolute_offset > end_offset) {
        return false;
    }

    data_offset = absolute_offset;
    nbytes_remain = end_offset - absolute_offset;

    return true;
}

struct default_impl_factory : public gguf_reader_impl_factory {
    std::unique_ptr<gguf_reader_impl> build_for(gguf_reader_callback_t callback,
                                                void * userdata,
                                                size_t max_chunk_read,
                                                uint64_t data_offset,
                                                uint64_t nbytes_remain) override {
        return std::make_unique<gguf_reader_impl>(callback, userdata, max_chunk_read, data_offset, nbytes_remain);
    }
};

static default_impl_factory default_factory;

static gguf_reader_impl_factory * impl_factory = &default_factory;

gguf_reader::gguf_reader(gguf_reader_callback_t callback,
                         void * userdata,
                         size_t max_chunk_read,
                         uint64_t data_offset,
                         uint64_t nbytes_remain)
:   impl(impl_factory->build_for(callback, userdata, max_chunk_read, data_offset, nbytes_remain))
{
    GGML_ASSERT(max_chunk_read > 0);
}

uint64_t gguf_reader::file_remain(FILE * file) {
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

bool gguf_reader::read(bool & dst) {
    int8_t tmp = -1;
    if (!read(tmp)) {
        return false;
    }
    dst = tmp != 0;
    return true;
}

bool gguf_reader::read(enum ggml_type & dst) {
    int32_t tmp = -1;
    if (!read(tmp)) {
        return false;
    }
    dst = ggml_type(tmp);
    return true;
}

bool gguf_reader::read(enum gguf_type & dst) {
    int32_t tmp = -1;
    if (!read(tmp)) {
        return false;
    }
    dst = gguf_type(tmp);
    return true;
}

bool gguf_reader::read(std::string & dst) {
    uint64_t size = 0;
    if (!read(size)) {
        return false;
    }
    if (size > GGUF_MAX_STRING_LENGTH) {
        GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_LENGTH);
        return false;
    }
    if (size > impl->remaining()) {
        GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, impl->remaining());
        return false;
    }
    dst.resize(static_cast<size_t>(size));
    return read_raw(dst.data(), static_cast<size_t>(size)) == size;
}

bool gguf_reader::read(void * dst, const size_t size) {
    if (size > impl->remaining()) {
        return false;
    }
    return read_raw(dst, size) == size;
}

size_t gguf_reader::read_raw(void * dst, size_t size) {
    return impl->read_raw(dst, size);
}

bool gguf_reader::seek(uint64_t absolute_offset) {
    return impl->seek(absolute_offset);
}

void gguf_set_default_reader_impl(struct gguf_reader_impl_factory * factory) {
    impl_factory = factory ? factory : &default_factory;
}
