#include "cloudcropper/io/byte_stream.hpp"

#include <algorithm>
#include <cstring>

namespace cc::io {

std::vector<std::byte> readAll(IByteSource& src) {
    std::vector<std::byte> out;
    std::byte              chunk[64 * 1024];
    while (!src.eof()) {
        const std::size_t n = src.read({chunk, sizeof(chunk)});
        if (n == 0) break;
        out.insert(out.end(), chunk, chunk + n);
    }
    return out;
}

std::size_t MemoryByteSource::read(std::span<std::byte> dst) {
    const std::size_t n = std::min(dst.size(), buf_.size() - pos_);
    if (n) std::memcpy(dst.data(), buf_.data() + pos_, n);
    pos_ += n;
    return n;
}

void MemoryByteSink::write(std::span<const std::byte> src) {
    buf_.insert(buf_.end(), src.begin(), src.end());
}

FileByteSource::FileByteSource(const std::string& path)
    : in_(path, std::ios::binary) {}

std::size_t FileByteSource::read(std::span<std::byte> dst) {
    in_.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(dst.size()));
    return static_cast<std::size_t>(in_.gcount());
}

bool FileByteSource::eof() const { return in_.eof() || !in_.good(); }

FileByteSink::FileByteSink(const std::string& path)
    : out_(path, std::ios::binary) {}

void FileByteSink::write(std::span<const std::byte> src) {
    out_.write(reinterpret_cast<const char*>(src.data()), static_cast<std::streamsize>(src.size()));
}

}  // namespace cc::io
