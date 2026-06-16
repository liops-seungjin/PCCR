// Byte source/sink abstraction — the streaming + gzip seam (docs/design/01 §4).
// Readers/writers target these, never file paths, so a gzip decorator or a
// future chunked transport composes without touching the codecs.
#pragma once

#include <cstddef>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace cc::io {

struct IByteSource {
    virtual ~IByteSource()                          = default;
    virtual std::size_t read(std::span<std::byte> dst) = 0;  // bytes read; 0 at EOF
    [[nodiscard]] virtual bool eof() const          = 0;
};

struct IByteSink {
    virtual ~IByteSink()                                = default;
    virtual void write(std::span<const std::byte> src)  = 0;
    virtual void flush() {}
};

std::vector<std::byte> readAll(IByteSource& src);

class MemoryByteSource : public IByteSource {
public:
    explicit MemoryByteSource(std::vector<std::byte> data) : buf_(std::move(data)) {}
    std::size_t        read(std::span<std::byte> dst) override;
    [[nodiscard]] bool eof() const override { return pos_ >= buf_.size(); }

private:
    std::vector<std::byte> buf_;
    std::size_t            pos_ = 0;
};

class MemoryByteSink : public IByteSink {
public:
    void                                       write(std::span<const std::byte> src) override;
    [[nodiscard]] const std::vector<std::byte>& data() const { return buf_; }

private:
    std::vector<std::byte> buf_;
};

class FileByteSource : public IByteSource {
public:
    explicit FileByteSource(const std::string& path);
    std::size_t        read(std::span<std::byte> dst) override;
    [[nodiscard]] bool eof() const override;
    [[nodiscard]] bool ok() const { return in_.good() || in_.eof(); }

private:
    std::ifstream in_;
};

class FileByteSink : public IByteSink {
public:
    explicit FileByteSink(const std::string& path);
    void               write(std::span<const std::byte> src) override;
    void               flush() override { out_.flush(); }
    [[nodiscard]] bool ok() const { return out_.good(); }

private:
    std::ofstream out_;
};

}  // namespace cc::io
