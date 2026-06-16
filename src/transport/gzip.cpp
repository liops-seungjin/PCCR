#include "cloudcropper/transport/gzip.hpp"

#include <cstring>

#include <zlib.h>

namespace cc::transport {
namespace {

constexpr int kGzipWindowBits = 15 + 16;  // 15 = max window, +16 = gzip header/trailer

// avail_in/avail_out are uInt (32-bit); feed the input in <=chunk slices so
// buffers larger than 4 GiB still work.
constexpr std::size_t kChunk = 1u << 20;  // 1 MiB

}  // namespace

Result<std::vector<std::byte>> gzipCompress(std::span<const std::byte> in, int level) {
    z_stream zs;
    std::memset(&zs, 0, sizeof(zs));
    if (deflateInit2(&zs, level, Z_DEFLATED, kGzipWindowBits, 8, Z_DEFAULT_STRATEGY) != Z_OK)
        return makeError(ErrorCode::IoError, "gzip: deflateInit2 failed");

    std::vector<std::byte> out;
    out.reserve(in.size() / 2 + 64);
    std::byte         buf[kChunk];
    const std::byte*  ip  = in.data();
    std::size_t       rem = in.size();
    int               ret = Z_OK;
    do {
        const std::size_t take = rem < kChunk ? rem : kChunk;
        zs.next_in  = reinterpret_cast<Bytef*>(const_cast<std::byte*>(ip));
        zs.avail_in = static_cast<uInt>(take);
        const bool last = (rem - take) == 0;
        do {
            zs.next_out  = reinterpret_cast<Bytef*>(buf);
            zs.avail_out = static_cast<uInt>(kChunk);
            ret          = deflate(&zs, last ? Z_FINISH : Z_NO_FLUSH);
            out.insert(out.end(), buf, buf + (kChunk - zs.avail_out));
        } while (zs.avail_out == 0);
        ip += take;
        rem -= take;
    } while (rem > 0 && ret == Z_OK);

    deflateEnd(&zs);
    if (ret != Z_STREAM_END) return makeError(ErrorCode::IoError, "gzip: deflate failed");
    return out;
}

Result<std::vector<std::byte>> gzipDecompress(std::span<const std::byte> in) {
    z_stream zs;
    std::memset(&zs, 0, sizeof(zs));
    if (inflateInit2(&zs, kGzipWindowBits) != Z_OK)
        return makeError(ErrorCode::IoError, "gzip: inflateInit2 failed");

    std::vector<std::byte> out;
    out.reserve(in.size() * 3 + 64);
    std::byte        buf[kChunk];
    const std::byte* ip  = in.data();
    std::size_t      rem = in.size();
    int              ret = Z_OK;
    do {
        const std::size_t take = rem < kChunk ? rem : kChunk;
        zs.next_in  = reinterpret_cast<Bytef*>(const_cast<std::byte*>(ip));
        zs.avail_in = static_cast<uInt>(take);
        do {
            zs.next_out  = reinterpret_cast<Bytef*>(buf);
            zs.avail_out = static_cast<uInt>(kChunk);
            ret          = inflate(&zs, Z_NO_FLUSH);
            if (ret != Z_OK && ret != Z_STREAM_END && ret != Z_BUF_ERROR) {
                inflateEnd(&zs);
                return makeError(ErrorCode::ParseError, "gzip: inflate failed (corrupt stream?)");
            }
            out.insert(out.end(), buf, buf + (kChunk - zs.avail_out));
        } while (zs.avail_out == 0 && ret != Z_STREAM_END);
        ip += take;
        rem -= take;
    } while (rem > 0 && ret != Z_STREAM_END);

    inflateEnd(&zs);
    if (ret != Z_STREAM_END) return makeError(ErrorCode::ParseError, "gzip: truncated stream");
    return out;
}

}  // namespace cc::transport
