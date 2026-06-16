#include "gl_util.hpp"

#include <cstdio>
#include <cstring>

namespace cc::viewer {

namespace {

GLuint compileStage(GLenum stage, const char* src, std::string& log) {
    GLuint sh = glCreateShader(stage);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = GL_FALSE;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (ok == GL_FALSE) {
        GLint len = 0;
        glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
        std::string buf(static_cast<std::size_t>(len > 0 ? len : 1), '\0');
        glGetShaderInfoLog(sh, len, nullptr, buf.data());
        log += buf;
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

// ---- Minimal PNG encoder (RGB8, stored/uncompressed deflate) --------------
// Self-contained so the viewer needs no zlib/stb for screenshots.

std::uint32_t crc32(const std::uint8_t* data, std::size_t n) {
    static std::uint32_t table[256];
    static bool          built = false;
    if (!built) {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        built = true;
    }
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; ++i) c = table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

std::uint32_t adler32(const std::uint8_t* data, std::size_t n) {
    std::uint32_t a = 1, b = 0;
    for (std::size_t i = 0; i < n; ++i) {
        a = (a + data[i]) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

void putU32(std::vector<std::uint8_t>& v, std::uint32_t x) {
    v.push_back(static_cast<std::uint8_t>(x >> 24));
    v.push_back(static_cast<std::uint8_t>(x >> 16));
    v.push_back(static_cast<std::uint8_t>(x >> 8));
    v.push_back(static_cast<std::uint8_t>(x));
}

void chunk(std::vector<std::uint8_t>& out, const char* type, const std::vector<std::uint8_t>& data) {
    putU32(out, static_cast<std::uint32_t>(data.size()));
    std::vector<std::uint8_t> crcbuf;
    crcbuf.insert(crcbuf.end(), type, type + 4);
    crcbuf.insert(crcbuf.end(), data.begin(), data.end());
    out.insert(out.end(), crcbuf.begin(), crcbuf.end());
    putU32(out, crc32(crcbuf.data(), crcbuf.size()));
}

bool writePng(const std::string& path, const std::uint8_t* rgba, int w, int h) {
    // Build the raw (filtered) scanlines: filter byte 0 + RGB per pixel.
    std::vector<std::uint8_t> raw;
    raw.reserve(static_cast<std::size_t>(h) * (1 + static_cast<std::size_t>(w) * 3));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        const std::uint8_t* row = rgba + static_cast<std::size_t>(y) * w * 4;
        for (int x = 0; x < w; ++x) {
            raw.push_back(row[x * 4 + 0]);
            raw.push_back(row[x * 4 + 1]);
            raw.push_back(row[x * 4 + 2]);
        }
    }

    // zlib stream with stored (uncompressed) deflate blocks.
    std::vector<std::uint8_t> z;
    z.push_back(0x78);
    z.push_back(0x01);
    std::size_t pos = 0;
    while (pos < raw.size()) {
        std::size_t   len   = std::min<std::size_t>(65535, raw.size() - pos);
        std::uint8_t  final = (pos + len >= raw.size()) ? 1 : 0;
        z.push_back(final);
        z.push_back(static_cast<std::uint8_t>(len & 0xFF));
        z.push_back(static_cast<std::uint8_t>((len >> 8) & 0xFF));
        std::uint16_t nlen = static_cast<std::uint16_t>(~len);
        z.push_back(static_cast<std::uint8_t>(nlen & 0xFF));
        z.push_back(static_cast<std::uint8_t>((nlen >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos),
                 raw.begin() + static_cast<std::ptrdiff_t>(pos + len));
        pos += len;
    }
    std::uint32_t ad = adler32(raw.data(), raw.size());
    z.push_back(static_cast<std::uint8_t>(ad >> 24));
    z.push_back(static_cast<std::uint8_t>(ad >> 16));
    z.push_back(static_cast<std::uint8_t>(ad >> 8));
    z.push_back(static_cast<std::uint8_t>(ad));

    std::vector<std::uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<std::uint8_t> ihdr;
    putU32(ihdr, static_cast<std::uint32_t>(w));
    putU32(ihdr, static_cast<std::uint32_t>(h));
    ihdr.push_back(8);  // bit depth
    ihdr.push_back(2);  // colour type RGB
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    chunk(out, "IHDR", ihdr);
    chunk(out, "IDAT", z);
    chunk(out, "IEND", {});

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
    return true;
}

bool writePpm(const std::string& path, const std::uint8_t* rgba, int w, int h) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* row = rgba + static_cast<std::size_t>(y) * w * 4;
        for (int x = 0; x < w; ++x) std::fwrite(row + x * 4, 1, 3, f);
    }
    std::fclose(f);
    return true;
}

}  // namespace

GLuint buildProgram(const char* vertexSrc, const char* fragmentSrc, std::string& log) {
    GLuint vs = compileStage(GL_VERTEX_SHADER, vertexSrc, log);
    if (!vs) return 0;
    GLuint fs = compileStage(GL_FRAGMENT_SHADER, fragmentSrc, log);
    if (!fs) {
        glDeleteShader(vs);
        return 0;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = GL_FALSE;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (ok == GL_FALSE) {
        GLint len = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string buf(static_cast<std::size_t>(len > 0 ? len : 1), '\0');
        glGetProgramInfoLog(prog, len, nullptr, buf.data());
        log += buf;
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

std::vector<std::uint8_t> readFramebuffer(int width, int height) {
    std::vector<std::uint8_t> px(static_cast<std::size_t>(width) * height * 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    // GL returns bottom-to-top; flip to top-to-bottom for image writers.
    std::vector<std::uint8_t> flipped(px.size());
    const std::size_t         rowBytes = static_cast<std::size_t>(width) * 4;
    for (int y = 0; y < height; ++y) {
        std::memcpy(flipped.data() + static_cast<std::size_t>(y) * rowBytes,
                    px.data() + static_cast<std::size_t>(height - 1 - y) * rowBytes, rowBytes);
    }
    return flipped;
}

bool writeImage(const std::string& path, const std::uint8_t* rgba, int width, int height) {
    const bool png = path.size() >= 4 && path.compare(path.size() - 4, 4, ".png") == 0;
    return png ? writePng(path, rgba, width, height) : writePpm(path, rgba, width, height);
}

}  // namespace cc::viewer
