#include "cloudcropper/core/attribute.hpp"

#include <cstring>

namespace cc {
namespace {

template <class T>
double loadAs(const std::byte* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return static_cast<double>(v);
}

template <class T>
void storeAs(std::byte* p, double value) {
    T v = static_cast<T>(value);
    std::memcpy(p, &v, sizeof(T));
}

}  // namespace

double readScalar(const AttributeColumn& col, std::size_t element) {
    const std::byte*  base = col.bytes().data();
    const std::size_t off  = element * attrTypeSize(col.type());
    const std::byte*  p    = base + off;
    switch (col.type()) {
        case AttrType::I8:  return loadAs<std::int8_t>(p);
        case AttrType::U8:  return loadAs<std::uint8_t>(p);
        case AttrType::I16: return loadAs<std::int16_t>(p);
        case AttrType::U16: return loadAs<std::uint16_t>(p);
        case AttrType::I32: return loadAs<std::int32_t>(p);
        case AttrType::U32: return loadAs<std::uint32_t>(p);
        case AttrType::I64: return loadAs<std::int64_t>(p);
        case AttrType::U64: return loadAs<std::uint64_t>(p);
        case AttrType::F32: return loadAs<float>(p);
        case AttrType::F64: return loadAs<double>(p);
    }
    return 0.0;
}

void writeScalar(AttributeColumn& col, std::size_t element, double value) {
    std::byte*        base = col.bytes().data();
    const std::size_t off  = element * attrTypeSize(col.type());
    std::byte*        p    = base + off;
    switch (col.type()) {
        case AttrType::I8:  storeAs<std::int8_t>(p, value); break;
        case AttrType::U8:  storeAs<std::uint8_t>(p, value); break;
        case AttrType::I16: storeAs<std::int16_t>(p, value); break;
        case AttrType::U16: storeAs<std::uint16_t>(p, value); break;
        case AttrType::I32: storeAs<std::int32_t>(p, value); break;
        case AttrType::U32: storeAs<std::uint32_t>(p, value); break;
        case AttrType::I64: storeAs<std::int64_t>(p, value); break;
        case AttrType::U64: storeAs<std::uint64_t>(p, value); break;
        case AttrType::F32: storeAs<float>(p, value); break;
        case AttrType::F64: storeAs<double>(p, value); break;
    }
}

}  // namespace cc
