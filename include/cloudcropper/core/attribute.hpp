// AttributeColumn — one named, typed, optional point attribute (SoA column).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace cc {

enum class AttrType : std::uint8_t {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
};

constexpr std::size_t attrTypeSize(AttrType t) {
    switch (t) {
        case AttrType::I8:
        case AttrType::U8:
            return 1;
        case AttrType::I16:
        case AttrType::U16:
            return 2;
        case AttrType::I32:
        case AttrType::U32:
        case AttrType::F32:
            return 4;
        case AttrType::I64:
        case AttrType::U64:
        case AttrType::F64:
            return 8;
    }
    return 0;
}

// A column holds `size()` rows, each `arity()` components wide, tightly packed.
class AttributeColumn {
public:
    AttributeColumn() = default;
    AttributeColumn(std::string name, AttrType type, std::uint8_t arity, std::size_t count)
        : name_(std::move(name)),
          type_(type),
          arity_(arity),
          data_(count * arity * attrTypeSize(type)) {}

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] AttrType            type() const { return type_; }
    [[nodiscard]] std::uint8_t        arity() const { return arity_; }
    [[nodiscard]] std::size_t         stride() const { return arity_ * attrTypeSize(type_); }
    [[nodiscard]] std::size_t         size() const {
        const std::size_t s = stride();
        return s ? data_.size() / s : 0;
    }

    [[nodiscard]] std::span<std::byte>       bytes() { return data_; }
    [[nodiscard]] std::span<const std::byte> bytes() const { return data_; }

    template <class T>
    [[nodiscard]] std::span<T> as() {
        return {reinterpret_cast<T*>(data_.data()), data_.size() / sizeof(T)};
    }
    template <class T>
    [[nodiscard]] std::span<const T> as() const {
        return {reinterpret_cast<const T*>(data_.data()), data_.size() / sizeof(T)};
    }

    [[nodiscard]] std::vector<std::byte>&       raw() { return data_; }
    [[nodiscard]] const std::vector<std::byte>& raw() const { return data_; }

private:
    std::string            name_;
    AttrType               type_  = AttrType::F32;
    std::uint8_t           arity_ = 1;
    std::vector<std::byte> data_;
};

// Read/write one scalar slot (element index = row*arity + component) as double.
// Used by codecs and the gather path; keeps type dispatch in one place.
double readScalar(const AttributeColumn& col, std::size_t element);
void   writeScalar(AttributeColumn& col, std::size_t element, double value);

}  // namespace cc
