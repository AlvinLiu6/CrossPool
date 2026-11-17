#pragma once

/// \file xpool/devkit/timeline/detail/wire.hpp
/// \brief Private fixed-width byte access shared by Event and chunk codecs.

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>

namespace xpool::devkit::timeline::detail {
/// \cond INTERNAL

// The codec owner establishes each complete fixed-width byte range first.
template <std::unsigned_integral T> T read_integer(std::span<const std::byte> bytes, std::size_t offset) noexcept {
  static_assert(sizeof(T) <= sizeof(std::uint64_t));
  auto value = std::uint64_t{0};
  for (auto index = std::size_t{0}; index < sizeof(T); ++index) {
    value |= std::to_integer<std::uint64_t>(bytes[offset + index]) << (index * 8);
  }
  return static_cast<T>(value);
}

template <std::unsigned_integral T>
void write_integer(std::span<std::byte> bytes, std::size_t offset, T value) noexcept {
  for (auto index = std::size_t{0}; index < sizeof(T); ++index) {
    bytes[offset + index] = static_cast<std::byte>((value >> (index * 8)) & T{255});
  }
}

/// \endcond
} // namespace xpool::devkit::timeline::detail
