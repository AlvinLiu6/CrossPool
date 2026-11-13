#pragma once

/// \file xpool/devkit/timeline/digest.hpp
/// \brief Allocation-free SHA-256 for exact immutable artifact bytes.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace xpool::devkit::timeline {

/// Incremental SHA-256 with fixed storage. Supports streaming immutable artifact
/// bytes before disk admission without retaining a second complete byte image.
/// finish() hashes a copy, so repeated reads and further updates remain valid.
class Sha256 {
public:
  /// Consume exact bytes in order, including an artifact's final newline.
  /// \throws CollectionError if total input exceeds the SHA-256 bit length.
  void update(std::span<const std::byte> bytes);
  /// Return the canonical digest without changing this stream.
  std::array<std::byte, 32> finish() const;
  /// Exact consumed byte count, suitable for pre-creation disk admission.
  std::uint64_t size() const noexcept { return size_; }

private:
  std::array<std::uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<std::byte, 64> tail_{};
  std::uint64_t size_ = 0;
};

/// SHA-256 digest in canonical byte order over exactly the supplied bytes.
/// Input remains stable during this bounded-memory operation; metadata callers
/// include its UTF-8 final newline. Uses FIPS 180-4 sections 5 and 6.2, with no
/// mutable global state or third-party runtime initialization.
/// \throws CollectionError with resource_budget for input beyond SHA-256's
/// representable bit length.
std::array<std::byte, 32> sha256(std::span<const std::byte> bytes);

} // namespace xpool::devkit::timeline
