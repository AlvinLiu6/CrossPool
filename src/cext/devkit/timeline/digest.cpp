#include <xpool/devkit/timeline/digest.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>

#include <xpool/devkit/timeline/error.hpp>

namespace xpool::devkit::timeline {

namespace {

// FIPS 180-4 section 4.2.2, first 32 fractional bits of prime cube roots.
constexpr auto constants = std::array<std::uint32_t, 64>{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

void compress(std::array<std::uint32_t, 8> &state, std::span<const std::byte, 64> block) {
  auto words = std::array<std::uint32_t, 64>{};
  for (auto index = std::size_t{0}; index < 16; ++index) {
    for (auto offset = std::size_t{0}; offset < 4; ++offset) {
      words[index] = (words[index] << 8) | std::to_integer<std::uint32_t>(block[index * 4 + offset]);
    }
  }
  for (auto index = std::size_t{16}; index < words.size(); ++index) {
    const auto first = words[index - 15];
    const auto second = words[index - 2];
    const auto sigma0 = std::rotr(first, 7) ^ std::rotr(first, 18) ^ (first >> 3);
    const auto sigma1 = std::rotr(second, 17) ^ std::rotr(second, 19) ^ (second >> 10);
    words[index] = words[index - 16] + sigma0 + words[index - 7] + sigma1;
  }
  auto [a, b, c, d, e, f, g, h] = state;
  for (auto index = std::size_t{0}; index < words.size(); ++index) {
    const auto upper = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
    const auto choose = (e & f) ^ (~e & g);
    const auto first = h + upper + choose + constants[index] + words[index];
    const auto lower = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
    const auto majority = (a & b) ^ (a & c) ^ (b & c);
    h = g;
    g = f;
    f = e;
    e = d + first;
    d = c;
    c = b;
    b = a;
    a = first + lower + majority;
  }
  const auto result = std::array{a, b, c, d, e, f, g, h};
  for (auto index = std::size_t{0}; index < state.size(); ++index) {
    state[index] += result[index];
  }
}

} // namespace

std::array<std::byte, 32> sha256(std::span<const std::byte> bytes) {
  auto digest = Sha256{};
  digest.update(bytes);
  return digest.finish();
}

void Sha256::update(std::span<const std::byte> bytes) {
  if (bytes.size() > std::numeric_limits<std::uint64_t>::max() / 8 - size_) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool SHA-256 input length overflows");
  }
  const auto used = static_cast<std::size_t>(size_ % 64);
  size_ += bytes.size();
  auto remaining = bytes;
  if (used != 0) {
    const auto count = std::min(remaining.size(), tail_.size() - used);
    std::copy_n(remaining.begin(), count, tail_.begin() + used);
    remaining = remaining.subspan(count);
    if (used + count == tail_.size()) {
      compress(state_, tail_);
    }
  }
  while (remaining.size() >= 64) {
    compress(state_, remaining.first<64>());
    remaining = remaining.subspan(64);
  }
  if (!remaining.empty()) {
    std::ranges::copy(remaining, tail_.begin());
  }
}

std::array<std::byte, 32> Sha256::finish() const {
  auto state = state_;
  const auto used = static_cast<std::size_t>(size_ % 64);
  auto tail = std::array<std::byte, 128>{};
  std::copy_n(tail_.begin(), used, tail.begin());
  tail[used] = std::byte{0x80};
  const auto tail_bytes = used < 56 ? std::size_t{64} : std::size_t{128};
  const auto bits = size_ * 8;
  for (auto index = std::size_t{0}; index < 8; ++index) {
    tail[tail_bytes - 1 - index] = static_cast<std::byte>((bits >> (index * 8)) & 0xff);
  }
  compress(state, std::span{tail}.first<64>());
  if (tail_bytes == 128) {
    compress(state, std::span{tail}.last<64>());
  }
  auto result = std::array<std::byte, 32>{};
  for (auto index = std::size_t{0}; index < state.size(); ++index) {
    for (auto offset = std::size_t{0}; offset < 4; ++offset) {
      result[index * 4 + offset] = static_cast<std::byte>((state[index] >> ((3 - offset) * 8)) & 0xff);
    }
  }
  return result;
}

} // namespace xpool::devkit::timeline
