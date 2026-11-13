#include <xpool/devkit/timeline/chunk.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include <xpool/devkit/timeline/detail/wire.hpp>
#include <xpool/devkit/timeline/digest.hpp>
#include <xpool/devkit/timeline/error.hpp>

namespace xpool::devkit::timeline {

namespace {

constexpr auto crc_table = [] {
  auto table = std::array<std::uint32_t, 256>{};
  for (auto index = std::size_t{0}; index < table.size(); ++index) {
    auto value = static_cast<std::uint32_t>(index);
    for (auto bit = 0; bit < 8; ++bit)
      value = (value >> 1) ^ ((value & 1U) != 0 ? std::uint32_t{0xedb88320} : std::uint32_t{0});
    table[index] = value;
  }
  return table;
}();

std::uint32_t crc32(std::span<const std::byte> bytes) noexcept {
  auto crc = std::numeric_limits<std::uint32_t>::max();
  for (const auto byte : bytes)
    crc = (crc >> 8) ^ crc_table[(crc ^ std::to_integer<std::uint32_t>(byte)) & 255U];
  return ~crc;
}

bool overlaps(std::span<const std::byte> left, std::span<const std::byte> right) noexcept {
  if (left.empty() || right.empty())
    return false;
  const auto first = reinterpret_cast<std::uintptr_t>(left.data());
  const auto second = reinterpret_cast<std::uintptr_t>(right.data());
  return first <= second ? second - first < left.size() : first - second < right.size();
}

} // namespace

std::size_t Chunk::required_bytes(std::size_t event_count) {
  if (event_count == 0 || event_count > maximum_events)
    throw std::invalid_argument("xpool chunk requires 1 through 4095 events");
  return header_bytes + event_count * Event::encoded_bytes;
}

std::span<const std::byte> Chunk::encode(std::span<std::byte> destination) const {
  // Preserve descriptor fields even when output aliases the original descriptor.
  const auto chunk = *this;
  const auto bytes = required_bytes(chunk.events.size());
  if (chunk.producer == 0 || chunk.chunk_sequence == 0 || chunk.metadata_revision == 0)
    throw std::invalid_argument("xpool chunk header identities must be positive");
  if (chunk.host_window_end_ns < chunk.host_window_begin_ns)
    throw std::invalid_argument("xpool chunk host collection window is reversed");
  if (destination.size() < bytes)
    throw std::invalid_argument("xpool chunk output capacity is insufficient");
  auto output = destination.first(bytes);
  if (overlaps(output, std::as_bytes(chunk.events)))
    throw std::invalid_argument("xpool chunk output overlaps input events");
  auto previous = std::uint64_t{0};
  for (const auto &event : chunk.events) {
    if (event.event_sequence <= previous)
      throw std::invalid_argument("xpool chunk event sequences must be positive and strictly increasing");
    event.validate(chunk.producer, chunk.events.front().clock_domain_id);
    previous = event.event_sequence;
  }

  auto header = output.first(header_bytes);
  auto payload = output.subspan(header_bytes);
  for (auto index = std::size_t{0}; index < chunk.events.size(); ++index)
    chunk.events[index].encode(
        payload.subspan(index * Event::encoded_bytes, Event::encoded_bytes).first<Event::encoded_bytes>());
  std::ranges::fill(header, std::byte{0});
  for (auto index = std::size_t{0}; index < kChunkMagic.size(); ++index)
    header[index] = static_cast<std::byte>(kChunkMagic[index]);
  detail::write_integer(header, 8, static_cast<std::uint16_t>(kContractVersion));
  detail::write_integer(header, 10, static_cast<std::uint16_t>(header_bytes));
  detail::write_integer(header, 12, static_cast<std::uint16_t>(Event::encoded_bytes));
  detail::write_integer(header, 16, chunk.producer);
  detail::write_integer(header, 24, chunk.chunk_sequence);
  detail::write_integer(header, 32, chunk.metadata_revision);
  detail::write_integer(header, 40, static_cast<std::uint64_t>(chunk.events.size()));
  detail::write_integer(header, 48, chunk.events.front().event_sequence);
  detail::write_integer(header, 56, chunk.events.back().event_sequence);
  detail::write_integer(header, 64, chunk.host_window_begin_ns);
  detail::write_integer(header, 72, chunk.host_window_end_ns);
  detail::write_integer(header, 80, static_cast<std::uint64_t>(payload.size()));
  detail::write_integer(header, 88, crc32(payload));
  std::ranges::copy(chunk.run_id, header.begin() + 96);
  std::ranges::copy(chunk.metadata_sha256_prefix, header.begin() + 112);
  detail::write_integer(header, 92, crc32(header));
  return output;
}

Chunk Chunk::decode(std::span<const std::byte> input, const ChunkMetadata &metadata, std::span<Event> destination,
                    std::optional<std::filesystem::path> path) {
  // Header identity context becomes available only after its layout/CRC checks.
  auto producer = std::uint64_t{0};
  const auto fail = [&](ParseErrorCode code, std::size_t offset, std::string_view reason,
                        std::optional<std::uint64_t> sequence = std::nullopt) {
    throw ParseError(code, std::string{"xpool "} + std::string{reason}, offset, producer, sequence, path,
                     std::string{reason});
  };
  if (input.size() < header_bytes)
    fail(ParseErrorCode::HeaderTruncated, input.size(), "truncated chunk header");
  for (auto index = std::size_t{0}; index < kChunkMagic.size(); ++index)
    if (input[index] != static_cast<std::byte>(kChunkMagic[index]))
      fail(ParseErrorCode::BadMagic, index, "invalid chunk signature");
  if (detail::read_integer<std::uint16_t>(input, 8) != kContractVersion)
    fail(ParseErrorCode::UnsupportedVersion, 8, "unsupported chunk version");
  if (detail::read_integer<std::uint16_t>(input, 10) != header_bytes)
    fail(ParseErrorCode::InvalidLayout, 10, "invalid header size");
  if (detail::read_integer<std::uint16_t>(input, 12) != Event::encoded_bytes)
    fail(ParseErrorCode::InvalidLayout, 12, "invalid event size");
  if (detail::read_integer<std::uint16_t>(input, 14) != 0)
    fail(ParseErrorCode::InvalidLayout, 14, "unknown chunk flags");
  auto header = std::array<std::byte, header_bytes>{};
  std::ranges::copy(input.first(header_bytes), header.begin());
  std::fill_n(header.begin() + 92, 4, std::byte{0});
  if (crc32(header) != detail::read_integer<std::uint32_t>(input, 92))
    fail(ParseErrorCode::HeaderChecksum, 92, "chunk header checksum mismatch");

  producer = detail::read_integer<std::uint64_t>(input, 16);
  const auto count = detail::read_integer<std::uint64_t>(input, 40);
  if (count == 0 || count > maximum_events)
    fail(ParseErrorCode::InvalidLength, 40, "invalid chunk event count");
  const auto event_count = static_cast<std::size_t>(count);
  const auto bytes = required_bytes(event_count);
  if (detail::read_integer<std::uint64_t>(input, 80) != count * Event::encoded_bytes || input.size() != bytes)
    fail(ParseErrorCode::InvalidLength, 80, "declared and actual chunk lengths differ");
  auto chunk = Chunk{.producer = producer,
                     .chunk_sequence = detail::read_integer<std::uint64_t>(input, 24),
                     .metadata_revision = detail::read_integer<std::uint64_t>(input, 32),
                     .host_window_begin_ns = detail::read_integer<std::uint64_t>(input, 64),
                     .host_window_end_ns = detail::read_integer<std::uint64_t>(input, 72)};
  if (producer == 0)
    fail(ParseErrorCode::InvalidLayout, 16, "invalid header Producer");
  if (chunk.chunk_sequence == 0)
    fail(ParseErrorCode::InvalidLayout, 24, "invalid chunk identity");
  if (chunk.metadata_revision == 0)
    fail(ParseErrorCode::InvalidLayout, 32, "invalid metadata revision");
  if (chunk.host_window_begin_ns > chunk.host_window_end_ns)
    fail(ParseErrorCode::InvalidLayout, 64, "reversed collection window");
  std::copy_n(input.begin() + 96, chunk.run_id.size(), chunk.run_id.begin());
  std::copy_n(input.begin() + 112, chunk.metadata_sha256_prefix.size(), chunk.metadata_sha256_prefix.begin());
  if (metadata.bytes.empty() || metadata.producer_id != producer || metadata.revision != chunk.metadata_revision ||
      metadata.run_id != chunk.run_id || metadata.clock_domain_id == 0)
    fail(ParseErrorCode::MetadataUnavailable, 32, "required metadata identity is unavailable");
  const auto digest = sha256(metadata.bytes);
  if (!std::equal(chunk.metadata_sha256_prefix.begin(), chunk.metadata_sha256_prefix.end(), digest.begin()))
    fail(ParseErrorCode::MetadataChecksum, 112, "metadata byte hash mismatch");
  const auto payload = input.subspan(header_bytes);
  if (crc32(payload) != detail::read_integer<std::uint32_t>(input, 88))
    fail(ParseErrorCode::PayloadChecksum, 88, "chunk payload checksum mismatch");

  auto previous = std::uint64_t{0};
  for (auto index = std::size_t{0}; index < event_count; ++index) {
    const auto offset = header_bytes + index * Event::encoded_bytes;
    const auto event = Event::decode(input.subspan(offset, Event::encoded_bytes).first<Event::encoded_bytes>());
    try {
      event.validate(producer, metadata.clock_domain_id);
    } catch (const ParseError &error) {
      fail(error.code(), offset + error.offset(), error.root_cause(), error.event_sequence());
    }
    if (event.event_sequence <= previous)
      fail(ParseErrorCode::InvalidEvent, offset, "chunk event order is not strictly increasing", event.event_sequence);
    if ((index == 0 && event.event_sequence != detail::read_integer<std::uint64_t>(input, 48)) ||
        (index + 1 == event_count && event.event_sequence != detail::read_integer<std::uint64_t>(input, 56)))
      fail(ParseErrorCode::InvalidEvent, offset, "header event range differs from payload", event.event_sequence);
    previous = event.event_sequence;
  }

  // Validate everything before materializing any Event in caller storage.
  if (destination.size() < event_count)
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool decoded event capacity is insufficient",
                          path);
  const auto written = destination.first(event_count);
  if (overlaps(std::as_bytes(written), input) || overlaps(std::as_bytes(written), metadata.bytes))
    throw std::invalid_argument("xpool decoded event destination overlaps input or metadata");
  for (auto index = std::size_t{0}; index < event_count; ++index)
    written[index] = Event::decode(
        payload.subspan(index * Event::encoded_bytes, Event::encoded_bytes).first<Event::encoded_bytes>());
  chunk.events = written;
  return chunk;
}

} // namespace xpool::devkit::timeline
