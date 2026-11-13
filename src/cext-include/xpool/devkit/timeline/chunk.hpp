#pragma once

/// \file xpool/devkit/timeline/chunk.hpp
/// \brief Bounded v4 byte codecs over caller-owned Event and metadata storage.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>

#include <xpool/devkit/timeline/contract.hpp>
#include <xpool/devkit/timeline/event.hpp>

namespace xpool::devkit::timeline {

/// Borrowed exact bytes and source identity from the metadata validation owner.
/// The caller validates JSON schema, membership, source ownership and references
/// before constructing this value. The byte codec verifies the declared header
/// identity/hash against it; this aggregate is not a metadata JSON parser.
struct ChunkMetadata {
  /// Run UUID in canonical opaque byte order.
  std::array<std::byte, 16> run_id{};
  /// Positive registered Producer in those exact metadata bytes.
  std::uint64_t producer_id = 0;
  /// Positive immutable metadata revision.
  std::uint64_t revision = 0;
  /// This Producer's single registered clock domain.
  std::uint64_t clock_domain_id = 0;
  /// Exact UTF-8 bytes, including the final newline; stable through decode.
  std::span<const std::byte> bytes{};
};

/// Non-owning complete chunk descriptor. Events share the registered Producer
/// and remain immutable throughout encoding. Decoded Events borrow the supplied
/// output storage; neither this value nor the codec creates a payload allocation.
struct Chunk {
  /// Frozen byte size of the raw v4 header.
  static constexpr std::size_t header_bytes = 128;
  /// Maximum complete file size, including its header.
  static constexpr std::size_t maximum_bytes = 1048576;
  /// Largest nonempty batch fitting the complete-file limit.
  static constexpr std::size_t maximum_events = (maximum_bytes - header_bytes) / Event::encoded_bytes;

  /// Registered source shared by every Event.
  std::uint64_t producer = 0;
  /// Positive chunk publication identity; late older Event sequences are legal.
  std::uint64_t chunk_sequence = 0;
  /// Immutable metadata revision that resolves all referenced identities.
  std::uint64_t metadata_revision = 0;
  /// CLOCK_MONOTONIC collection window start, not Device event time.
  std::uint64_t host_window_begin_ns = 0;
  /// Collection window end, at least its start.
  std::uint64_t host_window_end_ns = 0;
  /// Run UUID bytes in canonical order.
  std::array<std::byte, 16> run_id{};
  /// First 16 SHA-256 bytes of the metadata file's exact bytes.
  std::array<std::byte, 16> metadata_sha256_prefix{};
  /// Strictly increasing positive Event sequences; holes remain legal.
  std::span<const Event> events{};

  /// Compute complete bytes for 1..4095 Events, checking count before arithmetic.
  /// \throws std::invalid_argument on empty or oversized input.
  static std::size_t required_bytes(std::size_t event_count);

  /// Encode explicit little-endian fields and CRC-32/ISO-HDLC into destination.
  /// Validate before writing; return only the written prefix. Destination must
  /// not overlap Event input. Native structs are never copied as wire images.
  /// Metadata reference semantics are the caller's preparation precondition.
  /// \throws std::invalid_argument for invalid count/header/window, ordering,
  /// insufficient output or overlap; ParseError for intrinsic Event failures.
  std::span<const std::byte> encode(std::span<std::byte> destination) const;

  /// Decode a complete file image into bounded caller-owned Event storage.
  /// Validate layout, header CRC, length, metadata identity/hash, payload CRC,
  /// intrinsic Event facts and ordering, in that order. Metadata JSON/membership
  /// and cross-Event reference validation belong to the caller's metadata owner;
  /// a successful byte decode alone is not complete strict-parser acceptance.
  /// No destination Event is changed on any validation or capacity failure.
  /// \throws ParseError with C0 code and file-relative byte/source context;
  /// CollectionError resource_budget for insufficient Event storage;
  /// std::invalid_argument when destination overlaps input/metadata bytes.
  static Chunk decode(std::span<const std::byte> input, const ChunkMetadata &metadata, std::span<Event> destination,
                      std::optional<std::filesystem::path> path = std::nullopt);
};

} // namespace xpool::devkit::timeline
