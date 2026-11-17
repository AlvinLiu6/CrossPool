#pragma once

/// \file xpool/devkit/timeline/error.hpp
/// \brief Structured C0 collection and artifact failures, independent of text.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include <c10/util/Exception.h>

namespace xpool::devkit::timeline {

/// Frozen collection failure values from the Native C0 boundary.
enum class CollectionErrorCode : std::uint32_t {
  ResourceBudget = 1,
  Allocation = 2,
  Transfer = 3,
  InvalidRelease = 4,
  Io = 5,
  MetadataUnavailable = 6,
};

/// Return the exact C0 lowercase name; invalid values return an empty view.
std::string_view collection_error_name(CollectionErrorCode code) noexcept;

/// Host collection failure. Recovery branches on code, not exception text.
class CollectionError : public c10::Error {
public:
  /// Retain the failing source, optional filesystem path and underlying cause.
  CollectionError(CollectionErrorCode code, std::uint64_t producer, std::string message,
                  std::optional<std::filesystem::path> path = std::nullopt, std::string root_cause = {});
  /// C0 category used for structured recovery.
  CollectionErrorCode code() const noexcept { return code_; }
  /// Source involved, or zero when no source has been admitted yet.
  std::uint64_t producer() const noexcept { return producer_; }
  /// Associated filesystem object, absent for memory-only failures.
  const std::optional<std::filesystem::path> &path() const noexcept { return path_; }
  /// Original cause retained without message-based classification.
  const std::string &root_cause() const noexcept { return root_cause_; }

private:
  CollectionErrorCode code_;
  std::uint64_t producer_;
  std::optional<std::filesystem::path> path_;
  std::string root_cause_;
};

/// Artifact validation categories. These ordinals are internal, never wire.
enum class ParseErrorCode {
  HeaderTruncated,
  BadMagic,
  UnsupportedVersion,
  InvalidLayout,
  HeaderChecksum,
  InvalidLength,
  MetadataUnavailable,
  MetadataChecksum,
  PayloadChecksum,
  InvalidEvent,
  UnknownKind,
  IdentityConflict,
  EventConflict,
};

/// Return the exact C0 error spelling; invalid values return an empty view.
std::string_view parse_error_name(ParseErrorCode code) noexcept;

/// Structured malformed-artifact error; legal missing evidence is not an error.
class ParseError : public c10::Error {
public:
  /// Preserve byte offset and available source/event/path context.
  /// Event::validate offsets are record-relative; Chunk::decode makes them
  /// absolute within the supplied complete file image.
  ParseError(ParseErrorCode code, std::string message, std::size_t offset = 0, std::uint64_t producer = 0,
             std::optional<std::uint64_t> event_sequence = std::nullopt,
             std::optional<std::filesystem::path> path = std::nullopt, std::string root_cause = {});
  /// Exact C0 parse failure category.
  ParseErrorCode code() const noexcept { return code_; }
  /// Byte offset in the owning input boundary.
  std::size_t offset() const noexcept { return offset_; }
  /// Header/source Producer, or zero when not yet available.
  std::uint64_t producer() const noexcept { return producer_; }
  /// Offending Event identity when decoded far enough to know it.
  const std::optional<std::uint64_t> &event_sequence() const noexcept { return event_sequence_; }
  /// Artifact path, absent for an unnamed in-memory input.
  const std::optional<std::filesystem::path> &path() const noexcept { return path_; }
  /// Underlying structural reason, preserved when adding file context.
  const std::string &root_cause() const noexcept { return root_cause_; }

private:
  ParseErrorCode code_;
  std::size_t offset_;
  std::uint64_t producer_;
  std::optional<std::uint64_t> event_sequence_;
  std::optional<std::filesystem::path> path_;
  std::string root_cause_;
};

} // namespace xpool::devkit::timeline
