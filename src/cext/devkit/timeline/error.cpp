#include <xpool/devkit/timeline/error.hpp>

#include <utility>

namespace xpool::devkit::timeline {

std::string_view collection_error_name(CollectionErrorCode code) noexcept {
  switch (code) {
  case CollectionErrorCode::ResourceBudget:
    return "resource_budget";
  case CollectionErrorCode::Allocation:
    return "allocation";
  case CollectionErrorCode::Transfer:
    return "transfer";
  case CollectionErrorCode::InvalidRelease:
    return "invalid_release";
  case CollectionErrorCode::Io:
    return "io";
  case CollectionErrorCode::MetadataUnavailable:
    return "metadata_unavailable";
  }
  return {};
}

CollectionError::CollectionError(CollectionErrorCode code, std::uint64_t producer, std::string message,
                                 std::optional<std::filesystem::path> path, std::string root_cause)
    : c10::Error(std::move(message)), code_(code), producer_(producer), path_(std::move(path)),
      root_cause_(std::move(root_cause)) {}

std::string_view parse_error_name(ParseErrorCode code) noexcept {
  switch (code) {
  case ParseErrorCode::HeaderTruncated:
    return "header_truncated";
  case ParseErrorCode::BadMagic:
    return "bad_magic";
  case ParseErrorCode::UnsupportedVersion:
    return "unsupported_version";
  case ParseErrorCode::InvalidLayout:
    return "invalid_layout";
  case ParseErrorCode::HeaderChecksum:
    return "header_checksum";
  case ParseErrorCode::InvalidLength:
    return "invalid_length";
  case ParseErrorCode::MetadataUnavailable:
    return "metadata_unavailable";
  case ParseErrorCode::MetadataChecksum:
    return "metadata_checksum";
  case ParseErrorCode::PayloadChecksum:
    return "payload_checksum";
  case ParseErrorCode::InvalidEvent:
    return "invalid_event";
  case ParseErrorCode::UnknownKind:
    return "unknown_kind";
  case ParseErrorCode::IdentityConflict:
    return "identity_conflict";
  case ParseErrorCode::EventConflict:
    return "event_conflict";
  }
  return {};
}

ParseError::ParseError(ParseErrorCode code, std::string message, std::size_t offset, std::uint64_t producer,
                       std::optional<std::uint64_t> event_sequence, std::optional<std::filesystem::path> path,
                       std::string root_cause)
    : c10::Error(std::move(message)), code_(code), offset_(offset), producer_(producer),
      event_sequence_(event_sequence), path_(std::move(path)), root_cause_(std::move(root_cause)) {}

} // namespace xpool::devkit::timeline
