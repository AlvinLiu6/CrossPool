#pragma once

/// \file xpool/devkit/timeline/evidence.hpp
/// \brief Bounded typed snapshot evidence and canonical field provenance.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <xpool/devkit/timeline/contract.hpp>
#include <xpool/devkit/timeline/json.hpp>
#include <xpool/devkit/timeline/legacy.hpp>

namespace xpool::devkit::timeline {

/// Membership in the legacy Record array, independent of evidence completeness.
enum class ProjectionStatus { Retained, Omitted, Unobserved };
/// Completeness of observed applicable facts and boundaries at one cutoff.
enum class EvidenceStatus { Complete, Partial, Unavailable };
/// Selected legacy projection domain.
enum class SnapshotObserver { Transport, Fabric };
/// Local role, absent only for an unobserved Fabric operation.
enum class SnapshotRole : std::uint32_t { Instance = 1, AtnAgent = 2, Coordinator = 3, FfnAgent = 4 };
/// Finite per-operation reasons; aggregate Producer loss is never a local cause.
enum class EvidenceReason : std::uint32_t {
  EventMissing = 1,
  UnknownResult = 2,
  LegacyFactMissing = 4,
  OperationUnobserved = 8,
  ApplicabilityUnknown = 16,
  LegacyZeroTimestamp = 32,
  TemporalConflict = 64,
  InvalidRequest = 128
};

/// Exact C0 lowercase role spelling shared by Native serializers and bindings.
std::string_view evidence_role_name(SnapshotRole role) noexcept;
/// Exact C0 local reason spelling; returns empty for an invalid enum value.
std::string_view evidence_reason_name(EvidenceReason reason) noexcept;

/// Explicit Transport identity, never inferred from a local trace ID.
struct OperationReference {
  /// Registered Endpoint owner.
  std::uint64_t endpoint_owner;
  /// Positive Endpoint lifetime.
  std::uint64_t endpoint_sequence;
  /// Positive protocol Operation sequence.
  std::uint64_t operation_sequence;
};
/// Explicit Fabric Invocation identity.
struct InvocationReference {
  /// Registered Generation.
  std::uint64_t generation;
  /// Zero-based Instance index.
  std::uint64_t instance_index;
  /// Positive Invocation sequence.
  std::uint64_t invocation_sequence;
};

/// A bounded metadata citation. Current C0 schema paths fit 128 bytes; reject
/// larger registration before service instead of truncating a source path.
struct EvidenceMetadataSource {
  /// Producer that published the referenced metadata document.
  std::uint64_t producer = 0;
  /// Exact immutable document revision.
  std::uint64_t revision = 0;
  /// Fixed path storage, not a mutable or borrowed string.
  std::array<char, 128> path_bytes{};
  /// Number of UTF-8 bytes in path_bytes.
  std::size_t path_size = 0;
  /// Create one validated bounded citation before readout.
  /// \throws CollectionError for missing identity/path or oversized path.
  static EvidenceMetadataSource make(std::uint64_t producer, std::uint64_t revision, std::string_view path);
  /// Exact path, valid for this value's lifetime.
  std::string_view path() const noexcept { return {path_bytes.data(), path_size}; }
};

/// Immutable inputs from actual metadata, not observation-enable flags.
struct SnapshotMetadata {
  /// Citation for the selected Generation's scheduler policy.
  EvidenceMetadataSource scheduler{};
  /// True only when actual layer/capability evidence proves no Routing branch.
  bool routing_not_applicable = false;
  /// Citation supporting routing_not_applicable.
  EvidenceMetadataSource routing{};
  /// Actual Generation topology; absence keeps peer applicability unknown.
  std::optional<std::uint64_t> ffnagent_count = std::nullopt;
  /// Citation supporting ffnagent_count.
  EvidenceMetadataSource topology{};
};

/// Raw anomalous Transport request and actual observed terminal boundaries.
/// Terminal indices are ResultPublished, ResultObserved, ResultAcknowledged,
/// Closed. A set bit preserves even a zero timestamp; result is the local
/// effective result whose repeated facts were checked by reconstruction.
struct TransportAnomalyEvidence {
  std::uint64_t layer_ordinal = 0;               ///< Received layer, including invalid geometry.
  std::uint64_t payload_rows = 0;                ///< Received physical row count.
  std::uint32_t forward_mode = 0;                ///< Original unmodified enum bits.
  std::uint32_t output_requirement = 0;          ///< Original unmodified enum bits.
  std::uint32_t dp_row_layout = 0;               ///< Original unmodified enum bits.
  std::uint32_t terminal_mask = 0;               ///< Known boundaries indexed as above.
  std::array<std::uint64_t, 4> terminal_times{}; ///< Raw times for set terminal bits.
  std::uint32_t result_code = 0;                 ///< Actual local result, meaningful for set bits.
};

/// One fixed-size evidence entry. Fields 0..15 are finite scalar paths;
/// 16..26 timestamp paths; 32..42 boundary decisions, interpreted by its role.
/// Bitsets are exposed as sorted path lists by serializers/bindings. Sources
/// retain one minimum Event ID or one metadata-source selector per field.
struct SnapshotEvidenceEntry {
  /// Positive eligible operation ID.
  std::uint64_t local_trace_id = 0;
  /// Own observed role; unobserved Fabric remains null.
  std::optional<SnapshotRole> role = std::nullopt;
  /// Present only with actual Operation facts.
  std::optional<OperationReference> operation = std::nullopt;
  /// Present only with actual Invocation facts.
  std::optional<InvocationReference> invocation = std::nullopt;
  /// Retained, omitted for missing scalars, or no local evidence.
  ProjectionStatus projection_status = ProjectionStatus::Unobserved;
  /// Completeness independent of legacy terminal counters.
  EvidenceStatus evidence_status = EvidenceStatus::Unavailable;
  /// Unknown finite scalar/timestamp paths, distinct from inapplicable paths.
  std::uint64_t unknown_fields = 0;
  /// Proven inapplicable field paths.
  std::uint64_t not_applicable_fields = 0;
  /// Expected but missing protocol boundaries, indexed in this role's family.
  std::uint32_t missing_boundaries = 0;
  /// Conditional boundaries with unresolved applicability.
  std::uint32_t unknown_applicability = 0;
  /// EvidenceReason bitset.
  std::uint32_t quality = 0;
  /// Canonical Event sequence, or metadata selector 1=scheduler, 2=routing,
  /// 3=topology. Full-width Event identities have no reserved flag bits.
  std::array<std::uint64_t, 43> sources{};
  /// Which finite fields carry a canonical source.
  std::uint64_t source_fields = 0;
  /// Subset of source_fields referring to metadata rather than an Event.
  std::uint64_t metadata_fields = 0;
  /// Present only for observed invalid Transport request enums. Its bytes
  /// remain part of the report's admitted capacity and coexistence peak.
  std::optional<TransportAnomalyEvidence> anomaly = std::nullopt;
};

/// Resolve one finite internal field to its exact legacy-relative C0 path.
/// Empty denotes an unused index for this selected role.
std::string_view evidence_field_name(SnapshotObserver observer, SnapshotRole role, std::size_t field) noexcept;
/// Resolve a protocol phase index to its stable C0 CamelCase kind name.
/// Empty denotes a phase outside this role's family.
std::string_view evidence_boundary_name(SnapshotObserver observer, SnapshotRole role, std::size_t phase) noexcept;

/// Immutable same-cutoff report. Create only after its serialized merge lane
/// retires or is held at the requested read barrier. The allocation and shared
/// control block remain charged through all Native readers and weak references.
class SnapshotEvidenceReport {
  class ConstructionKey {
    friend class SnapshotEvidenceReport;
    ConstructionKey() = default;
  };

public:
  /// Allocate declared capacity before populating entries. Sources and counts
  /// come from the same stable store image; never combine independent reads.
  /// \throws CollectionError on admission, missing metadata or invalid sources.
  static std::shared_ptr<SnapshotEvidenceReport> create(std::shared_ptr<MemoryBudget> budget,
                                                        const detail::LegacyStore &store, SnapshotMetadata metadata);
  /// Retire the report slab after every reader ends.
  ~SnapshotEvidenceReport() = default;
  SnapshotEvidenceReport(const SnapshotEvidenceReport &) = delete;
  SnapshotEvidenceReport &operator=(const SnapshotEvidenceReport &) = delete;
  /// Fixed artifact contract.
  const std::uint32_t contract_version = kContractVersion;
  /// Registered source/Run; its Producer is not a PE or buffer index.
  const BatchSource source;
  /// Selected projection family.
  const SnapshotObserver observer;
  /// Positive domain Generation.
  const std::uint64_t generation;
  /// Endpoint is set only for Transport.
  const std::optional<std::array<std::uint64_t, 2>> endpoint;
  /// Instance/AtnAgent for Transport, null for Fabric.
  const std::optional<SnapshotRole> site;
  /// Native-only reads have no published JSON hash. A publisher includes the
  /// exact snapshot digest in its immutable serialized sidecar.
  const std::optional<std::array<std::byte, 32>> snapshot_sha256 = std::nullopt;
  /// Earliest-N retention window.
  const std::uint64_t capacity;
  /// Exact original operation attempts, or null when unavailable.
  const std::optional<std::uint64_t> sequence;
  /// Exact capacity exclusions, or null with unknown sequence.
  const std::optional<std::uint64_t> dropped;
  /// Eligible IDs with no local evidence, or null with unknown sequence.
  const std::optional<std::uint64_t> unobserved_operations;
  /// Sorted immutable entries, including unobserved IDs for exact totals.
  std::span<const SnapshotEvidenceEntry> entries() const noexcept { return entries_; }
  /// Metadata citations used by finite source selectors.
  const SnapshotMetadata &metadata() const noexcept { return metadata_; }
  /// Stream the exact C0 report schema with sorted unique lists and sources.
  /// A null sink counts bytes for admission. The optional digest is supplied
  /// only by the publisher after hashing the exact paired snapshot bytes;
  /// Native-only reads serialize a null hash. Uses bounded stack storage.
  /// \throws CollectionError for invalid retained field/source mappings;
  /// destination failures propagate and leave cleanup to the publication owner.
  std::size_t serialize(JsonSink sink = {},
                        std::optional<std::array<std::byte, 32>> snapshot_digest = std::nullopt) const;
  /// Allocator-only constructor; use create() to charge its control block.
  SnapshotEvidenceReport(ConstructionKey, std::shared_ptr<MemoryBudget> budget, const detail::LegacyStore &store,
                         SnapshotMetadata metadata);

private:
  const std::shared_ptr<MemoryBudget> budget_;
  const SnapshotMetadata metadata_;
  MemoryBudget::Reservation entries_charge_;
  std::vector<SnapshotEvidenceEntry> entries_;
};

} // namespace xpool::devkit::timeline
