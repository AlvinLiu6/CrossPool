#pragma once

/// \file xpool/devkit/timeline/event.hpp
/// \brief Immutable timeline event values and reservation identities.

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace xpool::devkit::timeline {

/// C0 event value. Presence bits distinguish absent facts from recorded zero.
/// Run and Producer belong to the containing batch, not this value. Native
/// padding is never a wire encoding; exporters encode fields explicitly.
struct alignas(8) Event {
  /// Fixed wire record size, independent of native padding or byte order.
  static constexpr std::size_t encoded_bytes = 256;

  /// Positive identity assigned by the source, independent of slot position.
  std::uint64_t event_sequence = 0;
  /// Raw nanoseconds; zero is valid when the event was recorded.
  std::uint64_t raw_time_ns = 0;
  /// Registered source clock domain; raw times retain this provenance.
  std::uint64_t clock_domain_id = 0;
  /// Explicit C0 event kind, independent of production Hook enum ordinals.
  std::uint32_t kind = 0;
  /// C0 known-fact group mask; absent fields stay zero.
  std::uint32_t presence = 0;
  /// Registered Fabric Generation reference.
  std::uint64_t generation = 0;
  /// Device Producer that owns the Transport Endpoint.
  std::uint64_t endpoint_owner = 0;
  /// Owner-local Endpoint creation identity.
  std::uint64_t endpoint_sequence = 0;
  /// Successful Claim identity within the Endpoint, independent of event loss.
  std::uint64_t operation_sequence = 0;
  /// Host Producer that owns the Graph template or execution.
  std::uint64_t graph_owner = 0;
  /// Owner-local Graph template identity; absent for Eager execution.
  std::uint64_t graph_template = 0;
  /// Fresh identity of the observed Graph execution.
  std::uint64_t graph_execution = 0;
  /// Zero-based Instance component of the production InvocationKey.
  std::uint64_t instance_index = 0;
  /// Positive production Invocation sequence.
  std::uint64_t invocation_sequence = 0;
  /// Zero-based Executor Lane index, present only when locally known.
  std::uint64_t lane_index = 0;
  /// Production lease sequence for the selected Lane.
  std::uint64_t lease_sequence = 0;
  /// Producer that allocated the Routing attachment identity.
  std::uint64_t attachment_owner = 0;
  /// Stable attachment identity, independent of tensor-pool position.
  std::uint64_t attachment_sequence = 0;
  /// Legacy local operation identity, independent of event reservations.
  std::uint64_t local_trace_id = 0;
  /// Zero-based model FFN layer ordinal.
  std::uint64_t layer_ordinal = 0;
  /// Live hidden-state row count.
  std::uint64_t payload_rows = 0;
  /// This attention DP rank's contribution rows.
  std::uint64_t dp_rank_payload_rows = 0;
  /// Actual selected FFN execution row capacity.
  std::uint64_t payload_row_capacity = 0;
  /// FIFO Coordinator ticket; presence distinguishes random scheduling.
  std::uint64_t ready_ticket = 0;
  /// Explicit FFN ForwardMode wire value.
  std::uint32_t forward_mode = 0;
  /// Explicit FFN OutputRequirement wire value.
  std::uint32_t output_requirement = 0;
  /// Explicit FFN DpRowLayout wire value.
  std::uint32_t dp_row_layout = 0;
  /// Actual Fabric delivery variant wire value.
  std::uint32_t delivery = 0;
  /// Site-local observed result; zero means Ok only when present.
  std::uint32_t result_code = 0;
  /// C0 observation role, distinct from the process runtime role.
  std::uint32_t role = 0;
  /// Zero-based Instance rank when present.
  std::uint32_t instance_rank = 0;
  /// Source NVSHMEM PE when present; absent stores zero.
  std::int32_t pe = 0;
  /// First kind-specific word, interpreted by the C0 event family.
  std::uint64_t data0 = 0;
  /// Second kind-specific word.
  std::uint64_t data1 = 0;
  /// Third kind-specific word.
  std::uint64_t data2 = 0;
  /// Fourth kind-specific word.
  std::uint64_t data3 = 0;
  /// Fifth kind-specific word.
  std::uint64_t data4 = 0;

  /// Encode all fields in little-endian order into caller-owned storage.
  /// Values are preserved exactly; semantic validity and metadata references
  /// remain the source owner's responsibility. Destination may overlap this
  /// Event: a stable value copy is taken before any output write.
  void encode(std::span<std::byte, encoded_bytes> destination) const noexcept;

  /// Decode the complete field layout without checking semantic validity.
  /// Call validate() and the metadata owner's reference checks before use.
  static Event decode(std::span<const std::byte, encoded_bytes> bytes) noexcept;

  /// Validate intrinsic supported kinds, presence, zeroed absent fields and family facts.
  /// Source clock and Routing publisher must match the supplied registered source.
  /// Metadata schema, membership and identity-reference resolution are caller
  /// preconditions owned by preparation; this method parses no JSON.
  /// \throws ParseError with unknown_kind or invalid_event and field byte offset.
  void validate(std::uint64_t producer, std::uint64_t source_clock_domain_id) const;

  /// Compare every declared field, independently of native padding.
  bool operator==(const Event &) const = default;
};

static_assert(std::is_standard_layout_v<Event> && std::is_trivially_copyable_v<Event>);
static_assert(sizeof(Event) == 256 && alignof(Event) == 8);
static_assert(offsetof(Event, event_sequence) == 0);
static_assert(offsetof(Event, raw_time_ns) == 8);
static_assert(offsetof(Event, clock_domain_id) == 16);
static_assert(offsetof(Event, kind) == 24);
static_assert(offsetof(Event, presence) == 28);
static_assert(offsetof(Event, generation) == 32);
static_assert(offsetof(Event, endpoint_owner) == 40);
static_assert(offsetof(Event, endpoint_sequence) == 48);
static_assert(offsetof(Event, operation_sequence) == 56);
static_assert(offsetof(Event, graph_owner) == 64);
static_assert(offsetof(Event, graph_template) == 72);
static_assert(offsetof(Event, graph_execution) == 80);
static_assert(offsetof(Event, instance_index) == 88);
static_assert(offsetof(Event, invocation_sequence) == 96);
static_assert(offsetof(Event, lane_index) == 104);
static_assert(offsetof(Event, lease_sequence) == 112);
static_assert(offsetof(Event, attachment_owner) == 120);
static_assert(offsetof(Event, attachment_sequence) == 128);
static_assert(offsetof(Event, local_trace_id) == 136);
static_assert(offsetof(Event, layer_ordinal) == 144);
static_assert(offsetof(Event, payload_rows) == 152);
static_assert(offsetof(Event, dp_rank_payload_rows) == 160);
static_assert(offsetof(Event, payload_row_capacity) == 168);
static_assert(offsetof(Event, ready_ticket) == 176);
static_assert(offsetof(Event, forward_mode) == 184);
static_assert(offsetof(Event, output_requirement) == 188);
static_assert(offsetof(Event, dp_row_layout) == 192);
static_assert(offsetof(Event, delivery) == 196);
static_assert(offsetof(Event, result_code) == 200);
static_assert(offsetof(Event, role) == 204);
static_assert(offsetof(Event, instance_rank) == 208);
static_assert(offsetof(Event, pe) == 212);
static_assert(offsetof(Event, data0) == 216);
static_assert(offsetof(Event, data1) == 224);
static_assert(offsetof(Event, data2) == 232);
static_assert(offsetof(Event, data3) == 240);
static_assert(offsetof(Event, data4) == 248);

/// Reservation outcome; only Reserved grants ownership.
enum class ReserveStatus : std::uint32_t { Reserved = 0, Full = 1, Stopped = 2, IdentityExhausted = 3 };
/// Result of committing one reserved slot generation.
enum class CommitStatus : std::uint32_t { Committed = 0, StaleToken = 1, AlreadyCommitted = 2 };
/// Result of retiring one uncommitted reservation.
enum class AbandonStatus : std::uint32_t { Abandoned = 0, StaleToken = 1, AlreadyCommitted = 2 };

/// Exclusive write identity; copying this value creates no additional ownership.
struct WriteToken {
  /// Slot index in the originating fixed pool.
  std::uint64_t slot = 0;
  /// Slot reuse generation; a token cannot authorize another generation.
  std::uint64_t slot_generation = 0;
  /// Positive attempted-event identity, independent of operation identity.
  std::uint64_t event_sequence = 0;
};

/// Non-Reserved outcomes carry an all-zero token and grant no ownership.
struct ReserveResult {
  /// Admission result from the originating Source.
  ReserveStatus status = ReserveStatus::Stopped;
  /// Write authority, valid only when status is Reserved.
  WriteToken token{};
};

/// Host batch ownership; its immutable span remains valid through release.
struct ConsumeToken {
  /// Originating Collector's opaque consume slot, independent of Event identity.
  std::uint64_t batch = 0;
  /// Reuse generation isolating old consumers.
  std::uint64_t generation = 0;
};

/// Copy ownership retires once, after complete payload transfer.
enum class ReleaseStatus : std::uint32_t { Released = 0, StaleToken = 1, NotAcquired = 2 };

/// Immutable slot read authority, distinct from Producer write ownership.
struct ReadToken {
  /// Originating slot index.
  std::uint64_t slot = 0;
  /// Generation retained until the complete read/copy retires.
  std::uint64_t slot_generation = 0;
  /// Identity of this committed Event.
  std::uint64_t event_sequence = 0;
};

/// Borrowed immutable payload; acquired pointers expire only on release.
struct ReadResult {
  /// Only true grants ownership; empty results carry no token/pointer.
  bool acquired = false;
  /// Exactly-once release authority in the originating pool.
  ReadToken token{};
  /// Stable while this generation remains Copying.
  const Event *event = nullptr;
};

} // namespace xpool::devkit::timeline
