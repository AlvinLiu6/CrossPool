#pragma once

/// \file xpool/devkit/timeline/fabricrecording.hpp
/// \brief Borrowed Fabric recording facts and bounded current-operation storage.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <xpool/devkit/timeline/view.hpp>
#include <xpool/fabric/protocol.hpp>

namespace xpool::devkit::timeline {

/// Local adapter outcome, never an Event kind, quality counter or wire enum.
enum class FabricRecordingStatus {
  Disabled,    ///< No installed recording owner; no Device state was accessed.
  Ready,       ///< Matching current operation and open Source admission.
  Stopped,     ///< Source is closed; already issued tokens remain valid.
  Unavailable, ///< Installed owner has no matching current trace for this key.
};

/// Bounded internal diagnostics for facts rejected before Event reservation.
/// These bits do not imply an Event attempt or a public C0 stop reason.
enum class FabricDiagnostic : std::uint64_t {
  MissingOperation = 1,     ///< Association has no published Operation fact.
  ConflictingOperation = 2, ///< Operation facts disagree; do not choose one.
  SourceMismatch = 4,       ///< Supplied identities do not belong to this Source.
  InvalidFacts = 8,         ///< Invalid geometry, protocol value or boundary facts.
  MissingTrace = 16,        ///< Ordinary boundary has no prepared current operation.
  CommitFailure = 32,       ///< Source rejected commit; its token counters are final.
};

/// Shared by every Fabric role/Instance of one Producer/Generation. Initialize
/// before users; live Device access uses atomics. Host reads require a completed
/// snapshot transfer or actual production quiescence, never mapped atomics.
struct FabricRecordingState {
  /// Original operation identities, independent of Source Event attempts.
  DeviceAttemptState traces{};
  /// Every actual record-begin, including operations after Source stop/full.
  DeviceCounterState operations{};
  /// Operations outside the configured earliest-N legacy window.
  DeviceCounterState dropped{};
  /// Sticky inability to allocate another positive local operation identity.
  std::uint64_t identity_exhausted = 0;
  /// Sticky internal reason bits; bounded space, no per-failure history.
  std::uint64_t diagnostics = 0;
};

/// Current facts for one serialized AtnAgent Instance. The producer thread
/// prepares and updates it; synchronous Hook callbacks only borrow/copy it.
/// The Collector does not inspect this mutable table. No history is retained.
struct FabricAtnCurrent {
  /// Zero before preparation or after the operation's final local Hook.
  std::uint64_t local_trace_id = 0;
  /// Actual validated contribution, copied before either Prepared listener.
  xpool::fabric::Submission submission{};
  /// Facts become known only at their actual local observation boundaries.
  std::uint64_t executor_lane_index = 0;
  /// Positive lease observed with the selected Lane; ignored until admission_known.
  std::uint64_t executor_lease_sequence = 0;
  /// Actual OutputCommit result; ignored until commit_known is true.
  xpool::ffn::ResultCode result = xpool::ffn::ResultCode::Ok;
  /// Admission was observed for this operation, including its exact lease.
  bool admission_known = false;
  /// OutputCommit was observed for this operation.
  bool commit_known = false;
};

/// Serialized Coordinator facts for one active Instance invocation.
struct FabricCoordinatorCurrent {
  std::uint64_t local_trace_id = 0;       ///< Shared Producer/Generation sequence.
  xpool::fabric::Invocation invocation{}; ///< Actual rendezvous result.
  std::uint64_t ready_ticket = 0;         ///< Actual FIFO ticket when configured.
  std::uint64_t lane_index = 0;           ///< Selected executor, known after Scheduled.
  std::uint64_t lease_sequence = 0;       ///< Positive selected lease.
  bool scheduled = false;                 ///< Whether selection facts have been observed.
};

/// Serialized FfnAgent facts for one executor Lane, copied at its begin Hook.
struct FabricFfnCurrent {
  std::uint64_t local_trace_id = 0;          ///< Shared with Coordinator in this Producer.
  xpool::fabric::LaneExecution execution{};  ///< Actual validated execution.
  std::uint64_t payload_row_capacity = 0;    ///< Installed execution capacity.
  xpool::fabric::DeliveryVariant delivery{}; ///< Actual selected branch.
};

/// Cold geometry of this participant. Role flags come from the actual PE;
/// only tables belonging to that role are allocated. No Device pointer is input.
struct FabricRecordingConfig {
  std::uint64_t generation = 0;      ///< Registered Generation reference.
  std::int32_t pe = -1;              ///< Generation-scoped physical participant index.
  std::size_t instances = 0;         ///< Actual Instance table extent.
  std::size_t lanes = 0;             ///< Actual executor Lane extent.
  std::uint64_t record_capacity = 0; ///< Earliest-N window across local roles.
  bool atnagent = false;             ///< Participant is in the AtnAgent PE prefix.
  bool coordinator = false;          ///< Participant is the first FfnAgent PE.
  bool fifo = false;                 ///< Actual production scheduler policy.
  bool operator==(const FabricRecordingConfig &) const = default;
};

/// Atomic cumulative Device sample carried inside the existing D2H frame.
/// Never a wire structure. exact requires actual producer quiescence and no
/// counter overflow; active samples are monotone lower bounds on the Host.
struct FabricProgress {
  std::uint64_t operations = 0;    ///< Original record-begin count, not Event attempts.
  std::uint64_t dropped = 0;       ///< Operations outside the earliest-N window.
  std::uint64_t diagnostics = 0;   ///< Internal sticky bits, not C0 JSON fields.
  bool counter_exhausted = false;  ///< Saturated original counters are inexact.
  bool identity_exhausted = false; ///< No further positive trace may be issued.
  bool exact = false;              ///< Producer quiescence, independent of export completion.
};

/// Stable process-local view installed by the actual budgeted recording owner.
/// All Device pointers stay alive through callbacks, kernels, Graph references
/// and final collection. An empty view is disabled and is never dereferenced.
/// Host preparation validates Source/Generation/PE membership and dimensions.
struct FabricRecordingView {
  /// The shared process/Device Source; association listeners use this same pool.
  BufferView buffer{};
  /// Registered Fabric Generation reference, not a pointer-derived identity.
  std::uint64_t generation = 0;
  /// Actual generation-scoped local PE supplied by preparation.
  std::int32_t pe = -1;
  /// Producer/Generation-wide operation counter and bounded diagnostics.
  FabricRecordingState *state = nullptr;
  /// One current record per Instance; distinct Instances may run concurrently.
  FabricAtnCurrent *atnagents = nullptr;
  /// Actual fixed table extent, validated before launch.
  std::size_t instance_count = 0;
  /// Original Fabric earliest-N retention capacity; zero disables that window.
  std::uint64_t record_capacity = 0;
  /// Optional Coordinator table in the same owner allocation.
  FabricCoordinatorCurrent *coordinators = nullptr;
  /// Optional FfnAgent table, indexed by executor Lane.
  FabricFfnCurrent *ffnagents = nullptr;
  /// Extent of the optional FfnAgent table.
  std::size_t lane_count = 0;
  /// Whether Coordinator tickets describe the FIFO production scheduler.
  bool fifo = false;
};

/// Copied consumer facts. Reading this value never allocates a trace or Event.
/// Only Ready authorizes a new reservation; retained tokens follow Source rules
/// independently. The consumer reads its own timestamp at its actual Hook.
struct FabricRecordingFacts {
  /// Disabled, source-stopped, missing/current-key mismatch, or writable.
  FabricRecordingStatus status = FabricRecordingStatus::Disabled;
  /// Borrowed Source; valid only within the owning runtime's retained lifetime.
  BufferView buffer{};
  /// Registered Fabric generation for this operation.
  std::uint64_t generation = 0;
  /// Actual recording participant.
  std::int32_t pe = -1;
  /// Shared original Fabric operation identity, never an Event sequence.
  std::uint64_t local_trace_id = 0;
};

/// Actual state/current-table layout; allocations use this complete rounded
/// extent once. The Source and transfer frames are separate existing owners.
struct FabricRecordingLayout {
  /// State starts at offset zero.
  std::size_t current_offset;
  /// Optional Coordinator current-table offset.
  std::size_t coordinator_offset;
  /// Optional FfnAgent current-table offset.
  std::size_t ffn_offset;
  /// Complete allocation size, including final allocation alignment.
  std::size_t total_bytes;
  /// Checked geometry only: no CUDA or allocation.
  /// \throws CollectionError resource_budget for empty/overflowing geometry.
  static FabricRecordingLayout for_instances(std::size_t instances, std::uint64_t producer = 0);
  /// Actual participant shape; rejects impossible role/geometry before CUDA.
  static FabricRecordingLayout for_config(FabricRecordingConfig config, std::uint64_t producer = 0);
};

static_assert(std::is_trivially_copyable_v<FabricRecordingState>);
static_assert(std::is_trivially_copyable_v<FabricAtnCurrent>);
static_assert(std::is_trivially_copyable_v<FabricRecordingView>);

} // namespace xpool::devkit::timeline
