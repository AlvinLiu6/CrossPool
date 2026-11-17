#pragma once

/// \file xpool/devkit/timeline/buffer.hpp
/// \brief Host-only immutable event slots, copy ownership, and checked sizing.

#include <cstddef>
#include <cstdint>
#include <span>

#include <xpool/devkit/timeline/event.hpp>

namespace xpool::devkit::timeline {

/// Private slot generations reserve four ownership bits. Exhausting this
/// representation stops the pool; it never reduces the Event sequence range.
inline constexpr std::uint64_t kSlotGenerationLimit = (std::uint64_t{1} << 60) - 1;

/// Caller-owned Host control. Initialize before construction; all subsequent
/// access goes through the pool until producers and copy owners retire.
struct alignas(8) SlotControl {
  /// Private packed generation and phase; first generation starts Free.
  std::uint64_t ownership = std::uint64_t{1} << 4;
  /// Attempt identity installed before the reservation token is returned.
  std::uint64_t event_sequence = 0;
};

/// Caller-owned Host scheduling state. Cursor wrap is harmless; it is never an
/// event identity. A stopped pool still permits existing owners to finish.
struct alignas(8) SlotPoolState {
  /// Rotating scan origin; unsigned wrap carries no identity semantics.
  std::uint64_t consume_cursor = 0;
  /// Private stop-reason flags, accessed atomically during service.
  std::uint64_t stopped = 0;
};

/// Single-allocation geometry in bytes, derived from actual sizes and alignment.
/// This accounts only for the slot pool, not process collection or export.
struct SlotPoolLayout {
  /// Byte count of Event storage beginning at allocation offset zero.
  std::size_t event_bytes;
  /// Aligned beginning of the slot control region.
  std::size_t control_offset;
  /// Actual bytes occupied by all controls.
  std::size_t control_bytes;
  /// Aligned beginning of shared scheduling state.
  std::size_t state_offset;
  /// Actual bytes occupied by scheduling state.
  std::size_t state_bytes;
  /// Complete requested allocation, including all layout padding.
  std::size_t total_bytes;
};

/// Current slot occupancy after quiescence, not cumulative process quality.
struct SlotCounts {
  /// Slots available to new reservations.
  std::size_t free = 0;
  /// Uncommitted reservations whose owners have not abandoned.
  std::size_t reserved = 0;
  /// Immutable events awaiting acquisition.
  std::size_t committed = 0;
  /// Slots held by copy owners, including delayed transfers.
  std::size_t copying = 0;
  /// Slots whose private generation can no longer advance.
  std::size_t retired = 0;
  /// Whether new admission has stopped.
  bool stopped = false;
  /// Whether generation exhaustion caused a stop.
  bool identity_exhausted = false;
};

/// Bounded multi-producer/multi-consumer Host slot ownership. Storage stays
/// fixed and alive until all calls and acquired payload reads finish. C0's
/// Collector, sequence allocation, quality accounting and CUDA publication
/// bridge belong above this mechanism. This type performs no allocation or I/O.
class HostSlotPool {
public:
  /// Borrow initialized, disjoint payload/control arrays and scheduling state.
  /// \pre Construction has exclusive access; each control is Free with a
  /// positive generation and zero event sequence. State is zero-initialized.
  /// \throws std::invalid_argument for empty/unequal spans, overlapping storage,
  /// or invalid initial controls/state. Standard errors are local construction
  /// errors, not C0 CollectionError values at the future Collector boundary.
  HostSlotPool(std::span<Event> events, std::span<SlotControl> slots, SlotPoolState &state);

  HostSlotPool(const HostSlotPool &) = delete;
  HostSlotPool &operator=(const HostSlotPool &) = delete;

  /// Claim at most one candidate per slot, returning Full only after observing
  /// each candidate occupied at its own claim point. Concurrent release may
  /// free capacity during the scan; Full is not a global fullness snapshot.
  /// \param event_sequence Caller-assigned positive identity, unique for every
  /// attempt in this pool's lifetime, including attempts returning Full.
  /// \throws std::invalid_argument when event_sequence is zero.
  ReserveResult try_reserve(std::uint64_t event_sequence);

  /// Copy and publish one complete value. The token's sequence replaces the
  /// input's event_sequence; the caller retains and may change its input later.
  /// A commit that has claimed its token already excludes abandon/other commits.
  CommitStatus commit(WriteToken token, const Event &event);

  /// Retire one uncommitted reservation. A committing/committed payload is kept.
  AbandonStatus abandon(WriteToken token);

  /// Acquire any committed generation, skipping pending lower sequences.
  /// Consumers own disjoint payloads; an empty result grants no ownership.
  ReadResult try_acquire();

  /// Recycle an acquired generation after its complete payload copy retires.
  /// \pre All reads through the acquired pointer have finished. A timeout does
  /// not establish completion. The pointer becomes invalid after release.
  ReleaseStatus release(ReadToken token);

  /// Stop admission; previously admitted reservations/copies remain retireable.
  /// A call concurrent with stop may finish an admission already in progress.
  void stop();

  /// Return current occupancy after all pool calls have quiesced. Reserved and
  /// held Copying slots may remain; this is not a live or cumulative checkpoint.
  SlotCounts quiescent_counts() const;

  /// Compute complete pool geometry for one allocation using checked arithmetic.
  /// \param capacity Positive number of fixed event slots.
  /// \param allocation_alignment Power of two, at least the required alignment
  /// of Event, SlotControl and SlotPoolState. Extra collector resources are added
  /// separately by their owner before process-budget admission.
  /// \throws std::invalid_argument for invalid capacity/alignment.
  /// \throws std::overflow_error for size, offset or rounding overflow.
  static SlotPoolLayout layout(std::size_t capacity, std::size_t allocation_alignment = 256);

private:
  friend class HostEventSource;

  // Source entry has already accepted this attempt before concurrent stop.
  // Only the source owner bypasses the independent low-level stop check.
  ReserveResult reserve_admitted(std::uint64_t event_sequence);
  bool matches(WriteToken token) const;
  void recycle(std::size_t slot, std::uint64_t generation);

  std::span<Event> events_;
  std::span<SlotControl> slots_;
  SlotPoolState &state_;
};

} // namespace xpool::devkit::timeline
