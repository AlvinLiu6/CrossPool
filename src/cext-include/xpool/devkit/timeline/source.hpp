#pragma once

/// \file xpool/devkit/timeline/source.hpp
/// \brief Concurrent Host admission, source identity and cumulative outcomes.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include <xpool/devkit/timeline/buffer.hpp>
#include <xpool/devkit/timeline/counter.hpp>
#include <xpool/macros.hpp>

namespace xpool::devkit::timeline {

/// Immutable Run/Producer/clock registration shared by Source and Host batches.
struct BatchSource {
  /// Opaque canonical UUID bytes; never derived from PID.
  std::array<std::byte, 16> run_id{};
  /// Registered Producer identity supplied by preparation.
  std::uint64_t producer = 0;
  /// Registered clock domain repeated on every Event.
  std::uint64_t clock_domain_id = 0;
};

/// Borrowed source ledger. Initialize exclusively before construction; retained
/// history belongs to the same registered source in the same live process.
/// Direct reads require all source calls to have quiesced.
struct alignas(8) SourceState {
  /// Active reservation calls and sticky stop flags, updated atomically.
  std::uint64_t admission = 0;
  /// Full-width attempted-event identity allocation and retained history.
  AttemptSequence attempts{};
  /// Distinct successful immutable commits, including already consumed events.
  ConcurrentCounter committed{};
  /// Attempts that found every slot occupied at its claim point.
  ConcurrentCounter dropped_full{};
  /// Distinct reservations successfully retired before commit.
  ConcurrentCounter abandoned{};
  /// Rejected commit/abandon/release tokens, excluding duplicate commits.
  ConcurrentCounter stale_token{};
  /// Commit calls rejected because this generation already claimed commit.
  ConcurrentCounter duplicate_commit{};
};

/// Source-local exact totals after quiescence, or saturated lower bounds.
/// Writer totals and live/legacy diagnostic projection belong to other owners.
struct SourceCounts {
  /// One identity per accepted reservation attempt, including Full.
  std::uint64_t attempted = 0;
  /// Last allocated source identity, never a buffer index.
  std::uint64_t last_attempted_sequence = 0;
  /// Cumulative immutable commits.
  std::uint64_t committed = 0;
  /// Cumulative attempts rejected by full slot capacity.
  std::uint64_t dropped_full = 0;
  /// Cumulative successful abandons.
  std::uint64_t abandoned = 0;
  /// Cumulative rejected token diagnostics.
  std::uint64_t stale_token = 0;
  /// Cumulative duplicate commit diagnostics.
  std::uint64_t duplicate_commit = 0;
  /// Reserved slots whose attempt has no terminal outcome yet.
  std::uint64_t pending = 0;
  /// Any diagnostic or outcome counter has overflowed its logical range.
  bool counter_exhausted = false;
  /// Whether cumulative values remain exact rather than lower bounds.
  bool exact_totals = true;
  /// Whether source or slot admission has closed.
  bool stopped = false;
  /// Whether source sequence or private slot generation exhausted.
  bool identity_exhausted = false;
};

/// Cumulative source evidence sampled by a collection lane. pending is usable
/// only when pending_known is true; active sampling never invents a final zero.
struct SourceProgress {
  /// Cumulative fields are lower bounds unless a quiescent sample says exact.
  SourceCounts counts{};
  /// Whether the sample was made after producer API retirement.
  bool pending_known = false;
};

/// Complete pool/source geometry for one allocation, not process admission.
struct SourceLayout {
  /// Slot storage beginning at allocation offset zero.
  SlotPoolLayout slots;
  /// Aligned beginning of source admission and cumulative counters.
  std::size_t source_state_offset;
  /// Actual SourceState storage bytes.
  std::size_t source_state_bytes;
  /// Complete allocation including final alignment padding.
  std::size_t total_bytes;
};

/// Plain Device counter storage. Initialize before launch; live access uses
/// Device atomics. Overflow is sticky and quiescent reads clamp to UINT64_MAX.
struct alignas(8) DeviceCounterState {
  /// Private modulo accumulator, or an exact initial retained count.
  std::uint64_t value = 0;
  /// Zero initially; one permanently after any addition overflows.
  std::uint64_t exhausted = 0;
};

/// Two-range Device identity storage. Initialize before launch and never reset
/// while callers exist. A retained value belongs to this same live Producer.
struct alignas(8) DeviceAttemptState {
  /// Private count in the positive lower identity range.
  std::uint64_t lower;
  /// Private offset count in the upper identity range.
  std::uint64_t upper;
  /// Permanent transition into the upper range.
  std::uint64_t upper_range;

  /// Initialize exact history; allocation starts after last_identity.
  XPOOL_HOST_DEVICE_FN explicit constexpr DeviceAttemptState(std::uint64_t last_identity = 0) noexcept
      : lower(last_identity < (std::uint64_t{1} << 63) ? last_identity : (std::uint64_t{1} << 63) - 1),
        upper(last_identity >= (std::uint64_t{1} << 63) ? last_identity - ((std::uint64_t{1} << 63) - 1) : 0),
        upper_range(last_identity >= ((std::uint64_t{1} << 63) - 1) ? 1 : 0) {}
};

/// Borrowed Device source ledger. The owner validates initial conservation and
/// disjoint storage before launch. Host reads/copies require Device retirement.
struct alignas(8) DeviceSourceState {
  /// Active reservations and sticky stop reasons; accessed with Device atomics.
  std::uint64_t admission;
  /// Automatic full-width attempted-event identities.
  DeviceAttemptState attempts;
  /// Successful immutable commits, including events already consumed.
  DeviceCounterState committed{};
  /// Attempts observing every candidate occupied.
  DeviceCounterState dropped_full{};
  /// Successful retirement before commit.
  DeviceCounterState abandoned{};
  /// Rejected token operations, excluding duplicate commits.
  DeviceCounterState stale_token{};
  /// Repeated commit claims on the same generation.
  DeviceCounterState duplicate_commit{};

  /// Initialize identities and closure. The owner separately supplies retained
  /// outcome counts satisfying conservation; this is not crash recovery.
  XPOOL_HOST_DEVICE_FN explicit constexpr DeviceSourceState(std::uint64_t last_identity = 0) noexcept
      : admission(last_identity == ~std::uint64_t{0} ? ((std::uint64_t{1} << 62) | (std::uint64_t{1} << 63)) : 0),
        attempts(last_identity) {}

  /// Host-only checked slot/source geometry using the actual state type.
  /// This query allocates nothing and performs no CUDA operation.
  /// \throws std::invalid_argument for invalid dimensions/alignment and
  /// std::overflow_error for arithmetic overflow.
  static SourceLayout layout(std::size_t capacity, std::size_t allocation_alignment = 256);
};

static_assert(std::is_trivially_copyable_v<DeviceSourceState>);

/// A source with concurrent reservation, outcome and read calls. All storage
/// outlives API calls and held readers. At most kAttemptConcurrencyLimit
/// reservations overlap. Valid service paths allocate no memory, perform no
/// I/O and do not wait. C0's typed-error HostBuffer facade remains separate.
class HostEventSource {
public:
  /// Bind immutable provenance and initialized, disjoint storage. Retained
  /// history must satisfy attempted = committed + dropped_full + abandoned;
  /// all newly bound slots are Free. A retained history is not crash recovery.
  /// \throws std::invalid_argument for invalid source/storage/history and
  /// std::overflow_error for layout arithmetic. The future C0 facade translates
  /// these low-level errors into CollectionError.
  HostEventSource(BatchSource source, std::span<Event> events, std::span<SlotControl> slots, SlotPoolState &pool_state,
                  SourceState &state);

  HostEventSource(const HostEventSource &) = delete;
  HostEventSource &operator=(const HostEventSource &) = delete;

  /// Return one full-width positive sequence for Reserved/Full, closing at
  /// UINT64_MAX. Failed reservations carry zero tokens. A call admitted before
  /// concurrent stop may finish. Contention itself is never reported as Full.
  /// \throws std::overflow_error when reservation concurrency exceeds
  /// kAttemptConcurrencyLimit; this call allocates no event identity.
  ReserveResult try_reserve();

  /// Publish and account once. A rejected clock leaves the reservation intact.
  /// \throws std::invalid_argument for an Event from another clock domain.
  CommitStatus commit(WriteToken token, const Event &event);
  /// Account a successful abandon once; rejected calls are diagnostics.
  AbandonStatus abandon(WriteToken token);
  /// Acquire any committed event; pending lower sequences do not block it.
  ReadResult try_acquire();
  /// Retire read ownership only after the complete payload read/copy finishes.
  ReleaseStatus release(ReadToken token);
  /// Close admission concurrently; existing owners remain retireable.
  void stop();
  /// Inspect after every source API call quiesces; held reservations and read
  /// ownership may remain. Exact totals satisfy the attempt conservation law.
  /// \throws std::logic_error for nonquiescent/inconsistent admission or totals.
  SourceCounts quiescent_counts() const;
  /// Sample cumulative atomic lower bounds during service. Pending remains
  /// unknown and exact_totals is false; this method does not freeze producers.
  SourceProgress progress() const;
  /// Immutable provenance supplied by registration.
  const BatchSource &source() const noexcept { return source_; }
  /// Checked slot/source storage, including actual counter types and alignment.
  /// \throws std::invalid_argument for invalid dimensions/alignment and
  /// std::overflow_error for arithmetic overflow.
  static SourceLayout layout(std::size_t capacity, std::size_t allocation_alignment = 256);

private:
  const BatchSource source_;
  HostSlotPool pool_;
  SourceState &state_;
};

} // namespace xpool::devkit::timeline
