#pragma once

/// \file xpool/transport/observation.hpp
/// \brief C0 Transport observation prefix and control-plane descriptor.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include <cuda_runtime_api.h>

#include <xpool/devkit/timeline/contract.hpp>
#include <xpool/macros.hpp>
#include <xpool/utils/hex.hpp>

namespace xpool::transport {

/// The versioned public IPC prefix. The successful Instance claimant writes
/// before Mailbox publication; AtnAgent reads after observing Published and
/// copies the value before publishing a result. The Mailbox release/acquire
/// pair orders these accesses. Host inspection requires actual quiescence.
struct alignas(8) ObservationWord {
  /// Positive Operation sequence, or zero before the first successful Claim.
  std::uint64_t operation_sequence = 0;
};
static_assert(sizeof(ObservationWord) == 8);
static_assert(alignof(ObservationWord) == 8);
static_assert(offsetof(ObservationWord, operation_sequence) == 0);
static_assert(std::is_trivially_copyable_v<ObservationWord>);

/// Internal Operation evidence state, independent of production result codes.
/// These values are not C0 wire enums. Unavailable or conflicting input stops
/// this local identity domain; exhaustion additionally reports its exact cause.
enum class OperationStatus : std::uint32_t { Available, Unavailable, Conflict, IdentityExhausted };

/// Copied identity evidence for one protocol transition. Consumers retain this
/// value before mailbox reuse; they never fill an old event from the live word.
struct OperationFact {
  /// Positive identity when available; zero carries no Operation association.
  std::uint64_t sequence = 0;
  /// Why this transition has or lacks a usable identity.
  OperationStatus status = OperationStatus::Unavailable;
  /// Actual AtnAgent Device Producer owning this Endpoint, or zero if disabled.
  std::uint64_t endpoint_owner = 0;
  /// Immutable Endpoint allocation sequence, or zero if disabled.
  std::uint64_t endpoint_sequence = 0;
  /// Registered Generation reference, or zero if disabled.
  std::uint64_t generation = 0;
};

/// One endpoint's private counter, separate from the public eight-byte word.
/// The Instance's successful mailbox claimant exclusively calls claim(); the
/// AtnAgent's single mailbox owner exclusively calls observe() on its own state.
/// Mailbox release/acquire orders the word and successive claims. There is no
/// Event reservation, allocation, wait or retry here. Live Host reads are invalid.
class OperationState {
public:
  /// Initialize before any users. A retained last value supports a quiescent
  /// continuation, never crash recovery or resetting a live identity domain.
  XPOOL_HOST_DEVICE_FN explicit constexpr OperationState(std::uint64_t last_sequence = 0)
      : last_sequence_(last_sequence) {}
  /// Allocate only after a successful Claim, before its opening Hook and
  /// request publication. Exhaustion clears the shared word on every later
  /// Claim, retains the last issued identity and leaves production unchanged.
  XPOOL_HOST_DEVICE_FN OperationFact claim(ObservationWord &word) {
    if (status_ == OperationStatus::Available) {
      if (last_sequence_ == std::numeric_limits<std::uint64_t>::max())
        status_ = OperationStatus::IdentityExhausted;
      else
        ++last_sequence_;
    }
    word.operation_sequence = status_ == OperationStatus::Available ? last_sequence_ : 0;
    return {word.operation_sequence, status_};
  }
  /// Copy after Published is acquire-observed, before validation or any result
  /// publication. Zero/repeated/decreasing values never reuse a prior identity;
  /// the local domain remains stopped even if a later input looks valid.
  XPOOL_HOST_DEVICE_FN OperationFact observe(const ObservationWord &word) {
    if (status_ != OperationStatus::Available)
      return {0, status_};
    const auto sequence = word.operation_sequence;
    if (sequence == 0)
      status_ = OperationStatus::Unavailable;
    else if (sequence <= last_sequence_)
      status_ = OperationStatus::Conflict;
    else
      last_sequence_ = sequence;
    return {status_ == OperationStatus::Available ? sequence : 0, status_};
  }
  /// Last accepted/issued identity. Read only by the owner or after quiescence.
  XPOOL_HOST_DEVICE_FN std::uint64_t last_sequence() const noexcept { return last_sequence_; }
  /// Retained reason for domain stop; observing zero cannot infer peer exhaustion.
  XPOOL_HOST_DEVICE_FN OperationStatus status() const noexcept { return status_; }

private:
  std::uint64_t last_sequence_ = 0;
  OperationStatus status_ = OperationStatus::Available;
};
static_assert(std::is_trivially_copyable_v<OperationState>);

/// Borrowed process-local observation addresses and immutable pairing facts.
/// Both pointers are present together or both null. The Region owner retains
/// them until all kernel/Graph users retire; a live Host dereference is invalid.
/// This value is a launch argument or Resident table entry, never IPC payload.
struct ObservationView {
  /// Shared word whose accesses are ordered by the production mailbox.
  ObservationWord *word = nullptr;
  /// Private counter belonging to this endpoint's local process.
  OperationState *state = nullptr;
  /// Actual AtnAgent Device Producer from the admitted descriptor.
  std::uint64_t endpoint_owner = 0;
  /// Immutable positive allocation identity from the admitted descriptor.
  std::uint64_t endpoint_sequence = 0;
  /// Registered Generation for this pairing.
  std::uint64_t generation = 0;

  /// Disabled observation borrows no resources and touches no counters.
  XPOOL_HOST_DEVICE_FN bool empty() const noexcept { return state == nullptr; }
  /// Called by the successful Instance claimant before its opening Hook.
  XPOOL_HOST_DEVICE_FN OperationFact claim() const { return empty() ? OperationFact{} : identify(state->claim(*word)); }
  /// Called once by the AtnAgent after acquire-observing Published. Every
  /// subsequent Hook in that cycle uses the returned value, not this view.
  XPOOL_HOST_DEVICE_FN OperationFact observe() const {
    return empty() ? OperationFact{} : identify(state->observe(*word));
  }

private:
  XPOOL_HOST_DEVICE_FN OperationFact identify(OperationFact fact) const {
    fact.endpoint_owner = endpoint_owner;
    fact.endpoint_sequence = endpoint_sequence;
    fact.generation = generation;
    return fact;
  }
};
static_assert(std::is_trivially_copyable_v<ObservationView>);

/// Exact six-field C0 sibling of the existing Transport Arena handle. The
/// opaque CUDA handle stays in the control plane, never in timeline metadata.
struct ObservationRegion {
  /// Required C0 version; participants reject incompatible preparation.
  std::uint32_t version = xpool::devkit::timeline::kContractVersion;
  /// Registered AtnAgent Device Producer that allocated the Endpoint.
  std::uint64_t owner_producer = 0;
  /// Fresh positive sequence scoped to that Producer's entire lifetime.
  std::uint64_t endpoint_sequence = 0;
  /// Registered Generation reference at Arena creation.
  std::uint64_t generation = 0;
  /// Actual requested allocation bytes, including allocation alignment.
  std::uint64_t allocation_bytes = 0;
  /// Canonical hexadecimal CUDA IPC handle using the Arena convention.
  xpool::utils::hex::HexValue<cudaIpcMemHandle_t> handle{};
  /// Compare the complete immutable descriptor, including handle bytes.
  bool operator==(const ObservationRegion &) const = default;
};

/// Check presence against immutable policy and identities against the actual
/// preparation before CUDA attachment. Disabled identity requires null. The
/// expected owner is the remote AtnAgent Device Producer, not the importer.
/// This checks the public prefix extent; the Native storage owner additionally
/// checks the exact same-ABI allocation extent before opening a mapping.
/// \throws timeline::CollectionError metadata_unavailable on mismatch.
void validate_observation_region(const std::optional<ObservationRegion> &region, bool transport_identity_required,
                                 std::uint64_t generation, std::uint64_t owner_producer);

} // namespace xpool::transport
