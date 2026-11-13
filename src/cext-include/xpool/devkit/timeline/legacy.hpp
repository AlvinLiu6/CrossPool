#pragma once

/// \file xpool/devkit/timeline/legacy.hpp
/// \brief Internal bounded legacy merge state; snapshot adapters own readout.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <xpool/devkit/timeline/resource.hpp>
#include <xpool/devkit/timeline/source.hpp>

namespace xpool::devkit::timeline::detail {

/// \cond INTERNAL

enum class LegacyFamily { TransportInstance, TransportAtnAgent, Fabric };
enum class LegacyPolicy { Fifo, Random };
enum class ReconstructionErrorCode { InvalidEvent, UnknownKind, IdentityConflict, EventConflict };
constexpr std::string_view reconstruction_error_name(ReconstructionErrorCode code) noexcept {
  switch (code) {
  case ReconstructionErrorCode::InvalidEvent:
    return "invalid_event";
  case ReconstructionErrorCode::UnknownKind:
    return "unknown_kind";
  case ReconstructionErrorCode::IdentityConflict:
    return "identity_conflict";
  case ReconstructionErrorCode::EventConflict:
    return "event_conflict";
  }
  return {};
}
struct ReconstructionFailure {
  ReconstructionErrorCode code;
  std::uint64_t producer;
  std::uint64_t event_sequence;
  std::optional<std::uint64_t> previous_event_sequence;
  // Borrowed static validation text, never an Event-owned mutable string.
  std::string_view detail = {};
};
class ReconstructionError : public c10::Error {
public:
  explicit ReconstructionError(ReconstructionFailure conflict)
      : c10::Error(
            std::string(reconstruction_error_name(conflict.code)) + ": " +
            std::string(conflict.detail.empty() ? "retained reconstruction evidence disagrees" : conflict.detail)),
        context_(conflict) {}
  const ReconstructionFailure &context() const noexcept { return context_; }

private:
  const ReconstructionFailure context_;
};
enum class MergeResult { Applied, Duplicate, Conflict, BeyondCapacity, Unrelated };
enum class LegacyFlag : std::uint64_t {
  Conflict = 1,
  MissingFacts = 2,
  UnknownResult = 4,
  ZeroTimestamp = 8,
  TemporalConflict = 16,
  InvalidRequest = 32
};

struct LegacyScope {
  BatchSource source;
  LegacyFamily family;
  std::uint64_t generation;
  std::uint64_t endpoint_owner = 0;
  std::uint64_t endpoint_sequence = 0;
  std::optional<LegacyPolicy> scheduler = std::nullopt;
};

// Fixed C0 reconstruction slot strides, including masks and raw timestamps.
// Fact arrays have a family-specific mapping documented on LegacyOperation.
struct TransportLegacySlot {
  std::uint64_t local_trace_id = 0;
  std::uint64_t clock_domain_id = 0;
  std::uint64_t known = 0;
  std::array<std::uint64_t, 11> times{};
  std::array<std::uint64_t, 12> facts{};
  std::uint64_t recorded = 0;
  std::uint64_t flags = 0;
  std::array<std::uint64_t, 4> reserved{};
};

struct FabricLegacySlot {
  std::uint64_t local_trace_id = 0;
  std::uint64_t clock_domain_id = 0;
  std::uint64_t known = 0;
  std::array<std::uint64_t, 9> times{};
  std::array<std::uint64_t, 18> facts{};
  std::uint64_t recorded = 0;
  std::uint64_t flags = 0;
  std::array<std::uint64_t, 16> reserved{};
};

static_assert(sizeof(TransportLegacySlot) == 256);
static_assert(sizeof(FabricLegacySlot) == 384);

// Separate bounded report storage, charged in addition to the frozen C0
// reconstruction strides. Each finite field retains one canonical Event ID;
// the Producer is fixed by the view domain and is never inferred from a PE.
struct LegacyProvenance {
  std::array<std::uint64_t, 18> facts{};
  std::array<std::uint64_t, 11> times{};
  std::uint64_t local_identity = 0;
  // Finite protocol-boundary fingerprints exclude only Event sequence, so
  // equal repeats can retain minimum provenance while changed context/facts fail.
  std::array<std::array<std::byte, 32>, 11> boundaries{};
};

// Internal copy used by a future Native Record adapter, not an A/B diagnostic
// result or a public JSON extension. Presence masks remain authoritative.
struct LegacyOperation {
  std::uint64_t local_trace_id = 0;
  std::uint64_t clock_domain_id = 0;
  std::uint64_t known = 0;
  std::uint64_t recorded = 0;
  std::uint64_t flags = 0;
  std::array<std::uint64_t, 11> times{};
  // Transport: generation, endpoint owner/sequence, operation, layer, rows,
  // forward mode, output requirement, DP layout, rank, result, role.
  // Fabric: generation, instance, invocation, layer, rows, output requirement,
  // DP rows, forward mode, DP layout, lane, lease, ticket, row capacity,
  // delivery, PE, role; final two values are reserved and remain absent.
  std::array<std::uint64_t, 18> facts{};
  LegacyProvenance sources{};
};

struct LegacyCounts {
  std::uint64_t sequence;
  std::uint64_t capacity_dropped;
  std::uint64_t retained;
  std::optional<std::uint64_t> unobserved;
  bool attempts_known;
};

// One serialized reconstruction lane; reads require that lane to retire.
// Source Events are semantically validated and identity-checked upstream.
// Semantic duplicate detection needs no unbounded Event-ID history here.
class LegacyStore {
public:
  LegacyStore(std::shared_ptr<MemoryBudget> budget, LegacyScope scope, std::size_t capacity);
  ~LegacyStore();
  LegacyStore(const LegacyStore &) = delete;
  LegacyStore &operator=(const LegacyStore &) = delete;
  MergeResult merge(const Event &event);
  // Original record-begin counter after producer retirement, independent of
  // retained Event count. It must cover every observed local trace identity.
  void observe_attempts(std::uint64_t total);
  // Production resumed beyond the previous cutoff, or the original counter is
  // no longer exact. Retain its monotonic lower bound but clear availability.
  void forget_attempts() noexcept { attempts_known_ = false; }
  std::optional<LegacyOperation> read(std::uint64_t local_trace_id) const;
  LegacyCounts counts() const noexcept;
  const LegacyScope &scope() const noexcept { return scope_; }
  std::size_t capacity() const noexcept { return capacity_; }
  const std::optional<ReconstructionFailure> &conflict() const noexcept { return conflict_; }
  std::size_t reconstruction_bytes() const noexcept { return bytes_; }
  std::size_t provenance_bytes() const noexcept { return provenance_bytes_; }
  std::size_t allocation_bytes() const;

private:
  bool transport() const noexcept;
  std::uint32_t role() const noexcept;
  std::uint64_t required(std::uint32_t role) const noexcept;
  std::uint64_t flags(LegacyOperation &operation) const;
  const std::shared_ptr<MemoryBudget> budget_;
  const LegacyScope scope_;
  const std::size_t capacity_;
  std::size_t bytes_ = 0;
  std::byte *storage_ = nullptr;
  std::size_t provenance_bytes_ = 0;
  LegacyProvenance *provenance_ = nullptr;
  std::uint64_t highest_ = 0;
  std::uint64_t attempted_ = 0;
  std::uint64_t retained_ = 0;
  bool attempts_known_ = false;
  std::optional<ReconstructionFailure> conflict_ = std::nullopt;
};

/// \endcond

} // namespace xpool::devkit::timeline::detail
