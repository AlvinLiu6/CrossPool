#include <xpool/devkit/timeline/legacy.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <span>
#include <utility>

#include <xpool/devkit/timeline/digest.hpp>

namespace xpool::devkit::timeline::detail {

namespace {

constexpr auto bit(std::size_t index) { return std::uint64_t{1} << index; }

template <typename Slot>
MergeResult merge_slot(Slot &slot, const Event &event, std::span<const std::uint64_t> facts, std::uint64_t known,
                       std::optional<std::size_t> phase, bool conflict) {
  const auto occupied = slot.local_trace_id != 0;
  for (auto index = std::size_t{0}; index < slot.facts.size(); ++index) {
    if ((slot.known & known & bit(index)) != 0 && slot.facts[index] != facts[index])
      conflict = true;
  }
  if (phase && (slot.recorded & bit(*phase)) != 0 && slot.times[*phase] != event.raw_time_ns)
    conflict = true;
  if (conflict) {
    slot.flags |= static_cast<std::uint64_t>(LegacyFlag::Conflict);
    return MergeResult::Conflict;
  }
  const auto duplicate = occupied && (known & ~slot.known) == 0 && (!phase || (slot.recorded & bit(*phase)) != 0);
  slot.local_trace_id = event.local_trace_id;
  slot.clock_domain_id = event.clock_domain_id;
  for (auto index = std::size_t{0}; index < slot.facts.size(); ++index) {
    if ((known & bit(index)) != 0)
      slot.facts[index] = facts[index];
  }
  slot.known |= known;
  if (phase) {
    slot.times[*phase] = event.raw_time_ns;
    slot.recorded |= bit(*phase);
  }
  return duplicate ? MergeResult::Duplicate : MergeResult::Applied;
}

} // namespace

LegacyStore::LegacyStore(std::shared_ptr<MemoryBudget> budget, LegacyScope scope, std::size_t capacity)
    : budget_(std::move(budget)), scope_(scope), capacity_(capacity) {
  const auto stride = transport() ? sizeof(TransportLegacySlot) : sizeof(FabricLegacySlot);
  if (!budget_ || capacity == 0 || capacity > std::numeric_limits<std::size_t>::max() / stride ||
      capacity > std::numeric_limits<std::size_t>::max() / sizeof(LegacyProvenance) || scope.source.producer == 0 ||
      scope.source.clock_domain_id == 0 || scope.generation == 0 ||
      (transport() && (scope.endpoint_owner == 0 || scope.endpoint_sequence == 0))) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, scope.source.producer,
                          "xpool legacy retention sizing is invalid");
  }
  bytes_ = capacity * stride;
  provenance_bytes_ = capacity * sizeof(LegacyProvenance);
  storage_ =
      static_cast<std::byte *>(budget_->allocate_storage(bytes_, 256, MemoryKind::Snapshot, scope.source.producer));
  try {
    provenance_ = static_cast<LegacyProvenance *>(
        budget_->allocate_storage(provenance_bytes_, 256, MemoryKind::Snapshot, scope.source.producer));
    for (auto index = std::size_t{0}; index < capacity; ++index) {
      if (transport())
        std::construct_at(reinterpret_cast<TransportLegacySlot *>(storage_) + index);
      else
        std::construct_at(reinterpret_cast<FabricLegacySlot *>(storage_) + index);
    }
    for (auto index = std::size_t{0}; index < capacity; ++index)
      std::construct_at(provenance_ + index);
  } catch (...) {
    budget_->deallocate_storage(provenance_, provenance_bytes_, 256, MemoryKind::Snapshot);
    budget_->deallocate_storage(storage_, bytes_, 256, MemoryKind::Snapshot);
    throw;
  }
}

LegacyStore::~LegacyStore() {
  budget_->deallocate_storage(provenance_, provenance_bytes_, 256, MemoryKind::Snapshot);
  budget_->deallocate_storage(storage_, bytes_, 256, MemoryKind::Snapshot);
}

std::size_t LegacyStore::allocation_bytes() const {
  return MemoryBudget::rounded_bytes(bytes_, 256) + MemoryBudget::rounded_bytes(provenance_bytes_, 256);
}

bool LegacyStore::transport() const noexcept {
  return scope_.family == LegacyFamily::TransportInstance || scope_.family == LegacyFamily::TransportAtnAgent;
}

std::uint32_t LegacyStore::role() const noexcept {
  return scope_.family == LegacyFamily::TransportInstance   ? 1
         : scope_.family == LegacyFamily::TransportAtnAgent ? 2
                                                            : 0;
}

std::uint64_t LegacyStore::required(std::uint32_t owner_role) const noexcept {
  if (transport())
    return (bit(12) - 1) & ~(bit(0) | bit(9) | bit(10));
  // Generation/Endpoint and PE/topology also come from the registered report
  // domain. A projection requires its own dynamic Invocation/request facts.
  auto mask = (bit(6) - 1) | bit(15);
  if (owner_role == 2)
    mask |= bit(6) | bit(7) | bit(8);
  if (owner_role == 3 && scope_.scheduler == LegacyPolicy::Fifo)
    mask |= bit(11);
  if (owner_role == 4)
    mask |= bit(9) | bit(10) | bit(12) | bit(13);
  return mask;
}

MergeResult LegacyStore::merge(const Event &event) {
  const auto base = transport() ? 100U : event.role == 2 ? 200U : event.role == 3 ? 300U : 400U;
  const auto count = transport() ? 11U : event.role == 2 ? 6U : event.role == 3 ? 8U : 9U;
  auto phase = std::optional<std::size_t>{};
  if (transport() && event.kind == 150 && event.data0 >= 100 && event.data0 <= 110)
    phase = event.data0 - 100;
  else if (event.kind >= base && event.kind - base < count)
    phase = event.kind - base;
  else if (!(event.kind >= 504 && event.kind <= 508) || (!transport() && event.role != 4))
    return MergeResult::Unrelated;
  if ((event.presence & bit(8)) == 0 || event.local_trace_id == 0)
    return MergeResult::Unrelated;
  if (attempts_known_ && event.local_trace_id > attempted_) {
    throw CollectionError(CollectionErrorCode::Transfer, scope_.source.producer,
                          "xpool legacy event exceeds the retired operation counter");
  }
  highest_ = std::max(highest_, event.local_trace_id);
  if (event.local_trace_id > capacity_)
    return MergeResult::BeyondCapacity;
  auto boundary_hash = std::array<std::byte, 32>{};
  auto &provenance = provenance_[event.local_trace_id - 1];
  auto byte_conflict = false;
  auto reused_identity = false;
  if (phase) {
    auto normalized = event;
    normalized.event_sequence = 0;
    auto bytes = std::array<std::byte, Event::encoded_bytes>{};
    normalized.encode(bytes);
    boundary_hash = sha256(bytes);
    byte_conflict = provenance.times[*phase] != 0 && provenance.boundaries[*phase] != boundary_hash;
    for (std::size_t i = 0; i < provenance.times.size(); ++i)
      reused_identity =
          reused_identity || (provenance.times[i] == event.event_sequence && (i != *phase || byte_conflict));
    byte_conflict = byte_conflict || reused_identity;
  }
  auto facts = std::array<std::uint64_t, 18>{};
  auto known = std::uint64_t{0};
  const auto set = [&](std::size_t index, std::uint64_t value) {
    facts[index] = value;
    known |= bit(index);
  };
  const auto present = [&](std::size_t index) { return (event.presence & bit(index)) != 0; };
  if (present(0))
    set(0, event.generation);
  auto conflict = byte_conflict || event.clock_domain_id != scope_.source.clock_domain_id ||
                  (transport() ? event.role != role() : event.role < 2 || event.role > 4) ||
                  (present(0) && event.generation != scope_.generation);
  const auto remember = [&](MergeResult result) {
    if (result == MergeResult::Conflict && !conflict_) {
      auto conflict = ReconstructionFailure{ReconstructionErrorCode::IdentityConflict, scope_.source.producer,
                                            event.event_sequence, std::nullopt};
      if (const auto previous = read(event.local_trace_id)) {
        auto fact_conflict = false;
        for (auto index = std::size_t{0}; index < facts.size(); ++index) {
          if ((previous->known & known & bit(index)) != 0 && previous->facts[index] != facts[index]) {
            conflict.previous_event_sequence = previous->sources.facts[index];
            fact_conflict = true;
            break;
          }
        }
        if (!fact_conflict && phase && (previous->recorded & bit(*phase)) != 0 &&
            (previous->times[*phase] != event.raw_time_ns || byte_conflict)) {
          conflict.code = ReconstructionErrorCode::EventConflict;
          conflict.previous_event_sequence = previous->sources.times[*phase];
        }
        // Changed bytes under a retained Event identity are an Event conflict,
        // including a different boundary or changed repeated scalar facts.
        if (reused_identity) {
          conflict.code = ReconstructionErrorCode::EventConflict;
          conflict.previous_event_sequence = event.event_sequence;
        }
      }
      conflict_ = conflict;
    } else if (result != MergeResult::Conflict) {
      auto &sources = provenance_[event.local_trace_id - 1];
      const auto retain = [&](std::uint64_t &identity) {
        if (identity == 0 || event.event_sequence < identity)
          identity = event.event_sequence;
      };
      for (auto index = std::size_t{0}; index < sources.facts.size(); ++index)
        if ((known & bit(index)) != 0)
          retain(sources.facts[index]);
      if (phase) {
        retain(sources.times[*phase]);
        sources.boundaries[*phase] = boundary_hash;
      }
      retain(sources.local_identity);
    }
    return result;
  };
  if (transport()) {
    if (present(1)) {
      set(1, event.endpoint_owner);
      set(2, event.endpoint_sequence);
      conflict = conflict || event.endpoint_owner != scope_.endpoint_owner ||
                 event.endpoint_sequence != scope_.endpoint_sequence;
    }
    if (present(2))
      set(3, event.operation_sequence);
    if (present(10)) {
      set(4, event.layer_ordinal);
      set(5, event.payload_rows);
      set(7, event.output_requirement);
    }
    if (present(9)) {
      set(4, event.layer_ordinal);
      set(5, event.payload_rows);
      set(6, event.forward_mode);
      set(7, event.output_requirement);
      set(8, event.dp_row_layout);
    }
    if (event.kind == 150) {
      set(4, event.data3);
      set(5, event.data4);
      set(6, static_cast<std::uint32_t>(event.data1));
      set(7, event.data1 >> 32);
      set(8, static_cast<std::uint32_t>(event.data2));
    }
    if (present(16))
      set(9, event.instance_rank);
    if (present(15))
      set(10, event.result_code);
    set(11, event.role);
    auto &slot = reinterpret_cast<TransportLegacySlot *>(storage_)[event.local_trace_id - 1];
    const auto was_empty = slot.local_trace_id == 0;
    const auto invalid_request = static_cast<std::uint64_t>(LegacyFlag::InvalidRequest);
    if (phase && !was_empty && (slot.known & bit(6)) != 0)
      conflict = conflict || ((slot.flags & invalid_request) != 0) != (event.kind == 150);
    const auto result = merge_slot(slot, event, facts, known, phase, conflict);
    if (result != MergeResult::Conflict && event.kind == 150)
      slot.flags |= invalid_request;
    if (was_empty && slot.local_trace_id != 0)
      ++retained_;
    return remember(result);
  }
  if (present(5)) {
    set(1, event.instance_index);
    set(2, event.invocation_sequence);
  }
  if (present(10)) {
    set(3, event.layer_ordinal);
    set(4, event.payload_rows);
    set(5, event.output_requirement);
  }
  if (present(11))
    set(6, event.dp_rank_payload_rows);
  if (present(9)) {
    set(7, event.forward_mode);
    set(8, event.dp_row_layout);
  }
  if (present(6)) {
    set(9, event.lane_index);
    set(10, event.lease_sequence);
  }
  if (present(13))
    set(11, event.ready_ticket);
  if (present(12))
    set(12, event.payload_row_capacity);
  if (present(14))
    set(13, event.delivery);
  if (present(17))
    set(14, static_cast<std::uint64_t>(static_cast<std::int64_t>(event.pe)));
  set(15, event.role);
  auto &slot = reinterpret_cast<FabricLegacySlot *>(storage_)[event.local_trace_id - 1];
  const auto was_empty = slot.local_trace_id == 0;
  const auto result = merge_slot(slot, event, facts, known, phase, conflict);
  if (was_empty && slot.local_trace_id != 0)
    ++retained_;
  return remember(result);
}

void LegacyStore::observe_attempts(std::uint64_t total) {
  if (total < highest_ || total < attempted_) {
    throw CollectionError(CollectionErrorCode::Transfer, scope_.source.producer,
                          "xpool legacy operation counter regressed");
  }
  attempted_ = total;
  attempts_known_ = true;
}

std::uint64_t LegacyStore::flags(LegacyOperation &operation) const {
  auto result = operation.flags;
  if (!transport() && operation.facts[15] == 3 && !scope_.scheduler)
    result |= static_cast<std::uint64_t>(LegacyFlag::MissingFacts);
  const auto required_facts = required(static_cast<std::uint32_t>(operation.facts[15]));
  if ((operation.known & required_facts) != required_facts)
    result |= static_cast<std::uint64_t>(LegacyFlag::MissingFacts);
  if (transport() && (operation.known & bit(10)) == 0)
    result |= static_cast<std::uint64_t>(LegacyFlag::UnknownResult);
  for (auto index = std::size_t{0}; index < operation.times.size(); ++index) {
    if ((operation.recorded & bit(index)) != 0 && operation.times[index] == 0)
      result |= static_cast<std::uint64_t>(LegacyFlag::ZeroTimestamp);
  }
  const auto check = [&](std::size_t begin, std::size_t end) {
    if ((operation.recorded & (bit(begin) | bit(end))) == (bit(begin) | bit(end)) &&
        operation.times[end] < operation.times[begin]) {
      result |= static_cast<std::uint64_t>(LegacyFlag::TemporalConflict);
    }
  };
  if (transport()) {
    check(0, 1);
    check(1, 2);
    check(3, 6);
    check(4, 5);
    check(7, 8);
    check(8, 9);
    check(0, 9);
    check(0, 10);
  } else if (operation.facts[15] == 2) {
    check(1, 2);
    check(2, 4);
    check(0, 5);
  } else if (operation.facts[15] == 3) {
    check(0, 1);
    check(3, 4);
    check(1, 7);
    check(0, 7);
  } else {
    check(0, 1);
    check(4, 5);
    check(0, 8);
  }
  const auto ordered = [&](const auto &phases) {
    auto previous = operation.times.size();
    for (const auto phase : phases) {
      if ((operation.recorded & bit(phase)) == 0)
        continue;
      if (previous < operation.times.size())
        check(previous, phase);
      previous = phase;
    }
  };
  if (transport()) {
    if (scope_.family == LegacyFamily::TransportInstance)
      ordered(std::array<std::size_t, 7>{0, 1, 2, 7, 8, 9, 10});
    else
      ordered(std::array<std::size_t, 4>{3, 4, 5, 6});
  } else if (operation.facts[15] == 2)
    ordered(std::array<std::size_t, 6>{0, 1, 2, 3, 4, 5});
  else if (operation.facts[15] == 3)
    ordered(std::array<std::size_t, 8>{0, 1, 2, 3, 4, 5, 6, 7});
  else
    ordered(std::array<std::size_t, 9>{0, 1, 3, 4, 2, 5, 6, 7, 8});
  return result;
}

std::optional<LegacyOperation> LegacyStore::read(std::uint64_t id) const {
  if (id == 0 || id > capacity_)
    return std::nullopt;
  auto result = LegacyOperation{};
  const auto copy = [&result](const auto &slot) {
    result.local_trace_id = slot.local_trace_id;
    result.clock_domain_id = slot.clock_domain_id;
    result.known = slot.known;
    result.recorded = slot.recorded;
    result.flags = slot.flags;
    std::ranges::copy(slot.facts, result.facts.begin());
    std::ranges::copy(slot.times, result.times.begin());
  };
  if (transport())
    copy(reinterpret_cast<const TransportLegacySlot *>(storage_)[id - 1]);
  else
    copy(reinterpret_cast<const FabricLegacySlot *>(storage_)[id - 1]);
  if (result.local_trace_id == 0)
    return std::nullopt;
  result.sources = provenance_[id - 1];
  result.flags = flags(result);
  return result;
}

LegacyCounts LegacyStore::counts() const noexcept {
  const auto sequence = attempts_known_ ? attempted_ : highest_;
  return {sequence, sequence > capacity_ ? sequence - capacity_ : 0, retained_,
          attempts_known_ ? std::optional<std::uint64_t>{std::min<std::uint64_t>(sequence, capacity_) - retained_}
                          : std::nullopt,
          attempts_known_};
}

} // namespace xpool::devkit::timeline::detail
