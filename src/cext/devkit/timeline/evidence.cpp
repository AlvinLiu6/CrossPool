#include <xpool/devkit/timeline/evidence.hpp>

#include <algorithm>
#include <limits>
#include <memory>

namespace xpool::devkit::timeline {

namespace {

constexpr auto bit(std::size_t index) { return std::uint64_t{1} << index; }
constexpr auto scalar_count = std::size_t{16};
constexpr auto boundary_offset = std::size_t{32};

void reason(SnapshotEvidenceEntry &entry, EvidenceReason value) { entry.quality |= static_cast<std::uint32_t>(value); }

SnapshotEvidenceEntry entry(const detail::LegacyOperation &operation, const detail::LegacyScope &scope,
                            const SnapshotMetadata &metadata) {
  const auto transport = scope.family != detail::LegacyFamily::Fabric;
  const auto owner_role = static_cast<std::uint32_t>(transport ? operation.facts[11] : operation.facts[15]);
  auto result =
      SnapshotEvidenceEntry{.local_trace_id = operation.local_trace_id, .role = static_cast<SnapshotRole>(owner_role)};
  const auto missing_facts = (operation.flags & static_cast<std::uint64_t>(detail::LegacyFlag::MissingFacts)) != 0;
  const auto anomaly = (operation.flags & static_cast<std::uint64_t>(detail::LegacyFlag::InvalidRequest)) != 0;
  const auto retained = !missing_facts && !anomaly;
  result.projection_status = retained ? ProjectionStatus::Retained : ProjectionStatus::Omitted;
  result.evidence_status = EvidenceStatus::Partial;
  if (transport && (operation.known & (bit(1) | bit(2) | bit(3))) == (bit(1) | bit(2) | bit(3))) {
    result.operation = OperationReference{operation.facts[1], operation.facts[2], operation.facts[3]};
  } else if (!transport && (operation.known & (bit(0) | bit(1) | bit(2))) == (bit(0) | bit(1) | bit(2))) {
    result.invocation = InvocationReference{operation.facts[0], operation.facts[1], operation.facts[2]};
  }
  const auto event_source = [&](std::size_t field, std::uint64_t identity) {
    if (identity == 0) {
      result.unknown_fields |= bit(field);
      return;
    }
    result.sources[field] = identity;
    result.source_fields |= bit(field);
  };
  const auto scalar = [&](std::size_t field, std::size_t fact) {
    event_source(field, (operation.known & bit(fact)) != 0 ? operation.sources.facts[fact] : 0);
  };
  const auto metadata_source = [&](std::size_t field, std::uint64_t selector) {
    result.sources[field] = selector;
    result.source_fields |= bit(field);
    result.metadata_fields |= bit(field);
  };
  event_source(0, operation.sources.local_identity);
  scalar(1, transport ? 4 : 3);
  scalar(2, transport ? 5 : 4);
  scalar(3, transport ? 7 : 5);
  if (transport) {
    scalar(4, 6);
    scalar(5, 8);
    scalar(6, 10);
    if ((operation.known & bit(10)) == 0)
      reason(result, EvidenceReason::UnknownResult);
  } else {
    scalar(7, 1);
    scalar(8, 2);
    scalar(9, 9);
    scalar(10, 10);
    if (owner_role == 2) {
      scalar(4, 7);
      scalar(5, 8);
      scalar(11, 6);
    } else
      result.not_applicable_fields |= bit(5);
    if (owner_role == 3) {
      if (scope.scheduler)
        metadata_source(14, 1);
      else
        result.unknown_fields |= bit(14);
      if (scope.scheduler == detail::LegacyPolicy::Fifo)
        scalar(15, 11);
      else if (scope.scheduler == detail::LegacyPolicy::Random)
        result.not_applicable_fields |= bit(15);
      else
        result.unknown_fields |= bit(15);
    } else if (owner_role == 4) {
      scalar(12, 12);
      scalar(13, 13);
    }
  }
  const auto count = transport ? 11U : owner_role == 2 ? 6U : owner_role == 3 ? 8U : 9U;
  const auto all = static_cast<std::uint32_t>((bit(count) - 1));
  const auto observed = static_cast<std::uint32_t>(operation.recorded);
  auto required = std::uint32_t{0};
  auto inapplicable = std::uint32_t{0};
  auto unknown = std::uint32_t{0};
  const auto decision_event = [&](std::size_t phase, std::uint64_t identity) {
    event_source(boundary_offset + phase, identity);
  };
  const auto decision_metadata = [&](std::size_t phase, std::uint64_t selector) {
    metadata_source(boundary_offset + phase, selector);
  };
  if (transport && owner_role == 1) {
    constexpr auto own = static_cast<std::uint32_t>(bit(0) | bit(1) | bit(2) | bit(7) | bit(8) | bit(9) | bit(10));
    inapplicable = all & ~own;
    const auto result_known = (operation.known & bit(10)) != 0;
    if ((observed & bit(9)) != 0) {
      required = own & ~static_cast<std::uint32_t>(bit(10));
      inapplicable |= bit(10);
      decision_event(10, operation.sources.times[9]);
    } else if ((observed & bit(10)) != 0 || (result_known && operation.facts[10] != 0)) {
      unknown = own & ~observed & ~static_cast<std::uint32_t>(bit(9));
      inapplicable |= bit(9);
      decision_event(9, (observed & bit(10)) != 0 ? operation.sources.times[10] : operation.sources.facts[10]);
    } else {
      required = own & ~static_cast<std::uint32_t>(bit(8) | bit(9) | bit(10));
      if (result_known) {
        required |= bit(8) | bit(9);
        inapplicable |= bit(10);
        decision_event(10, operation.sources.facts[10]);
      } else
        unknown = bit(8) | bit(9) | bit(10);
    }
  } else if (transport) {
    required = bit(3) | bit(4) | bit(5) | bit(6);
    inapplicable = all & ~required;
    if ((operation.known & bit(10)) != 0 && operation.facts[10] != 0) {
      unknown = static_cast<std::uint32_t>(bit(4) | bit(5)) & ~observed;
      required &= ~unknown;
    }
  } else if (owner_role != 4)
    required = all;
  else {
    required = bit(0) | bit(1) | bit(4) | bit(5) | bit(8);
    unknown = bit(2) | bit(3) | bit(6) | bit(7);
    if (metadata.routing_not_applicable) {
      inapplicable |= bit(2) | bit(3);
      decision_metadata(2, 2);
      decision_metadata(3, 2);
    }
    if ((operation.known & bit(13)) != 0 && operation.facts[13] == 1) {
      inapplicable |= bit(6) | bit(7);
      decision_event(6, operation.sources.facts[13]);
      decision_event(7, operation.sources.facts[13]);
    } else if ((operation.known & bit(13)) != 0 && (operation.facts[13] == 2 || operation.facts[13] == 3)) {
      if ((operation.known & bit(4)) != 0 && operation.facts[4] == 0) {
        inapplicable |= bit(6) | bit(7);
        decision_event(6, operation.sources.facts[4]);
        decision_event(7, operation.sources.facts[4]);
      } else if (metadata.ffnagent_count == 1) {
        inapplicable |= bit(6) | bit(7);
        decision_metadata(6, 3);
        decision_metadata(7, 3);
      }
    }
  }
  // Retained boundaries prove their own applicability, including raw zero.
  required |= observed;
  inapplicable &= ~observed;
  unknown &= ~(required | inapplicable);
  for (auto phase = std::size_t{0}; phase < count; ++phase) {
    if ((observed & bit(phase)) != 0) {
      result.source_fields &= ~bit(boundary_offset + phase);
      result.metadata_fields &= ~bit(boundary_offset + phase);
    }
    const auto field = scalar_count + phase;
    if ((inapplicable & bit(phase)) != 0)
      result.not_applicable_fields |= bit(field);
    else
      event_source(field, (observed & bit(phase)) != 0 ? operation.sources.times[phase] : 0);
    if ((observed & bit(phase)) != 0 && operation.times[phase] == 0)
      reason(result, EvidenceReason::LegacyZeroTimestamp);
  }
  for (auto field = std::size_t{0}; field < result.sources.size(); ++field) {
    if ((result.metadata_fields & bit(field)) == 0)
      continue;
    const auto selector = result.sources[field];
    const auto &citation = selector == 1 ? metadata.scheduler : selector == 2 ? metadata.routing : metadata.topology;
    if (citation.producer == 0 || citation.revision == 0 || citation.path_size == 0 ||
        citation.path_size > citation.path_bytes.size()) {
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, scope.source.producer,
                            "xpool snapshot decision requires its exact metadata citation");
    }
  }
  result.missing_boundaries = required & ~observed;
  result.unknown_applicability = unknown;
  if (result.missing_boundaries != 0)
    reason(result, EvidenceReason::EventMissing);
  if (unknown != 0)
    reason(result, EvidenceReason::ApplicabilityUnknown);
  if (missing_facts)
    reason(result, EvidenceReason::LegacyFactMissing);
  if (anomaly) {
    reason(result, EvidenceReason::InvalidRequest);
    auto raw = TransportAnomalyEvidence{.layer_ordinal = operation.facts[4],
                                        .payload_rows = operation.facts[5],
                                        .forward_mode = static_cast<std::uint32_t>(operation.facts[6]),
                                        .output_requirement = static_cast<std::uint32_t>(operation.facts[7]),
                                        .dp_row_layout = static_cast<std::uint32_t>(operation.facts[8])};
    constexpr auto phases = std::array<std::size_t, 4>{6, 7, 9, 10};
    if ((operation.known & bit(10)) != 0) {
      raw.result_code = static_cast<std::uint32_t>(operation.facts[10]);
      for (std::size_t index = 0; index < phases.size(); ++index)
        if ((observed & bit(phases[index])) != 0) {
          raw.terminal_mask |= std::uint32_t{1} << index;
          raw.terminal_times[index] = operation.times[phases[index]];
        }
    }
    result.anomaly = raw;
  }
  const auto temporal = (operation.flags & static_cast<std::uint64_t>(detail::LegacyFlag::TemporalConflict)) != 0;
  if (temporal)
    reason(result, EvidenceReason::TemporalConflict);
  if (!missing_facts && result.unknown_fields == 0 && unknown == 0 && !temporal)
    result.evidence_status = EvidenceStatus::Complete;
  return result;
}

} // namespace

EvidenceMetadataSource EvidenceMetadataSource::make(std::uint64_t producer, std::uint64_t revision,
                                                    std::string_view path) {
  if (producer == 0 || revision == 0 || path.empty())
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, producer,
                          "xpool evidence metadata source is missing");
  auto result = EvidenceMetadataSource{};
  if (path.size() > result.path_bytes.size())
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer,
                          "xpool evidence metadata path exceeds its declared capacity");
  result.producer = producer;
  result.revision = revision;
  result.path_size = path.size();
  std::ranges::copy(path, result.path_bytes.begin());
  return result;
}

std::shared_ptr<SnapshotEvidenceReport> SnapshotEvidenceReport::create(std::shared_ptr<MemoryBudget> budget,
                                                                       const detail::LegacyStore &store,
                                                                       SnapshotMetadata metadata) {
  return make_budgeted_shared<SnapshotEvidenceReport>(budget, MemoryKind::Snapshot, store.scope().source.producer,
                                                      ConstructionKey{}, budget, store, metadata);
}

SnapshotEvidenceReport::SnapshotEvidenceReport(ConstructionKey, std::shared_ptr<MemoryBudget> budget,
                                               const detail::LegacyStore &store, SnapshotMetadata metadata)
    : source(store.scope().source),
      observer(store.scope().family == detail::LegacyFamily::Fabric ? SnapshotObserver::Fabric
                                                                    : SnapshotObserver::Transport),
      generation(store.scope().generation),
      endpoint(observer == SnapshotObserver::Transport
                   ? std::optional{std::array{store.scope().endpoint_owner, store.scope().endpoint_sequence}}
                   : std::nullopt),
      site(observer == SnapshotObserver::Transport
               ? std::optional{store.scope().family == detail::LegacyFamily::TransportInstance ? SnapshotRole::Instance
                                                                                               : SnapshotRole::AtnAgent}
               : std::nullopt),
      capacity(store.capacity()),
      sequence(store.counts().attempts_known ? std::optional{store.counts().sequence} : std::nullopt),
      dropped(store.counts().attempts_known ? std::optional{store.counts().capacity_dropped} : std::nullopt),
      unobserved_operations(store.counts().unobserved), budget_(std::move(budget)), metadata_(metadata) {
  if (store.conflict())
    throw detail::ReconstructionError{*store.conflict()};
  if (!budget_ || capacity > std::numeric_limits<std::size_t>::max() / sizeof(SnapshotEvidenceEntry)) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer,
                          "xpool snapshot report capacity overflows");
  }
  entries_charge_ = budget_->reserve(
      MemoryKind::Snapshot,
      MemoryBudget::array_bytes(capacity, sizeof(SnapshotEvidenceEntry), alignof(SnapshotEvidenceEntry)),
      source.producer);
  entries_.reserve(capacity);
  const auto limit = sequence ? std::min(*sequence, capacity) : capacity;
  for (auto id = std::uint64_t{1}; id <= limit; ++id) {
    const auto operation = store.read(id);
    if (!operation && !sequence)
      continue;
    auto value = operation ? entry(*operation, store.scope(), metadata_) : SnapshotEvidenceEntry{};
    value.local_trace_id = id;
    if (!operation) {
      value.role = site;
      reason(value, EvidenceReason::OperationUnobserved);
    }
    entries_.push_back(value);
  }
}

} // namespace xpool::devkit::timeline
