#include <xpool/devkit/timeline/legacyjson.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <utility>

namespace xpool::devkit::timeline {
namespace {

using detail::JsonOutput;
using TransportRecord = xpool::devkit::transport_observer::Record;
using FabricRecord = xpool::devkit::fabric_observer::Record;
using TransportKind = xpool::hooks::TransportProtocolEventKind;
using AtnKind = xpool::hooks::FabricAtnAgentProtocolEvent::Kind;
using CoordinatorKind = xpool::hooks::FabricCoordinatorProtocolEvent::Kind;
using FfnKind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind;

struct Phase {
  std::string_view name;
  std::size_t start;
  std::size_t end;
};
constexpr std::array transport_phases = {Phase{"request_staging", 0, 1},
                                         Phase{"request_publication", 1, 2},
                                         Phase{"request_processing", 3, 6},
                                         Phase{"execution", 4, 5},
                                         Phase{"output_copy", 7, 8},
                                         Phase{"result_acknowledgement", 8, 9},
                                         Phase{"result_acknowledged_total", 0, 9},
                                         Phase{"closed_total", 0, 10}};
constexpr std::array atn_phases = {Phase{"admission_wait", 1, 2}, Phase{"output_wait", 2, 4}, Phase{"total", 0, 5}};
constexpr std::array coordinator_phases = {Phase{"scheduling", 0, 1}, Phase{"execution", 3, 4},
                                           Phase{"active_total", 1, 7}, Phase{"total", 0, 7}};
constexpr std::array ffn_phases = {Phase{"input_wait", 0, 1}, Phase{"compute", 4, 5}, Phase{"total", 0, 8}};

[[noreturn]] void invalid(std::uint64_t producer, std::string_view message) {
  throw CollectionError(CollectionErrorCode::MetadataUnavailable, producer, std::string(message));
}
std::string_view spelling(xpool::ffn::ForwardMode value) {
  switch (value) {
  case xpool::ffn::ForwardMode::Prefill:
    return "prefill";
  case xpool::ffn::ForwardMode::Decode:
    return "decode";
  case xpool::ffn::ForwardMode::Idle:
    return "idle";
  }
  invalid(0, "xpool legacy forward mode is invalid");
}
std::string_view spelling(xpool::ffn::OutputRequirement value) {
  switch (value) {
  case xpool::ffn::OutputRequirement::PerRankComplete:
    return "per_rank_complete";
  case xpool::ffn::OutputRequirement::GroupSumComplete:
    return "group_sum_complete";
  }
  invalid(0, "xpool legacy output requirement is invalid");
}
std::string_view spelling(xpool::ffn::DpRowLayout value) {
  switch (value) {
  case xpool::ffn::DpRowLayout::None:
    return "none";
  case xpool::ffn::DpRowLayout::UniformByRank:
    return "uniform_by_rank";
  case xpool::ffn::DpRowLayout::PackedByRank:
    return "packed_by_rank";
  }
  invalid(0, "xpool legacy DP layout is invalid");
}
std::string_view spelling(xpool::ffn::ResultCode value) {
  switch (value) {
  case xpool::ffn::ResultCode::Ok:
    return "ok";
  case xpool::ffn::ResultCode::Shutdown:
    return "shutdown";
  case xpool::ffn::ResultCode::ProtocolMismatch:
    return "protocol_mismatch";
  case xpool::ffn::ResultCode::Timeout:
    return "timeout";
  }
  invalid(0, "xpool legacy result is invalid");
}
std::string_view spelling(xpool::fabric::DeliveryVariant value) {
  switch (value) {
  case xpool::fabric::DeliveryVariant::DirectPartial:
    return "direct_partial";
  case xpool::fabric::DeliveryVariant::SingleComplete:
    return "single_complete";
  case xpool::fabric::DeliveryVariant::ReplicatedComplete:
    return "replicated_complete";
  }
  invalid(0, "xpool legacy delivery is invalid");
}
void member(JsonOutput &out, std::string_view key, std::uint64_t value) {
  out.key(key);
  out.number(value);
}
void member(JsonOutput &out, std::string_view key, std::string_view value) {
  out.key(key);
  out.quoted(value);
}
void nullable_member(JsonOutput &out, std::string_view key, std::optional<std::uint64_t> value) {
  out.key(key);
  out.nullable(value);
}
std::optional<std::uint64_t> duration(std::uint64_t start, std::uint64_t end) {
  return start != 0 && end >= start ? std::optional{end - start} : std::nullopt;
}
template <typename Timestamp, std::size_t Count>
void phases(JsonOutput &out, Timestamp timestamp, const std::array<Phase, Count> &table) {
  out.text("{");
  auto comma = false;
  for (const auto &phase : table) {
    if (const auto value = duration(timestamp(phase.start), timestamp(phase.end))) {
      if (comma)
        out.text(",");
      comma = true;
      member(out, phase.name, *value);
    }
  }
  out.text("}");
}
// Match Python round((count - 1) * percentile), including ties-to-even. The
// percentile multiplication deliberately uses the same binary double values.
std::size_t percentile_index(std::size_t count, double percentile) {
  const auto value = static_cast<double>(count - 1) * percentile;
  const auto lower = static_cast<std::size_t>(std::floor(value));
  const auto fraction = value - static_cast<double>(lower);
  return lower + (fraction > 0.5 || (fraction == 0.5 && lower % 2 != 0) ? 1 : 0);
}
void transport_record(JsonOutput &out, const TransportRecord &record) {
  out.text("{");
  member(out, "trace_id", record.trace_id());
  out.text(",");
  member(out, "payload_rows", record.payload_rows());
  out.text(",");
  member(out, "layer_ordinal", record.layer_ordinal());
  out.text(",");
  member(out, "forward_mode", spelling(record.forward_mode()));
  out.text(",");
  member(out, "output_requirement", spelling(record.output_requirement()));
  out.text(",");
  member(out, "dp_row_layout", spelling(record.dp_row_layout()));
  out.text(",");
  member(out, "result_code", spelling(record.result_code()));
  const auto timestamp = [&record](std::size_t index) { return record.timestamp(static_cast<TransportKind>(index)); };
  for (std::size_t index = 0; index != 11; ++index) {
    out.text(",");
    member(out, evidence_field_name(SnapshotObserver::Transport, SnapshotRole::Instance, 16 + index), timestamp(index));
  }
  out.text(",\"durations_ns\":");
  phases(out, timestamp, transport_phases);
  out.text("}");
}
SnapshotRole fabric_role(const FabricRecord &record) {
  using Kind = xpool::devkit::fabric_observer::RecordKind;
  switch (record.kind()) {
  case Kind::AtnAgent:
    return SnapshotRole::AtnAgent;
  case Kind::Coordinator:
    return SnapshotRole::Coordinator;
  case Kind::FfnAgent:
    return SnapshotRole::FfnAgent;
  }
  invalid(0, "xpool legacy Fabric role is invalid");
}
std::uint64_t timestamp(const FabricRecord &record, SnapshotRole role, std::size_t index) {
  if (role == SnapshotRole::AtnAgent)
    return record.timestamp(static_cast<AtnKind>(index));
  if (role == SnapshotRole::Coordinator)
    return record.timestamp(static_cast<CoordinatorKind>(index));
  return record.timestamp(static_cast<FfnKind>(index));
}
void fabric_record(JsonOutput &out, const FabricRecord &record) {
  const auto role = fabric_role(record);
  out.text("{");
  member(out, "local_trace_id", record.local_trace_id());
  out.text(",");
  member(out, "instance_index", record.key().instance_index);
  out.text(",");
  member(out, "invocation_sequence", record.key().invocation_sequence);
  out.text(",");
  member(out, "layer_ordinal", record.layer_ordinal());
  out.text(",");
  member(out, "payload_rows", record.payload_rows());
  out.text(",");
  member(out, "output_requirement", spelling(record.output_requirement()));
  out.text(",");
  member(out, "kind", evidence_role_name(role));
  out.text(",\"dp_row_layout\":");
  if (role == SnapshotRole::AtnAgent)
    out.quoted(spelling(*record.dp_row_layout()));
  else
    out.text("null");
  out.text(",\"facts\":{");
  if (role == SnapshotRole::AtnAgent) {
    member(out, "dp_rank_payload_rows", *record.dp_rank_payload_rows());
    out.text(",");
    member(out, "forward_mode", spelling(*record.forward_mode()));
    out.text(",");
  }
  nullable_member(out, "executor_lane_index", record.executor_lane_index());
  out.text(",");
  nullable_member(out, "executor_lease_sequence", record.executor_lease_sequence());
  if (role == SnapshotRole::Coordinator) {
    out.text(",\"scheduler\":{");
    member(out, "policy", record.ready_ticket() ? "fifo" : "random");
    if (record.ready_ticket()) {
      out.text(",");
      member(out, "ready_ticket", *record.ready_ticket());
    }
    out.text("}");
  } else if (role == SnapshotRole::FfnAgent) {
    out.text(",");
    member(out, "payload_row_capacity", *record.payload_row_capacity());
    out.text(",");
    member(out, "delivery", spelling(*record.delivery()));
  }
  out.text("},\"events_ns\":{");
  const auto count = role == SnapshotRole::AtnAgent ? 6U : role == SnapshotRole::Coordinator ? 8U : 9U;
  for (std::size_t index = 0; index != count; ++index) {
    if (index != 0)
      out.text(",");
    member(out, evidence_field_name(SnapshotObserver::Fabric, role, 16 + index).substr(10),
           timestamp(record, role, index));
  }
  out.text("},\"durations_ns\":");
  const auto read_time = [&record, role](std::size_t index) { return timestamp(record, role, index); };
  if (role == SnapshotRole::AtnAgent)
    phases(out, read_time, atn_phases);
  else if (role == SnapshotRole::Coordinator)
    phases(out, read_time, coordinator_phases);
  else
    phases(out, read_time, ffn_phases);
  out.text("}");
}

} // namespace

std::shared_ptr<LegacyJsonFormatter>
LegacyJsonFormatter::create(std::shared_ptr<MemoryBudget> host, detail::LegacyScope scope, std::size_t capacity,
                            std::variant<TransportJsonMetadata, FabricJsonMetadata> metadata) {
  return make_budgeted_shared<LegacyJsonFormatter>(host, MemoryKind::Writer, scope.source.producer, ConstructionKey{},
                                                   host, scope, capacity, metadata);
}
LegacyJsonFormatter::LegacyJsonFormatter(ConstructionKey, std::shared_ptr<MemoryBudget> host, detail::LegacyScope scope,
                                         std::size_t capacity,
                                         std::variant<TransportJsonMetadata, FabricJsonMetadata> metadata)
    : host_(std::move(host)), scope_(scope), capacity_(capacity),
      resource_(*host_, MemoryKind::Writer, scope.source.producer), identity_(&resource_), suffix_(&resource_),
      topologies_(&resource_), durations_(&resource_) {
  const auto fabric = scope.family == detail::LegacyFamily::Fabric;
  if (capacity == 0 || scope.source.producer == 0 || scope.generation == 0 ||
      fabric != std::holds_alternative<FabricJsonMetadata>(metadata))
    invalid(scope.source.producer, "xpool legacy formatter registration is invalid");
  if (fabric) {
    const auto &facts = std::get<FabricJsonMetadata>(metadata);
    if (facts.generation.size() != 32 ||
        facts.generation.find_first_not_of("0123456789abcdef") != std::string_view::npos || facts.atnagent_count == 0 ||
        facts.ffnagent_count == 0 || facts.model_topologies.empty() ||
        (facts.pe >= facts.atnagent_count && facts.pe - facts.atnagent_count >= facts.ffnagent_count))
      invalid(scope.source.producer, "xpool legacy Fabric envelope metadata is invalid");
    for (const auto &topology : facts.model_topologies)
      if (topology.atn_tp_size == 0 || topology.atn_dp_size == 0)
        invalid(scope.source.producer, "xpool legacy Fabric model topology is invalid");
    identity_.assign(facts.generation);
    pid_or_pe_ = facts.pe;
    rank_or_atnagent_count_ = facts.atnagent_count;
    ffnagent_count_ = facts.ffnagent_count;
    topologies_.assign(facts.model_topologies.begin(), facts.model_topologies.end());
  } else {
    const auto &facts = std::get<TransportJsonMetadata>(metadata);
    if (facts.pid == 0 || facts.model_id.empty() || facts.model_id.find('\0') != std::string_view::npos ||
        facts.arena_handle_suffix.size() != 16 ||
        facts.arena_handle_suffix.find_first_not_of("0123456789abcdef") != std::string_view::npos ||
        scope.endpoint_owner == 0 || scope.endpoint_sequence == 0)
      invalid(scope.source.producer, "xpool legacy Transport envelope metadata is invalid");
    identity_.assign(facts.model_id);
    suffix_.assign(facts.arena_handle_suffix);
    pid_or_pe_ = facts.pid;
    rank_or_atnagent_count_ = facts.rank;
    durations_.resize(capacity);
  }
}

std::size_t LegacyJsonFormatter::serialize(const SnapshotRead &read, JsonSink sink) {
  const auto &report = *read.evidence();
  const auto fabric = scope_.family == detail::LegacyFamily::Fabric;
  const auto site =
      scope_.family == detail::LegacyFamily::TransportInstance ? SnapshotRole::Instance : SnapshotRole::AtnAgent;
  if (!read.belongs_to(scope_) || report.capacity > capacity_ || !report.sequence || !report.dropped ||
      (fabric && report.metadata().ffnagent_count && report.metadata().ffnagent_count != ffnagent_count_))
    invalid(scope_.source.producer, "xpool legacy JSON read does not match its registered lifecycle or counters");
  auto out = JsonOutput{sink};
  out.text("{");
  if (fabric) {
    member(out, "generation", identity_);
    out.text(",");
    member(out, "pe", pid_or_pe_);
    out.text(",");
    member(out, "atnagent_count", rank_or_atnagent_count_);
    out.text(",");
    member(out, "ffnagent_count", ffnagent_count_);
    out.text(",\"model_topologies\":[");
    for (std::size_t index = 0; index != topologies_.size(); ++index) {
      if (index != 0)
        out.text(",");
      out.text("{");
      member(out, "atn_tp_size", topologies_[index].atn_tp_size);
      out.text(",");
      member(out, "atn_dp_size", topologies_[index].atn_dp_size);
      out.text("}");
    }
    out.text("]");
  } else {
    member(out, "pid", pid_or_pe_);
    out.text(",");
    member(out, "site", evidence_role_name(site));
    out.text(",");
    member(out, "model_id", identity_);
    out.text(",");
    member(out, "rank", rank_or_atnagent_count_);
    out.text(",");
    member(out, "arena_handle_suffix", suffix_);
  }
  out.text(",");
  member(out, "sequence", *report.sequence);
  out.text(",");
  member(out, "dropped", *report.dropped);
  if (!fabric) {
    const auto records = read.transport_records();
    out.text(",\"phase_summary\":{");
    auto comma = false;
    for (const auto &phase : transport_phases) {
      auto count = std::size_t{0};
      for (const auto &record : records)
        if (const auto value = duration(record.timestamp(static_cast<TransportKind>(phase.start)),
                                        record.timestamp(static_cast<TransportKind>(phase.end))))
          durations_[count++] = *value;
      if (count == 0)
        continue;
      std::sort(durations_.begin(), durations_.begin() + count);
      if (comma)
        out.text(",");
      comma = true;
      out.key(phase.name);
      out.text("{");
      member(out, "count", count);
      out.text(",");
      member(out, "min_ns", durations_[0]);
      out.text(",");
      member(out, "median_ns", durations_[(count - 1) / 2]);
      out.text(",");
      member(out, "p95_ns", durations_[percentile_index(count, 0.95)]);
      out.text(",");
      member(out, "p99_ns", durations_[percentile_index(count, 0.99)]);
      out.text(",");
      member(out, "max_ns", durations_[count - 1]);
      out.text("}");
    }
    std::size_t completed = 0, closed = 0;
    for (const auto &record : records) {
      const auto acknowledged = record.recorded(static_cast<TransportKind>(9));
      const auto is_closed = record.recorded(static_cast<TransportKind>(10));
      if (acknowledged && is_closed)
        invalid(scope_.source.producer, "xpool legacy Transport record has conflicting terminals");
      completed += record.recorded(static_cast<TransportKind>(site == SnapshotRole::AtnAgent ? 6 : 9));
      closed += is_closed;
    }
    if (completed > records.size() || closed > records.size() - completed)
      invalid(scope_.source.producer, "xpool legacy Transport terminal counts conflict");
    out.text("},\"record_counts\":{");
    member(out, "retained", records.size());
    out.text(",");
    member(out, "completed", completed);
    out.text(",");
    member(out, "closed", closed);
    out.text(",");
    member(out, "incomplete", records.size() - completed - closed);
    out.text("}");
  }
  out.text(",\"records\":[");
  if (fabric) {
    auto comma = false;
    for (const auto &record : read.fabric_records()) {
      if (comma)
        out.text(",");
      comma = true;
      fabric_record(out, record);
    }
  } else {
    auto comma = false;
    for (const auto &record : read.transport_records()) {
      if (comma)
        out.text(",");
      comma = true;
      transport_record(out, record);
    }
  }
  out.text("]}\n");
  return out.finish();
}
SnapshotDocument LegacyJsonFormatter::document() noexcept {
  return {this, [](const SnapshotRead &read, void *context, JsonSink sink) {
            static_cast<LegacyJsonFormatter *>(context)->serialize(read, sink);
          }};
}

} // namespace xpool::devkit::timeline
