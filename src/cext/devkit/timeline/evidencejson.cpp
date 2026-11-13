#include <algorithm>
#include <array>
#include <string_view>
#include <utility>

#include <xpool/devkit/timeline/evidence.hpp>

namespace xpool::devkit::timeline {

namespace {

constexpr auto bit(std::size_t index) { return std::uint64_t{1} << index; }

template <std::size_t N>
void hex(detail::JsonOutput &output, const std::array<std::byte, N> &bytes, bool uuid = false) {
  constexpr auto digits = std::string_view{"0123456789abcdef"};
  output.text("\"");
  for (auto index = std::size_t{0}; index < N; ++index) {
    if (uuid && (index == 4 || index == 6 || index == 8 || index == 10))
      output.text("-");
    const auto value = std::to_integer<unsigned char>(bytes[index]);
    const char pair[] = {digits[value >> 4], digits[value & 15]};
    output.text({pair, 2});
  }
  output.text("\"");
}

struct FieldOrder {
  std::array<std::size_t, 43> fields{};
  std::size_t size = 0;
};

FieldOrder field_order(const SnapshotEvidenceReport &report, const SnapshotEvidenceEntry &entry, std::uint64_t mask,
                       bool boundary) {
  auto result = FieldOrder{};
  const auto name = [&](std::size_t index) {
    const auto role = entry.role.value_or(SnapshotRole::Instance);
    return boundary ? evidence_boundary_name(report.observer, role, index)
                    : evidence_field_name(report.observer, role, index);
  };
  const auto count = boundary ? std::size_t{11} : std::size_t{43};
  for (auto field = std::size_t{0}; field < count; ++field) {
    if ((mask & bit(field)) == 0)
      continue;
    if (name(field).empty())
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, report.source.producer,
                            "xpool snapshot contains an invalid evidence field");
    result.fields[result.size++] = field;
  }
  std::sort(result.fields.begin(), result.fields.begin() + result.size,
            [&](auto left, auto right) { return name(left) < name(right); });
  return result;
}

void fields(detail::JsonOutput &out, const SnapshotEvidenceReport &report, const SnapshotEvidenceEntry &entry,
            std::uint64_t mask, bool boundary = false) {
  const auto order = field_order(report, entry, mask, boundary);
  out.text("[");
  for (auto index = std::size_t{0}; index < order.size; ++index) {
    if (index != 0)
      out.text(",");
    const auto field = order.fields[index];
    const auto role = entry.role.value_or(SnapshotRole::Instance);
    out.quoted(boundary ? evidence_boundary_name(report.observer, role, field)
                        : evidence_field_name(report.observer, role, field));
  }
  out.text("]");
}

void serialize_entry(detail::JsonOutput &out, const SnapshotEvidenceReport &report,
                     const SnapshotEvidenceEntry &entry) {
  out.text("{\"local_trace_id\":");
  out.number(entry.local_trace_id);
  out.text(",\"role\":");
  if (entry.role)
    out.quoted(evidence_role_name(*entry.role));
  else
    out.text("null");
  out.text(",\"operation\":");
  if (entry.operation) {
    out.text("{\"endpoint_owner\":");
    out.number(entry.operation->endpoint_owner);
    out.text(",\"endpoint_sequence\":");
    out.number(entry.operation->endpoint_sequence);
    out.text(",\"operation_sequence\":");
    out.number(entry.operation->operation_sequence);
    out.text("}");
  } else
    out.text("null");
  out.text(",\"invocation\":");
  if (entry.invocation) {
    out.text("{\"generation\":");
    out.number(entry.invocation->generation);
    out.text(",\"instance_index\":");
    out.number(entry.invocation->instance_index);
    out.text(",\"invocation_sequence\":");
    out.number(entry.invocation->invocation_sequence);
    out.text("}");
  } else
    out.text("null");
  out.text(",\"projection_status\":");
  out.quoted(entry.projection_status == ProjectionStatus::Retained  ? "retained"
             : entry.projection_status == ProjectionStatus::Omitted ? "omitted"
                                                                    : "unobserved");
  out.text(",\"evidence_status\":");
  out.quoted(entry.evidence_status == EvidenceStatus::Complete  ? "complete"
             : entry.evidence_status == EvidenceStatus::Partial ? "partial"
                                                                : "unavailable");
  out.text(",\"unknown_fields\":");
  fields(out, report, entry, entry.unknown_fields);
  out.text(",\"not_applicable_fields\":");
  fields(out, report, entry, entry.not_applicable_fields);
  out.text(",\"missing_boundaries\":");
  fields(out, report, entry, entry.missing_boundaries, true);
  out.text(",\"unknown_applicability\":");
  fields(out, report, entry, entry.unknown_applicability, true);
  out.text(",\"invalid_fields\":[");
  if (entry.anomaly) {
    const auto &raw = *entry.anomaly;
    const auto mask = transport_invalid_fields(raw.forward_mode, raw.output_requirement, raw.dp_row_layout);
    auto separator = false;
    for (const auto &[bit, name] : std::array<std::pair<std::uint32_t, std::string_view>, 3>{
             {{4, "dp_row_layout"}, {1, "forward_mode"}, {2, "output_requirement"}}}) {
      if ((mask & bit) == 0)
        continue;
      if (separator)
        out.text(",");
      out.quoted(name);
      separator = true;
    }
  }
  out.text("],\"raw_request\":");
  if (entry.anomaly) {
    const auto &raw = *entry.anomaly;
    out.text("{\"layer_ordinal\":");
    out.number(raw.layer_ordinal);
    out.text(",\"payload_rows\":");
    out.number(raw.payload_rows);
    out.text(",\"forward_mode\":");
    out.number(raw.forward_mode);
    out.text(",\"output_requirement\":");
    out.number(raw.output_requirement);
    out.text(",\"dp_row_layout\":");
    out.number(raw.dp_row_layout);
    out.text("}");
  } else
    out.text("null");
  out.text(",\"terminal_boundaries\":[");
  if (entry.anomaly) {
    constexpr auto phases = std::array<std::size_t, 4>{6, 7, 9, 10};
    const auto &raw = *entry.anomaly;
    auto separator = false;
    for (std::size_t index = 0; index < phases.size(); ++index) {
      if ((raw.terminal_mask & (std::uint32_t{1} << index)) == 0)
        continue;
      if (separator)
        out.text(",");
      out.text("{\"boundary\":");
      out.quoted(evidence_boundary_name(SnapshotObserver::Transport, *entry.role, phases[index]));
      out.text(",\"raw_time_ns\":");
      out.number(raw.terminal_times[index]);
      out.text(",\"result_code\":");
      out.number(raw.result_code);
      out.text("}");
      separator = true;
    }
  }
  out.text("]");
  out.text(",\"quality\":[");
  constexpr std::array reasons = {EvidenceReason::ApplicabilityUnknown, EvidenceReason::EventMissing,
                                  EvidenceReason::InvalidRequest,       EvidenceReason::LegacyFactMissing,
                                  EvidenceReason::LegacyZeroTimestamp,  EvidenceReason::OperationUnobserved,
                                  EvidenceReason::TemporalConflict,     EvidenceReason::UnknownResult};
  auto separator = false;
  for (const auto reason : reasons) {
    if ((entry.quality & static_cast<std::uint32_t>(reason)) == 0)
      continue;
    if (separator)
      out.text(",");
    out.quoted(evidence_reason_name(reason));
    separator = true;
  }
  out.text("],\"sources\":[");
  const auto order = field_order(report, entry, entry.source_fields, false);
  for (auto index = std::size_t{0}; index < order.size; ++index) {
    if (index != 0)
      out.text(",");
    const auto field = order.fields[index];
    out.text("{\"field\":");
    out.quoted(evidence_field_name(report.observer, entry.role.value_or(SnapshotRole::Instance), field));
    out.text(",\"event\":");
    if ((entry.metadata_fields & bit(field)) == 0) {
      out.text("{\"producer\":");
      out.number(report.source.producer);
      out.text(",\"event_sequence\":");
      out.number(entry.sources[field]);
      out.text("},\"metadata\":null}");
    } else {
      const auto selector = entry.sources[field];
      const auto &metadata = report.metadata();
      const auto &source = selector == 1 ? metadata.scheduler : selector == 2 ? metadata.routing : metadata.topology;
      out.text("null,\"metadata\":{\"producer\":");
      out.number(source.producer);
      out.text(",\"revision\":");
      out.number(source.revision);
      out.text(",\"path\":");
      out.quoted(source.path());
      out.text("}}");
    }
  }
  out.text("]}");
}

} // namespace

std::size_t SnapshotEvidenceReport::serialize(JsonSink sink,
                                              std::optional<std::array<std::byte, 32>> snapshot_digest) const {
  auto out = detail::JsonOutput{sink};
  out.text("{\"contract_version\":");
  out.number(contract_version);
  out.text(",\"run_id\":");
  hex(out, source.run_id, true);
  out.text(",\"producer_id\":");
  out.number(source.producer);
  out.text(",\"observer\":");
  out.quoted(observer == SnapshotObserver::Transport ? "transport" : "fabric");
  out.text(",\"generation\":");
  out.number(generation);
  out.text(",\"endpoint\":");
  if (endpoint) {
    out.text("{\"owner\":");
    out.number((*endpoint)[0]);
    out.text(",\"sequence\":");
    out.number((*endpoint)[1]);
    out.text("}");
  } else
    out.text("null");
  out.text(",\"site\":");
  if (site)
    out.quoted(evidence_role_name(*site));
  else
    out.text("null");
  out.text(",\"snapshot_sha256\":");
  if (snapshot_digest)
    hex(out, *snapshot_digest);
  else
    out.text("null");
  out.text(",\"capacity\":");
  out.number(capacity);
  out.text(",\"sequence\":");
  out.nullable(sequence);
  out.text(",\"dropped\":");
  out.nullable(dropped);
  out.text(",\"unobserved_operations\":");
  out.nullable(unobserved_operations);
  out.text(",\"entries\":[");
  auto first = true;
  for (const auto &entry : entries()) {
    if (!first)
      out.text(",");
    serialize_entry(out, *this, entry);
    first = false;
  }
  out.text("]}\n");
  return out.finish();
}

} // namespace xpool::devkit::timeline
