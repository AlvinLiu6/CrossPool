#include <array>

#include <xpool/devkit/timeline/evidence.hpp>

namespace xpool::devkit::timeline {

namespace {

constexpr std::array<std::string_view, 11> transport_times = {"request_staging_started",
                                                              "request_staging_completed",
                                                              "request_published",
                                                              "request_observed",
                                                              "execution_started",
                                                              "execution_completed",
                                                              "result_published",
                                                              "result_observed",
                                                              "output_copied",
                                                              "result_acknowledged",
                                                              "closed"};
constexpr std::array<std::string_view, 11> transport_boundaries = {"boundary.RequestStagingStarted",
                                                                   "boundary.RequestStagingCompleted",
                                                                   "boundary.RequestPublished",
                                                                   "boundary.RequestObserved",
                                                                   "boundary.ExecutionStarted",
                                                                   "boundary.ExecutionCompleted",
                                                                   "boundary.ResultPublished",
                                                                   "boundary.ResultObserved",
                                                                   "boundary.OutputCopied",
                                                                   "boundary.ResultAcknowledged",
                                                                   "boundary.Closed"};
constexpr std::array<std::string_view, 6> atn_times = {
    "events_ns.submission_prepared",    "events_ns.submission_published",
    "events_ns.admission_observed",     "events_ns.input_ready_published",
    "events_ns.output_commit_observed", "events_ns.output_acknowledgement_published"};
constexpr std::array<std::string_view, 6> atn_boundaries = {
    "boundary.SubmissionPrepared",  "boundary.SubmissionPublished",  "boundary.AdmissionObserved",
    "boundary.InputReadyPublished", "boundary.OutputCommitObserved", "boundary.OutputAcknowledgementPublished"};
constexpr std::array<std::string_view, 8> coordinator_times = {"events_ns.enqueued",
                                                               "events_ns.scheduled",
                                                               "events_ns.admission_published",
                                                               "events_ns.lane_execution_published",
                                                               "events_ns.ffnagent_completions_observed",
                                                               "events_ns.output_commit_published",
                                                               "events_ns.output_acknowledgements_observed",
                                                               "events_ns.lane_released"};
constexpr std::array<std::string_view, 8> coordinator_boundaries = {"boundary.Enqueued",
                                                                    "boundary.Scheduled",
                                                                    "boundary.AdmissionPublished",
                                                                    "boundary.LaneExecutionPublished",
                                                                    "boundary.FfnAgentCompletionsObserved",
                                                                    "boundary.OutputCommitPublished",
                                                                    "boundary.OutputAcknowledgementsObserved",
                                                                    "boundary.LaneReleased"};
constexpr std::array<std::string_view, 9> ffn_times = {"events_ns.lane_execution_observed",
                                                       "events_ns.input_ready_observed",
                                                       "events_ns.routing_metadata_published",
                                                       "events_ns.routing_metadata_observed",
                                                       "events_ns.compute_started",
                                                       "events_ns.compute_completed",
                                                       "events_ns.partial_ready_published",
                                                       "events_ns.peer_partials_ready_observed",
                                                       "events_ns.completion_published"};
constexpr std::array<std::string_view, 9> ffn_boundaries = {
    "boundary.LaneExecutionObserved",    "boundary.InputReadyObserved",
    "boundary.RoutingMetadataPublished", "boundary.RoutingMetadataObserved",
    "boundary.ComputeStarted",           "boundary.ComputeCompleted",
    "boundary.PartialReadyPublished",    "boundary.PeerPartialsReadyObserved",
    "boundary.CompletionPublished"};

std::string_view phase_name(SnapshotObserver observer, SnapshotRole role, std::size_t phase, bool boundary) {
  const auto select = [phase, boundary](const auto &times, const auto &boundaries) {
    return phase < times.size() ? (boundary ? boundaries[phase] : times[phase]) : std::string_view{};
  };
  if (observer == SnapshotObserver::Transport)
    return select(transport_times, transport_boundaries);
  if (role == SnapshotRole::AtnAgent)
    return select(atn_times, atn_boundaries);
  if (role == SnapshotRole::Coordinator)
    return select(coordinator_times, coordinator_boundaries);
  return select(ffn_times, ffn_boundaries);
}

} // namespace

std::string_view evidence_role_name(SnapshotRole role) noexcept {
  switch (role) {
  case SnapshotRole::Instance:
    return "instance";
  case SnapshotRole::AtnAgent:
    return "atnagent";
  case SnapshotRole::Coordinator:
    return "coordinator";
  case SnapshotRole::FfnAgent:
    return "ffnagent";
  }
  return {};
}

std::string_view evidence_reason_name(EvidenceReason reason) noexcept {
  switch (reason) {
  case EvidenceReason::ApplicabilityUnknown:
    return "applicability_unknown";
  case EvidenceReason::EventMissing:
    return "event_missing";
  case EvidenceReason::InvalidRequest:
    return "invalid_request";
  case EvidenceReason::LegacyFactMissing:
    return "legacy_fact_missing";
  case EvidenceReason::LegacyZeroTimestamp:
    return "legacy_zero_timestamp";
  case EvidenceReason::OperationUnobserved:
    return "operation_unobserved";
  case EvidenceReason::TemporalConflict:
    return "temporal_conflict";
  case EvidenceReason::UnknownResult:
    return "unknown_result";
  }
  return {};
}

std::string_view evidence_field_name(SnapshotObserver observer, SnapshotRole role, std::size_t field) noexcept {
  if (field >= 32)
    return phase_name(observer, role, field - 32, true);
  if (field >= 16)
    return phase_name(observer, role, field - 16, false);
  switch (field) {
  case 0:
    return observer == SnapshotObserver::Transport ? "trace_id" : "local_trace_id";
  case 1:
    return "layer_ordinal";
  case 2:
    return "payload_rows";
  case 3:
    return "output_requirement";
  case 4:
    return observer == SnapshotObserver::Transport ? "forward_mode"
           : role == SnapshotRole::AtnAgent        ? "facts.forward_mode"
                                                   : "";
  case 5:
    return "dp_row_layout";
  case 6:
    return observer == SnapshotObserver::Transport ? "result_code" : "";
  case 7:
    return observer == SnapshotObserver::Fabric ? "instance_index" : "";
  case 8:
    return observer == SnapshotObserver::Fabric ? "invocation_sequence" : "";
  case 9:
    return observer == SnapshotObserver::Fabric ? "facts.executor_lane_index" : "";
  case 10:
    return observer == SnapshotObserver::Fabric ? "facts.executor_lease_sequence" : "";
  case 11:
    return observer == SnapshotObserver::Fabric && role == SnapshotRole::AtnAgent ? "facts.dp_rank_payload_rows" : "";
  case 12:
    return observer == SnapshotObserver::Fabric && role == SnapshotRole::FfnAgent ? "facts.payload_row_capacity" : "";
  case 13:
    return observer == SnapshotObserver::Fabric && role == SnapshotRole::FfnAgent ? "facts.delivery" : "";
  case 14:
    return observer == SnapshotObserver::Fabric && role == SnapshotRole::Coordinator ? "facts.scheduler.policy" : "";
  case 15:
    return observer == SnapshotObserver::Fabric && role == SnapshotRole::Coordinator ? "facts.scheduler.ready_ticket"
                                                                                     : "";
  default:
    return {};
  }
}

std::string_view evidence_boundary_name(SnapshotObserver observer, SnapshotRole role, std::size_t phase) noexcept {
  const auto field = phase_name(observer, role, phase, true);
  return field.empty() ? field : field.substr(9);
}

} // namespace xpool::devkit::timeline
