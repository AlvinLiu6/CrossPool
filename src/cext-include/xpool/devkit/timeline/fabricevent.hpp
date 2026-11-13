#pragma once

/// \file xpool/devkit/timeline/fabricevent.hpp
/// \brief Pure assembly of ordinary AtnAgent Fabric events from current facts.

#include <xpool/devkit/timeline/fabricrecording.hpp>
#include <xpool/fabric/hooks.hpp>

namespace xpool::devkit::timeline {

/// Assemble a normal 200--205 event without reserving an Event or operation.
/// The owner updates Admission/Commit facts before calling this at that Hook.
/// Returns false for incomplete or invalid facts and clears output. A true
/// result still needs commit_device to supply its positive event_sequence.
XPOOL_HOST_DEVICE_FN inline bool make_fabric_atn_event(const FabricAtnCurrent &current, BatchSource source,
                                                       std::uint64_t generation, std::int32_t pe,
                                                       xpool::hooks::FabricAtnAgentProtocolEvent::Kind kind,
                                                       std::uint64_t time_ns, Event &output) {
  output = {};
  using Kind = xpool::hooks::FabricAtnAgentProtocolEvent::Kind;
  auto wire_kind = std::uint32_t{0};
  switch (kind) {
  case Kind::SubmissionPrepared:
    wire_kind = 200;
    break;
  case Kind::SubmissionPublished:
    wire_kind = 201;
    break;
  case Kind::AdmissionObserved:
    wire_kind = 202;
    break;
  case Kind::InputReadyPublished:
    wire_kind = 203;
    break;
  case Kind::OutputCommitObserved:
    wire_kind = 204;
    break;
  case Kind::OutputAcknowledgementPublished:
    wire_kind = 205;
    break;
  default:
    return false;
  }
  const auto &submission = current.submission;
  if (source.producer == 0 || source.clock_domain_id == 0 || generation == 0 || pe < 0 || current.local_trace_id == 0 ||
      submission.validate() != xpool::ffn::ResultCode::Ok ||
      (wire_kind >= 202 && (!current.admission_known || current.executor_lease_sequence == 0)) ||
      (wire_kind >= 204 && (!current.commit_known || !xpool::ffn::is_valid(current.result))))
    return false;
  // C0: Generation, Invocation, local trace, request/FFN facts, DP rows, PE.
  output = Event{.raw_time_ns = time_ns,
                 .clock_domain_id = source.clock_domain_id,
                 .kind = wire_kind,
                 .presence = 0x20f21,
                 .generation = generation,
                 .instance_index = submission.key.instance_index,
                 .invocation_sequence = submission.key.invocation_sequence,
                 .local_trace_id = current.local_trace_id,
                 .layer_ordinal = submission.layer_ordinal,
                 .payload_rows = submission.payload_rows,
                 .dp_rank_payload_rows = submission.dp_rank_payload_rows,
                 .forward_mode = static_cast<std::uint32_t>(submission.forward_mode),
                 .output_requirement = static_cast<std::uint32_t>(submission.output_requirement),
                 .dp_row_layout = static_cast<std::uint32_t>(submission.dp_row_layout),
                 .role = 2,
                 .pe = pe};
  if (wire_kind >= 202) {
    output.presence |= std::uint32_t{1} << 6;
    output.lane_index = current.executor_lane_index;
    output.lease_sequence = current.executor_lease_sequence;
  }
  if (wire_kind >= 204) {
    output.presence |= std::uint32_t{1} << 15;
    output.result_code = static_cast<std::uint32_t>(current.result);
  }
  return true;
}

/// Assemble a Coordinator boundary with the actual scheduler and selected lease.
XPOOL_HOST_DEVICE_FN inline bool make_fabric_coordinator_event(const FabricCoordinatorCurrent &current,
                                                               BatchSource source, std::uint64_t generation,
                                                               std::int32_t pe, bool fifo,
                                                               xpool::hooks::FabricCoordinatorProtocolEvent::Kind kind,
                                                               std::uint64_t time_ns, Event &output) {
  output = {};
  using Kind = xpool::hooks::FabricCoordinatorProtocolEvent::Kind;
  std::uint32_t wire = 0;
  switch (kind) {
  case Kind::Enqueued:
    wire = 300;
    break;
  case Kind::Scheduled:
    wire = 301;
    break;
  case Kind::AdmissionPublished:
    wire = 302;
    break;
  case Kind::LaneExecutionPublished:
    wire = 303;
    break;
  case Kind::FfnAgentCompletionsObserved:
    wire = 304;
    break;
  case Kind::OutputCommitPublished:
    wire = 305;
    break;
  case Kind::OutputAcknowledgementsObserved:
    wire = 306;
    break;
  case Kind::LaneReleased:
    wire = 307;
    break;
  default:
    return false;
  }
  const auto &invocation = current.invocation;
  if (source.producer == 0 || source.clock_domain_id == 0 || generation == 0 || pe < 0 || current.local_trace_id == 0 ||
      invocation.validate() != xpool::ffn::ResultCode::Ok || (fifo && current.ready_ticket == 0) ||
      (wire >= 301 && (!current.scheduled || current.lease_sequence == 0)))
    return false;
  output = {.raw_time_ns = time_ns,
            .clock_domain_id = source.clock_domain_id,
            .kind = wire,
            .presence = 0x20521,
            .generation = generation,
            .instance_index = invocation.key.instance_index,
            .invocation_sequence = invocation.key.invocation_sequence,
            .local_trace_id = current.local_trace_id,
            .layer_ordinal = invocation.layer_ordinal,
            .payload_rows = invocation.payload_rows,
            .output_requirement = static_cast<std::uint32_t>(invocation.output_requirement),
            .role = 3,
            .pe = pe};
  if (wire >= 301) {
    output.presence |= 1U << 6;
    output.lane_index = current.lane_index;
    output.lease_sequence = current.lease_sequence;
  }
  if (fifo) {
    output.presence |= 1U << 13;
    output.ready_ticket = current.ready_ticket;
  }
  return true;
}

/// Assemble an FfnAgent boundary; installed capacity and actual delivery repeat
/// even when the opening event was lost. No timestamp is reconstructed.
XPOOL_HOST_DEVICE_FN inline bool make_fabric_ffn_event(const FabricFfnCurrent &current, BatchSource source,
                                                       std::uint64_t generation, std::int32_t pe, std::size_t lane,
                                                       xpool::hooks::FabricFfnAgentProtocolEvent::Kind kind,
                                                       std::uint64_t time_ns, Event &output) {
  output = {};
  using Kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind;
  std::uint32_t wire = 0;
  switch (kind) {
  case Kind::LaneExecutionObserved:
    wire = 400;
    break;
  case Kind::InputReadyObserved:
    wire = 401;
    break;
  case Kind::RoutingMetadataPublished:
    wire = 402;
    break;
  case Kind::RoutingMetadataObserved:
    wire = 403;
    break;
  case Kind::ComputeStarted:
    wire = 404;
    break;
  case Kind::ComputeCompleted:
    wire = 405;
    break;
  case Kind::PartialReadyPublished:
    wire = 406;
    break;
  case Kind::PeerPartialsReadyObserved:
    wire = 407;
    break;
  case Kind::CompletionPublished:
    wire = 408;
    break;
  default:
    return false;
  }
  const auto &execution = current.execution;
  if (source.producer == 0 || source.clock_domain_id == 0 || generation == 0 || pe < 0 || current.local_trace_id == 0 ||
      execution.validate() != xpool::ffn::ResultCode::Ok || current.payload_row_capacity == 0 ||
      execution.payload_rows > current.payload_row_capacity || !xpool::fabric::is_valid(current.delivery))
    return false;
  output = {.raw_time_ns = time_ns,
            .clock_domain_id = source.clock_domain_id,
            .kind = wire,
            .presence = 0x25561,
            .generation = generation,
            .instance_index = execution.key.instance_index,
            .invocation_sequence = execution.key.invocation_sequence,
            .lane_index = lane,
            .lease_sequence = execution.executor_lease_sequence,
            .local_trace_id = current.local_trace_id,
            .layer_ordinal = execution.layer_ordinal,
            .payload_rows = execution.payload_rows,
            .payload_row_capacity = current.payload_row_capacity,
            .output_requirement = static_cast<std::uint32_t>(execution.output_requirement),
            .delivery = static_cast<std::uint32_t>(current.delivery),
            .role = 4,
            .pe = pe};
  return true;
}

} // namespace xpool::devkit::timeline
