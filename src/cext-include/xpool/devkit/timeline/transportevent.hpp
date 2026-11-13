#pragma once

/// \file xpool/devkit/timeline/transportevent.hpp
/// \brief Assemble ordinary and anomalous Transport events from original facts.

#include <xpool/devkit/timeline/contract.hpp>
#include <xpool/devkit/timeline/transportrecording.hpp>
#include <xpool/transport/hooks.hpp>

namespace xpool::devkit::timeline {

/// Explicit Hook-to-wire mapping; production enum ordinals are not a contract.
XPOOL_HOST_DEVICE_FN inline std::uint32_t transport_boundary(xpool::hooks::TransportProtocolEventKind kind) {
  using Kind = xpool::hooks::TransportProtocolEventKind;
  switch (kind) {
  case Kind::RequestStagingStarted:
    return 100;
  case Kind::RequestStagingCompleted:
    return 101;
  case Kind::RequestPublished:
    return 102;
  case Kind::RequestObserved:
    return 103;
  case Kind::ExecutionStarted:
    return 104;
  case Kind::ExecutionCompleted:
    return 105;
  case Kind::ResultPublished:
    return 106;
  case Kind::ResultObserved:
    return 107;
  case Kind::OutputCopied:
    return 108;
  case Kind::ResultAcknowledged:
    return 109;
  case Kind::Closed:
    return 110;
  default:
    return 0;
  }
}

/// No identity allocation or reservation. Invalid enums become kind150; legal
/// enums with bad geometry remain ordinary events. False clears the output and
/// means required identity/current/result facts are unavailable, not a drop count.
XPOOL_HOST_DEVICE_FN inline bool make_transport_event(const TransportRecordingEndpoint &endpoint, BatchSource source,
                                                      xpool::hooks::TransportProtocolEventKind kind,
                                                      std::uint64_t time_ns, Event &output) {
  output = {};
  const auto boundary = transport_boundary(kind);
  const auto &current = endpoint.current;
  const auto &operation = current.operation;
  const auto own = endpoint.role == 1 ? (boundary >= 100 && boundary <= 102) || (boundary >= 107 && boundary <= 110)
                                      : endpoint.role == 2 && boundary >= 103 && boundary <= 106;
  if (!own || source.producer == 0 || source.clock_domain_id == 0 || endpoint.endpoint_owner == 0 ||
      endpoint.endpoint_sequence == 0 || endpoint.generation == 0 || current.local_trace_id == 0 ||
      operation.status != xpool::transport::OperationStatus::Available || operation.sequence == 0 ||
      operation.endpoint_owner != endpoint.endpoint_owner ||
      operation.endpoint_sequence != endpoint.endpoint_sequence || operation.generation != endpoint.generation ||
      (endpoint.role == 2 && endpoint.endpoint_owner != source.producer) ||
      (boundary >= 106 && (!current.result_known || !xpool::ffn::is_valid(current.result))))
    return false;
  output = {.raw_time_ns = time_ns,
            .clock_domain_id = source.clock_domain_id,
            .kind = boundary,
            .presence = 0x10107U,
            .generation = endpoint.generation,
            .endpoint_owner = endpoint.endpoint_owner,
            .endpoint_sequence = endpoint.endpoint_sequence,
            .operation_sequence = operation.sequence,
            .local_trace_id = current.local_trace_id,
            .role = endpoint.role,
            .instance_rank = endpoint.instance_rank};
  const auto &request = current.request;
  const auto forward = static_cast<std::uint32_t>(request.forward_mode);
  const auto requirement = static_cast<std::uint32_t>(request.output_requirement);
  const auto layout = static_cast<std::uint32_t>(request.dp_row_layout);
  const auto invalid = transport_invalid_fields(forward, requirement, layout);
  if (invalid != 0) {
    output.kind = static_cast<std::uint32_t>(EventKind::TransportAnomaly);
    output.presence |= static_cast<std::uint32_t>(Presence::RawRequest);
    output.data0 = boundary;
    output.data1 = forward | (static_cast<std::uint64_t>(requirement) << 32);
    output.data2 = layout | (static_cast<std::uint64_t>(invalid) << 32);
    output.data3 = request.layer_ordinal;
    output.data4 = current.payload_rows;
  } else {
    output.presence |= 0x600U;
    output.layer_ordinal = request.layer_ordinal;
    output.payload_rows = current.payload_rows;
    output.forward_mode = forward;
    output.output_requirement = requirement;
    output.dp_row_layout = layout;
  }
  if (boundary >= 106) {
    output.presence |= static_cast<std::uint32_t>(Presence::Result);
    output.result_code = static_cast<std::uint32_t>(current.result);
  }
  return true;
}

} // namespace xpool::devkit::timeline
