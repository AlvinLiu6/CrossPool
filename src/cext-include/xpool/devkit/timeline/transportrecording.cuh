#pragma once

/// \file xpool/devkit/timeline/transportrecording.cuh
/// \brief Bounded Transport recording on the existing Device Source.

#include <xpool/devkit/timeline/collection.cuh>
#include <xpool/devkit/timeline/transportevent.hpp>

namespace xpool::devkit::timeline {

/// The serialized protocol owner calls this once at each actual Hook. Current
/// facts and original operation counts advance even after an event is lost.
/// It never waits, allocates memory, or issues a second Operation identity.
XPOOL_DEVICE_FN inline void record_transport_event(TransportRecordingView view, TransportRecordingEndpoint &endpoint,
                                                   xpool::hooks::TransportProtocolEventKind kind,
                                                   const xpool::transport::OperationFact &operation,
                                                   const xpool::transport::RequestMetadata *request, std::uint64_t rows,
                                                   xpool::ffn::ResultCode result, std::uint64_t time_ns) {
  if (!detail::valid(view.buffer) || endpoint.endpoint_owner == 0)
    return;
  using Kind = xpool::hooks::TransportProtocolEventKind;
  const auto begin = endpoint.role == 1 ? kind == Kind::RequestStagingStarted : kind == Kind::RequestObserved;
  auto &current = endpoint.current;
  if (begin) {
    current = {};
    if (endpoint.operations == ~std::uint64_t{0}) {
      endpoint.exhausted = true;
      detail::source(view.buffer).stop_identity_exhausted();
      return;
    }
    ++endpoint.operations;
    if (endpoint.operations > endpoint.capacity)
      ++endpoint.dropped;
    if (!request) {
      endpoint.diagnostics |= 1U;
      return;
    }
    current.local_trace_id = endpoint.operations;
    current.operation = operation;
    current.request = *request;
    current.payload_rows = rows;
  }
  if (operation.status != xpool::transport::OperationStatus::Available || operation.sequence == 0 ||
      operation.sequence != current.operation.sequence ||
      operation.endpoint_owner != current.operation.endpoint_owner ||
      operation.endpoint_sequence != current.operation.endpoint_sequence ||
      operation.generation != current.operation.generation) {
    endpoint.diagnostics |= 2U;
    if (operation.status == xpool::transport::OperationStatus::IdentityExhausted)
      detail::source(view.buffer).stop_identity_exhausted();
    return;
  }
  if (kind == Kind::ResultPublished || kind == Kind::ResultObserved ||
      (kind == Kind::Closed && !current.result_known)) {
    current.result = result;
    current.result_known = true;
  }
  auto event = Event{};
  if (!make_transport_event(endpoint, view.buffer.source, kind, time_ns, event)) {
    endpoint.diagnostics |= 4U;
  } else {
    const auto reservation = try_reserve_device(view.buffer);
    if (reservation.status == ReserveStatus::Reserved &&
        commit_device(view.buffer, reservation.token, event) != CommitStatus::Committed)
      endpoint.diagnostics |= 8U;
  }
  if (kind == Kind::ResultPublished || kind == Kind::ResultAcknowledged || kind == Kind::Closed)
    current = {};
}

} // namespace xpool::devkit::timeline
