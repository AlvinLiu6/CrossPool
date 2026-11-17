#pragma once

/// \file xpool/devkit/timeline/routingevent.hpp
/// \brief Routing publication facts independent of tensor and Event capacity.
#include <xpool/devkit/timeline/contract.hpp>
#include <xpool/devkit/timeline/view.hpp>
#include <xpool/fabric/protocol.hpp>

namespace xpool::devkit::timeline {
/// Immutable publication outcome; final file outcomes belong to the index.
enum class RoutingAttachmentState : std::uint32_t {
  Reserved = 1,
  NotCollected = 2,
  CapacityExhausted = 3,
  PayloadOversize = 4,
};
/// Assemble facts at the actual RoutingMetadataPublished Hook. Attachment
/// identity is allocated separately before this call; no Event is reserved here.
XPOOL_HOST_DEVICE_FN inline Event make_routing_event(BatchSource source, std::uint64_t generation, std::int32_t pe,
                                                     const xpool::fabric::LaneExecution &execution, std::size_t lane,
                                                     std::uint64_t sequence, std::size_t topk,
                                                     RoutingAttachmentState state, std::uint64_t time_ns) {
  return {.raw_time_ns = time_ns,
          .clock_domain_id = source.clock_domain_id,
          .kind = static_cast<std::uint32_t>(EventKind::RoutingPublished),
          .presence = static_cast<std::uint32_t>(Presence::Generation) |
                      static_cast<std::uint32_t>(Presence::Invocation) | static_cast<std::uint32_t>(Presence::Lease) |
                      static_cast<std::uint32_t>(Presence::Attachment) |
                      static_cast<std::uint32_t>(Presence::InvocationFacts) | static_cast<std::uint32_t>(Presence::Pe),
          .generation = generation,
          .instance_index = execution.key.instance_index,
          .invocation_sequence = execution.key.invocation_sequence,
          .lane_index = lane,
          .lease_sequence = execution.executor_lease_sequence,
          .attachment_owner = source.producer,
          .attachment_sequence = sequence,
          .layer_ordinal = execution.layer_ordinal,
          .payload_rows = execution.payload_rows,
          .output_requirement = static_cast<std::uint32_t>(execution.output_requirement),
          .role = static_cast<std::uint32_t>(EventRole::FfnAgent),
          .pe = pe,
          .data0 = static_cast<std::uint64_t>(state),
          .data1 = topk};
}
} // namespace xpool::devkit::timeline
