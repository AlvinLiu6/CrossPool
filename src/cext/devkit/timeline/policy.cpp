#include <xpool/devkit/timeline/policy.hpp>

#include <span>

#include <xpool/devkit/timeline/digest.hpp>
#include <xpool/devkit/timeline/error.hpp>

namespace xpool::devkit::timeline {

bool CollectionPolicy::enabled() const noexcept {
  return transport_events() || fabric_events() || graph_events() || routing_events() ||
         explicit_transport_identity_consumer;
}
bool CollectionPolicy::transport_identity_required() const noexcept {
  return unified || legacy.transport_observer.enable || explicit_transport_identity_consumer;
}

CollectionPolicy make_collection_policy(const Options &options, const xpool::debug::Options &legacy,
                                        std::array<std::byte, 32> configuration_fingerprint,
                                        bool explicit_transport_identity_consumer) {
  options.validate();
  if ((legacy.transport_observer.enable && legacy.transport_observer.record_capacity == 0) ||
      (legacy.fabric_observer.enable && legacy.fabric_observer.record_capacity == 0) ||
      (legacy.ffn_routing_observer.enable && legacy.ffn_routing_observer.record_capacity == 0))
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0,
                          "xpool enabled legacy observation requires positive capacity");
  auto policy = CollectionPolicy{
      .limits = {options.device_bytes, options.host_bytes, options.chunk_bytes, options.collection_interval_ms,
                 options.disk_bytes, options.metadata_reserve_bytes, options.shutdown_wait_ms,
                 options.calibration_block_attempts, options.device_event_slot_capacity, options.endpoint_capacity,
                 options.context_slot_capacity, options.sampler_slot_capacity},
      .legacy = legacy,
      .unified = options.enable,
      .explicit_transport_identity_consumer = explicit_transport_identity_consumer,
      .configuration_fingerprint = configuration_fingerprint,
      .unified_outdir_fingerprint = std::nullopt};
  if (options.outdir) {
    const auto &path = options.outdir->native();
    policy.unified_outdir_fingerprint = sha256(std::as_bytes(std::span{path.data(), path.size()}));
  }
  return policy;
}

} // namespace xpool::devkit::timeline
