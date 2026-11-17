#pragma once

/// \file xpool/devkit/ffn_routing_observer.hpp
/// \brief Routing tensors collected at real publication and copied after drain.
#include <cstddef>

#include <xpool/fabric/projection.hpp>

namespace xpool::devkit::ffn_routing_observer {
/// Submit the installed Routing payload once after actual producer drain.
/// No file I/O or serialization occurs here; failures are explicit.
void collect_drained();
/// Complete independent payload and Host copy/serialization coexistence demand.
struct MemoryDemand {
  /// Complete Device payload, control and alignment bytes.
  std::size_t device_bytes = 0;
  /// Simultaneously live copy, tensor and serialization bytes.
  std::size_t host_bytes = 0;
};
/// Determine complete allocation admission from the actual placed projection.
MemoryDemand memory_demand(const xpool::fabric::ArenaProjection &projection, std::size_t ffnagent_index);
/// Actual maximum Routing width on the selected FfnAgent, before allocation.
std::size_t element_capacity(const xpool::fabric::ArenaProjection &projection, std::size_t ffnagent_index);
/// Actual independent Routing payload allocation requirement.
std::size_t allocation_bytes(std::size_t element_capacity);
} // namespace xpool::devkit::ffn_routing_observer
