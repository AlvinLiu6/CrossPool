#pragma once

/// \file xpool/devkit/timeline/view.hpp
/// \brief Trivially copyable borrowed Device collection descriptor.

#include <cstddef>
#include <type_traits>

#include <xpool/devkit/timeline/source.hpp>

namespace xpool::devkit::timeline {

/// Host-prepared Device descriptor. Registration, geometry, owning physical
/// Device and disjoint initialized allocations are validated before launch.
/// All pointed-to storage outlives every kernel, transfer and acquired reader.
struct BufferView {
  /// Immutable Run, Producer and raw clock domain registration.
  BatchSource source{};
  /// Device payload array containing capacity Events.
  Event *events = nullptr;
  /// Device slot controls, initialized before launch.
  SlotControl *slots = nullptr;
  /// Device scheduling and private-generation stop state.
  SlotPoolState *pool_state = nullptr;
  /// Device automatic identities and cumulative outcomes.
  DeviceSourceState *source_state = nullptr;
  /// Matching positive payload and control lengths.
  std::size_t capacity = 0;
};

static_assert(std::is_trivially_copyable_v<BufferView>);

} // namespace xpool::devkit::timeline
