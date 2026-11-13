#pragma once

/// \file xpool/devkit/timeline/devicesequence.cuh
/// \brief Bounded full-width Device sequence allocation over borrowed storage.

#include <cstdint>

#include <cuda/atomic>

#include <xpool/devkit/timeline/source.hpp>
#include <xpool/macros.hpp>

namespace xpool::devkit::timeline::detail {

/// \cond INTERNAL

// At most kAttemptConcurrencyLimit callers overlap. Overshoot remains inside
// the unused half-word; no CAS retry, waiting, or wrapping published identity.
// Storage is initialized before launch and inspected only after quiescence.
XPOOL_DEVICE_FN inline std::uint64_t next_device_sequence(DeviceAttemptState &state) {
  constexpr auto upper_capacity = std::uint64_t{1} << 63;
  constexpr auto lower_capacity = upper_capacity - 1;
  auto range = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state.upper_range};
  if (range.fetch_or(std::uint64_t{0}, cuda::memory_order_acquire) == 0) {
    const auto ticket = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state.lower}.fetch_add(
        std::uint64_t{1}, cuda::memory_order_relaxed);
    if (ticket < lower_capacity) {
      if (ticket == lower_capacity - 1)
        range.store(std::uint64_t{1}, cuda::memory_order_release);
      return ticket + 1;
    }
    range.store(std::uint64_t{1}, cuda::memory_order_release);
  }
  auto upper = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state.upper};
  if (upper.fetch_or(std::uint64_t{0}, cuda::memory_order_relaxed) >= upper_capacity)
    return 0;
  const auto ticket = upper.fetch_add(std::uint64_t{1}, cuda::memory_order_relaxed);
  return ticket < upper_capacity ? upper_capacity + ticket : 0;
}

/// \endcond

} // namespace xpool::devkit::timeline::detail
