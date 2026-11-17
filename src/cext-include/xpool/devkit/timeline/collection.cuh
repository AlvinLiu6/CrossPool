#pragma once

/// \file xpool/devkit/timeline/collection.cuh
/// \brief C0 Device reservation, commit and abandon entry points.

#include <cuda/std/span>

#include <xpool/devkit/timeline/devicesource.cuh>
#include <xpool/devkit/timeline/view.hpp>
#include <xpool/macros.hpp>

namespace xpool::devkit::timeline {

namespace detail {

/// Check installation before dereferencing any borrowed Device pointer.
XPOOL_DEVICE_FN inline bool valid(BufferView buffer) {
  return buffer.events != nullptr && buffer.slots != nullptr && buffer.pool_state != nullptr &&
         buffer.source_state != nullptr && buffer.capacity != 0 && buffer.source.producer != 0 &&
         buffer.source.clock_domain_id != 0;
}

/// Borrow after valid() establishes installed geometry. Host setup owns the
/// allocation and metadata validation; this conversion performs no allocation.
XPOOL_DEVICE_FN inline DeviceEventSource source(BufferView buffer) {
  return {buffer.source, cuda::std::span{buffer.events, buffer.capacity},
          cuda::std::span{buffer.slots, buffer.capacity}, *buffer.pool_state, *buffer.source_state};
}

} // namespace detail

/// Reserve through one registered Device source using bounded nonblocking work.
/// Failed results grant zero tokens. Empty/uninstalled views allocate nothing.
XPOOL_DEVICE_FN inline ReserveResult try_reserve_device(BufferView buffer) {
  return detail::valid(buffer) ? detail::source(buffer).try_reserve() : ReserveResult{.status = ReserveStatus::Stopped};
}

/// Publish once under the token identity. The complete Event describes this
/// source and its registered clock. A collective Hook selects its own emitter.
XPOOL_DEVICE_FN inline CommitStatus commit_device(BufferView buffer, WriteToken token, const Event &event) {
  return detail::valid(buffer) ? detail::source(buffer).commit(token, event) : CommitStatus::StaleToken;
}

/// Retire an uncommitted reservation once; no new Event is published.
XPOOL_DEVICE_FN inline AbandonStatus abandon_device(BufferView buffer, WriteToken token) {
  return detail::valid(buffer) ? detail::source(buffer).abandon(token) : AbandonStatus::StaleToken;
}

} // namespace xpool::devkit::timeline
