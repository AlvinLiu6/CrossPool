#pragma once

/// \file xpool/devkit/timeline/sink.hpp
/// \brief Borrowed bounded reconstruction over completed immutable Event copies.

#include <span>

#include <xpool/devkit/timeline/event.hpp>

namespace xpool::devkit::timeline {

/// Collection-lane reconstruction boundary, independent of Writer publication.
/// The retained channel owns context through every call. merge consumes the
/// same registered Producer's completed copy synchronously and retains finite
/// facts only; it performs no disk I/O, CUDA synchronization or dynamic growth.
/// A missing callback disables legacy reconstruction for this channel.
struct ReconstructionSink {
  /// Registered bounded Host view registry.
  void *context = nullptr;
  /// Merge immutable copied Events before their staging can be recycled.
  void (*merge)(void *, std::span<const Event>) = nullptr;
};

} // namespace xpool::devkit::timeline
