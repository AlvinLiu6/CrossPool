#pragma once

/// \file xpool/devkit/timeline/handoff.hpp
/// \brief Borrowed allocation-free metadata resolution after a completed copy.
#include <span>

#include <xpool/devkit/timeline/batch.hpp>
namespace xpool::devkit::timeline {
/// Registered preparation owner remains alive through all collection calls.
/// The callback freezes a revision covering the completed copy, without I/O.
struct MetadataResolver {
  /// Borrowed semantic registry owned by the process/channel.
  void *context = nullptr;
  /// Resolve only after Host payload completion, before Writer handoff.
  BatchInfo (*freeze)(void *, std::span<const Event>) = nullptr;
};
} // namespace xpool::devkit::timeline
