#pragma once

/// \file xpool/devkit/timeline/hostchannel.hpp
/// \brief Budget-owned Host source, independent scratch and immutable batches.

#include <memory>
#include <optional>

#include <xpool/devkit/timeline/collector.hpp>
#include <xpool/devkit/timeline/resource.hpp>

namespace xpool::devkit::timeline {

/// Retain one shared channel through every producer/Collector/Writer operation.
/// No destructor establishes caller retirement. Control and shared weak-reference
/// storage are charged using the implementation's actual allocator request.
class HostChannel {
  class ConstructionKey {
    friend class HostChannel;
    ConstructionKey() = default;
  };

public:
  /// Prepare a complete bounded Host pipeline before publishing any Hook view.
  /// \throws CollectionError for registration, size, admission or allocation.
  static std::shared_ptr<HostChannel> create(std::shared_ptr<MemoryBudget> host, BatchSource source, BatchInfo info,
                                             std::size_t slots, std::size_t batches, std::size_t events_per_batch);
  /// Stable registered producer facade.
  HostBuffer &buffer() noexcept { return *buffer_; }
  /// Serialized collection/Writer lane interface.
  Collector &collector() noexcept { return *collector_; }
  /// Initial immutable metadata; a resolver may freeze a later complete revision.
  const BatchInfo &info() const noexcept { return info_; }
  /// Allocator-only constructor; create() admits control and all three slabs.
  HostChannel(ConstructionKey, std::shared_ptr<MemoryBudget> host, BatchSource source, BatchInfo info,
              std::size_t slots, std::size_t batches, std::size_t events_per_batch);

private:
  friend class CollectionService;
  const std::shared_ptr<MemoryBudget> host_;
  const BatchInfo info_;
  MemoryBudget::HostAllocation delivery_;
  MemoryBudget::HostAllocation scratch_;
  std::optional<HostBuffer> buffer_;
  std::optional<HostBatchPool> batches_;
  std::optional<Collector> collector_;
};

} // namespace xpool::devkit::timeline
