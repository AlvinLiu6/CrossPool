#pragma once

/// \file xpool/devkit/timeline/hostbuffer.hpp
/// \brief Budget-owned Host source storage and typed C0 producer entry points.

#include <cstddef>
#include <optional>

#include <xpool/devkit/timeline/resource.hpp>
#include <xpool/devkit/timeline/source.hpp>

namespace xpool::devkit::timeline {

/// Stable source storage. The budget and this facade outlive producer calls and
/// acquired readers. Destruction requires their retirement; stop alone is not
/// retirement. The enclosing owner accounts sizeof(HostBuffer) once as control.
class HostBuffer {
public:
  /// Admit one complete actual source layout before allocating its Host slab.
  /// \throws CollectionError resource_budget/allocation for setup failures.
  HostBuffer(MemoryBudget &budget, BatchSource source, std::size_t capacity);
  /// Borrow initialized caller-owned storage instead. The caller admits and
  /// retains those exact allocations through all readers; no storage is owned.
  /// \throws CollectionError for invalid source/storage/history.
  HostBuffer(BatchSource source, std::span<Event> events, std::span<SlotControl> slots, SlotPoolState &pool,
             SourceState &state);
  /// Storage ownership cannot be copied or moved after views are published.
  HostBuffer(const HostBuffer &) = delete;
  /// Storage ownership cannot be replaced while readers exist.
  HostBuffer &operator=(const HostBuffer &) = delete;
  /// Registered provenance and bounded source operations used by Collector.
  HostEventSource &source() noexcept { return *source_; }
  /// Read registration/progress without changing source ownership.
  const HostEventSource &source() const noexcept { return *source_; }
  /// Active cumulative lower bounds, never a fabricated final pending count.
  SourceProgress progress() const { return source_->progress(); }
  /// Requires every source call to have quiesced; may still report Reserved.
  SourceCounts quiescent_counts() const { return source_->quiescent_counts(); }
  /// Close new admission; existing tokens remain retireable.
  void stop() { source_->stop(); }
  /// Complete charged slab bytes; facade/control belongs to its enclosing owner.
  std::size_t storage_bytes() const noexcept { return storage_.charged_bytes(); }
  /// Actual checked layout, available before allocation or CUDA initialization.
  static SourceLayout layout(std::size_t capacity) { return HostEventSource::layout(capacity); }

private:
  MemoryBudget::HostAllocation storage_;
  std::optional<HostEventSource> source_;
};

/// Allocate an attempted identity and bounded slot authority, including Full.
/// \throws CollectionError resource_budget if concurrency exceeds its bound.
ReserveResult try_reserve_host(HostBuffer &buffer);
/// Commit a complete event. The token supplies its sequence. Wrong-clock input
/// leaves the reservation intact and raises CollectionError metadata_unavailable.
CommitStatus commit_host(HostBuffer &buffer, WriteToken token, const Event &event);
/// Retire an uncommitted reservation; rejected tokens only add diagnostics.
AbandonStatus abandon_host(HostBuffer &buffer, WriteToken token);

} // namespace xpool::devkit::timeline
