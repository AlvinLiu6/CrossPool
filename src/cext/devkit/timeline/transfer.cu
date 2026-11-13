#include <xpool/devkit/timeline/transfer.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>

#include <cuda/atomic>

#include <xpool/devkit/timeline/collection.cuh>
#include <xpool/devkit/timeline/fabricrecording.cuh>
#include <xpool/macros.hpp>

namespace xpool::devkit::timeline {

namespace {

XPOOL_KERNEL_FN void acquire_frame(BufferView view, std::byte *frame, TransferLayout layout, std::size_t cursor,
                                   bool stop, bool retired, FabricRecordingView fabric, bool quiescent) {
  const auto source = detail::source(view);
  if (stop || retired) {
    source.stop();
  }
  auto &header = *reinterpret_cast<TransferHeader *>(frame);
  auto *tokens = reinterpret_cast<ReadToken *>(frame + layout.token_offset);
  header.count = 0;
  header.release_failures = 0;
  auto index = cursor;
  for (auto inspected = std::size_t{0}; inspected < view.capacity && header.count < layout.capacity; ++inspected) {
    const auto read = source.try_acquire_at(index);
    if (read.acquired) {
      tokens[header.count++] = read.token;
    }
    index = index + 1 == view.capacity ? 0 : index + 1;
  }
  header.progress =
      retired ? SourceProgress{.counts = source.quiescent_counts(), .pending_known = true} : source.progress();
  header.fabric = fabric_progress(fabric, quiescent);
}

XPOOL_KERNEL_FN void pack_frame(BufferView view, std::byte *frame, TransferLayout layout) {
  const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const auto &header = *reinterpret_cast<const TransferHeader *>(frame);
  if (index < header.count) {
    const auto *tokens = reinterpret_cast<const ReadToken *>(frame + layout.token_offset);
    auto *events = reinterpret_cast<Event *>(frame + layout.event_offset);
    events[index] = view.events[tokens[index].slot];
  }
}

XPOOL_KERNEL_FN void reclaim_frame(BufferView view, std::byte *frame, TransferLayout layout) {
  const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  auto &header = *reinterpret_cast<TransferHeader *>(frame);
  if (index < header.count) {
    const auto *tokens = reinterpret_cast<const ReadToken *>(frame + layout.token_offset);
    if (detail::source(view).release(tokens[index]) != ReleaseStatus::Released) {
      cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{header.release_failures}.fetch_add(
          std::uint64_t{1}, cuda::memory_order_relaxed);
    }
  }
}

std::uint64_t host_time(std::uint64_t producer) {
  auto time = timespec{};
  if (::clock_gettime(CLOCK_MONOTONIC, &time) != 0) {
    throw CollectionError(CollectionErrorCode::Transfer, producer, "xpool cannot read transfer clock", std::nullopt,
                          std::strerror(errno));
  }
  return static_cast<std::uint64_t>(time.tv_sec) * 1000000000ULL + static_cast<std::uint64_t>(time.tv_nsec);
}

} // namespace

DeviceTransfer::DeviceTransfer(BufferView source, Collector &collector, TransferLayout layout, std::byte *device_frame,
                               std::byte *pinned_frame, cudaStream_t stream, cudaEvent_t copied, cudaEvent_t reclaimed,
                               FabricRecordingView fabric)
    : source_(source), fabric_(fabric), collector_(collector), layout_(layout), device_(device_frame),
      host_(pinned_frame), stream_(stream), copied_(copied), reclaimed_(reclaimed) {
  const auto expected = TransferLayout::for_capacity(layout.capacity);
  if (fabric.state &&
      (fabric.buffer.events != source.events || fabric.buffer.slots != source.slots ||
       fabric.buffer.pool_state != source.pool_state || fabric.buffer.source_state != source.source_state ||
       fabric.buffer.capacity != source.capacity || fabric.buffer.source.run_id != source.source.run_id ||
       fabric.buffer.source.producer != source.source.producer ||
       fabric.buffer.source.clock_domain_id != source.source.clock_domain_id))
    throw CollectionError(CollectionErrorCode::Allocation, source.source.producer,
                          "xpool Fabric progress must belong to the transferred Source");
  if (!source.events || !source.slots || !source.pool_state || !source.source_state || source.capacity == 0 ||
      source.source.producer != collector.source().producer ||
      source.source.clock_domain_id != collector.source().clock_domain_id ||
      source.source.run_id != collector.source().run_id || !device_ || !host_ || !stream_ || !copied_ || !reclaimed_ ||
      copied_ == reclaimed_ || layout.event_offset != expected.event_offset ||
      layout.host_bytes != expected.host_bytes || layout.token_offset != expected.token_offset ||
      layout.device_bytes != expected.device_bytes || layout.capacity > collector.batches().events_per_batch()) {
    throw CollectionError(CollectionErrorCode::Allocation, source.source.producer,
                          "xpool async transfer descriptors are invalid");
  }
}

void DeviceTransfer::check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    phase_ = Phase::Uncertain;
    collector_.fail_transfer();
    throw CollectionError(CollectionErrorCode::Transfer, source_.source.producer, operation, std::nullopt,
                          cudaGetErrorString(status));
  }
}

bool DeviceTransfer::tick(BatchInfo metadata, bool producer_retired, bool stop_admission, MetadataResolver resolver,
                          ReconstructionSink reconstruction) {
  try {
    if (phase_ == Phase::Closed) {
      return false;
    }
    if (phase_ == Phase::Uncertain) {
      throw CollectionError(CollectionErrorCode::Transfer, source_.source.producer,
                            "xpool transfer retirement is uncertain");
    }
    if (phase_ == Phase::Idle) {
      if (metadata.metadata_revision == 0) {
        collector_.fail(StopReason::MetadataUnavailable);
        throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.source.producer,
                              "xpool transfer metadata is unavailable");
      }
      metadata_ = metadata;
      metadata_.host_window_begin_ns = host_time(source_.source.producer);
      producer_retired_ = producer_retired;
      active_drain_ = requested_drain_ > completed_drain_ ? requested_drain_ : 0;
      // Any failure after the first launch can leave outstanding ownership.
      phase_ = Phase::Uncertain;
      acquire_frame<<<1, 1, 0, stream_>>>(source_, device_, layout_, cursor_,
                                          collector_.stop_requested() || stop_admission, producer_retired, fabric_,
                                          producer_retired || active_drain_ != 0);
      check(cudaGetLastError(), "xpool acquire launch failed");
      const auto blocks = static_cast<unsigned>((layout_.capacity + 255) / 256);
      pack_frame<<<blocks, 256, 0, stream_>>>(source_, device_, layout_);
      check(cudaGetLastError(), "xpool pack launch failed");
      check(cudaMemcpyAsync(host_, device_, layout_.host_bytes, cudaMemcpyDeviceToHost, stream_),
            "xpool device-to-host copy failed");
      check(cudaEventRecord(copied_, stream_), "xpool copy event record failed");
      phase_ = Phase::Copying;
      const auto increment = layout_.capacity % source_.capacity;
      cursor_ =
          increment >= source_.capacity - cursor_ ? increment - (source_.capacity - cursor_) : cursor_ + increment;
      return true;
    }
    // These events belong to the preallocated noncapturing collection Stream.
    // A model may capture on another thread; its Global mode must not prohibit
    // polling this independent transfer. Restore the caller's mode before any
    // status handling can return or throw.
    auto capture_mode = cudaStreamCaptureModeThreadLocal;
    check(cudaThreadExchangeStreamCaptureMode(&capture_mode), "xpool query capture mode entry failed");
    const auto status = cudaEventQuery(phase_ == Phase::Copying ? copied_ : reclaimed_);
    check(cudaThreadExchangeStreamCaptureMode(&capture_mode), "xpool query capture mode restoration failed");
    if (status == cudaErrorNotReady) {
      return false;
    }
    check(status, phase_ == Phase::Copying ? "xpool copy event query failed" : "xpool reclaim event query failed");
    const auto &header = *reinterpret_cast<const TransferHeader *>(host_);
    if (header.count > layout_.capacity) {
      check(cudaErrorInvalidValue, "xpool copied frame count exceeds capacity");
    }
    if (phase_ == Phase::Copying) {
      // Do not replay stage accounting if a Host-side invariant throws. The
      // owner retains this frame and all held reads until retirement is proved.
      phase_ = Phase::Uncertain;
      collector_.observe_source(header.progress);
      if (fabric_.state != nullptr)
        collector_.observe_fabric(header.fabric);
      collector_.record_copied(header.count);
      const auto input = std::span{reinterpret_cast<const Event *>(host_ + layout_.event_offset),
                                   static_cast<std::size_t>(header.count)};
      auto reconstruction_failed = false;
      try {
        if (!input.empty() && resolver.freeze) {
          const auto begin = metadata_.host_window_begin_ns;
          metadata_ = resolver.freeze(resolver.context, input);
          metadata_.host_window_begin_ns = begin;
        }
        if (!input.empty() && reconstruction.merge)
          reconstruction.merge(reconstruction.context, input);
      } catch (const CollectionError &error) {
        collector_.record_unstaged_loss(input.size());
        if (error.code() == CollectionErrorCode::Transfer)
          throw;
        collector_.fail(StopReason::MetadataUnavailable);
        reconstruction_failed = true;
      } catch (const std::exception &) {
        collector_.record_unstaged_loss(input.size());
        collector_.fail(StopReason::MetadataUnavailable);
        reconstruction_failed = true;
      }
      const auto batch = reconstruction_failed ? BatchReservation{} : collector_.batches().try_reserve();
      if (batch.status == BatchReserveStatus::Reserved) {
        if (header.count == 0) {
          collector_.batches().cancel(batch.token);
        } else {
          std::ranges::copy(input, batch.events.begin());
          std::ranges::sort(batch.events.first(input.size()), {}, &Event::event_sequence);
          try {
            metadata_.host_window_end_ns = host_time(source_.source.producer);
            if (collector_.batches().publish(batch.token, input.size(), metadata_) != BatchPublishStatus::Published) {
              throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.source.producer,
                                    "xpool transfer batch publication rejected its owner");
            }
          } catch (const CollectionError &error) {
            collector_.record_unstaged_loss(input.size());
            collector_.batches().cancel(batch.token);
            if (error.code() == CollectionErrorCode::Transfer)
              throw;
            collector_.fail(StopReason::MetadataUnavailable);
          } catch (const std::exception &) {
            collector_.record_unstaged_loss(input.size());
            collector_.batches().cancel(batch.token);
            collector_.fail(StopReason::MetadataUnavailable);
          }
        }
      } else if (!reconstruction_failed) {
        collector_.record_unstaged_loss(header.count);
        if (batch.status == BatchReserveStatus::IdentityExhausted) {
          collector_.fail(StopReason::IdentityExhausted);
        }
      }
      phase_ = Phase::Uncertain;
      const auto blocks = static_cast<unsigned>((layout_.capacity + 255) / 256);
      reclaim_frame<<<blocks, 256, 0, stream_>>>(source_, device_, layout_);
      check(cudaGetLastError(), "xpool reclaim launch failed");
      check(cudaMemcpyAsync(host_, device_, sizeof(TransferHeader), cudaMemcpyDeviceToHost, stream_),
            "xpool reclaim header copy failed");
      check(cudaEventRecord(reclaimed_, stream_), "xpool reclaim event record failed");
      phase_ = Phase::Reclaiming;
      return true;
    }
    if (header.release_failures != 0) {
      check(cudaErrorInvalidValue, "xpool reclaim frame has rejected ownership tokens");
    }
    if (header.count == 0 && active_drain_ != 0)
      completed_drain_ = active_drain_;
    if (producer_retired_ && header.count == 0) {
      collector_.seal_source(header.progress.counts);
      phase_ = Phase::Closed;
    } else {
      phase_ = Phase::Idle;
    }
    return true;
  } catch (const CollectionError &error) {
    if (error.code() == CollectionErrorCode::Transfer) {
      phase_ = Phase::Uncertain;
      collector_.fail_transfer();
    }
    throw;
  } catch (...) {
    if (phase_ == Phase::Uncertain)
      collector_.fail_transfer();
    throw;
  }
}

bool DeviceTransfer::in_flight() const noexcept { return phase_ != Phase::Idle && phase_ != Phase::Closed; }
bool DeviceTransfer::closed() const noexcept { return phase_ == Phase::Closed; }

} // namespace xpool::devkit::timeline
