#pragma once

/// \file xpool/devkit/timeline/transfer.hpp
/// \brief Async immutable Device packing, D2H and completion-gated reclamation.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include <xpool/devkit/timeline/collector.hpp>
#include <xpool/devkit/timeline/transferlayout.hpp>
#include <xpool/devkit/timeline/view.hpp>

namespace xpool::devkit::timeline {

/// Serial collection-lane driver over owner-provided, budgeted storage. The
/// owner validates initialized source storage, one nonblocking management
/// Stream, two distinct disable-timing events, Device frame and page-locked
/// Host frame on the owning Device. Every resource and this driver outlives
/// all submitted operations, including errors and shutdown timeout. This layer
/// performs no allocation/free, file I/O or production-Stream synchronization.
class DeviceTransfer {
public:
  /// Bind source, matching Collector and fixed transfer geometry. Construction
  /// is a pure descriptor check; CUDA work starts only on tick().
  /// \throws CollectionError for invalid source, geometry or borrowed resources.
  DeviceTransfer(BufferView source, Collector &collector, TransferLayout layout, std::byte *device_frame,
                 std::byte *pinned_frame, cudaStream_t stream, cudaEvent_t copied, cudaEvent_t reclaimed,
                 FabricRecordingView fabric = {});
  DeviceTransfer(const DeviceTransfer &) = delete;
  DeviceTransfer &operator=(const DeviceTransfer &) = delete;
  /// Progress at most one state transition. A supplied resolver runs only after
  /// the D2H completion query succeeds and freezes metadata over that completed
  /// copy. Otherwise metadata is a validated fixed registration snapshot.
  /// producer_retired is a
  /// runtime proof that every source API operation has actually retired, never
  /// inferred from elapsed time or source admission closure. If true, stop
  /// admission and drain before sealing final source counts on an empty pass.
  /// \throws CollectionError with transfer on CUDA failure. In that state
  /// retain all resources until external CUDA retirement proves they are idle.
  bool tick(BatchInfo metadata, bool producer_retired = false, bool stop_admission = false,
            MetadataResolver resolver = {}, ReconstructionSink reconstruction = {});
  /// True while operations are outstanding or retirement is uncertain on error.
  bool in_flight() const noexcept;
  /// A final empty pass and reclamation have sealed this source's true ledger.
  bool closed() const noexcept;
  /// Collection-lane request for a fresh complete empty scan. The caller holds
  /// original producers quiescent through acknowledgment and dependent reads.
  /// A transfer launched before this call cannot acknowledge its identity.
  void request_drain(std::uint64_t identity) noexcept { requested_drain_ = identity; }
  /// Collection-lane completion after that empty scan's actual reclamation
  /// event query. A retired Source has already proved its final complete drain.
  bool drained(std::uint64_t identity) const noexcept { return closed() || completed_drain_ >= identity; }

private:
  enum class Phase { Idle, Copying, Reclaiming, Closed, Uncertain };
  void check(cudaError_t status, const char *operation);
  const BufferView source_;
  const FabricRecordingView fabric_;
  Collector &collector_;
  const TransferLayout layout_;
  std::byte *const device_;
  std::byte *const host_;
  const cudaStream_t stream_;
  const cudaEvent_t copied_;
  const cudaEvent_t reclaimed_;
  Phase phase_ = Phase::Idle;
  BatchInfo metadata_{};
  std::size_t cursor_ = 0;
  bool producer_retired_ = false;
  std::uint64_t requested_drain_ = 0;
  std::uint64_t active_drain_ = 0;
  std::uint64_t completed_drain_ = 0;
};

} // namespace xpool::devkit::timeline
