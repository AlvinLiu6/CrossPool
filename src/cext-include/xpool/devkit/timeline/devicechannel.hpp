#pragma once

/// \file xpool/devkit/timeline/devicechannel.hpp
/// \brief Budgeted CUDA source/transfer ownership through actual retirement.

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>

#include <xpool/devkit/timeline/resource.hpp>
#include <xpool/devkit/timeline/transfer.hpp>
#include <xpool/devkit/timeline/transportrecording.hpp>

namespace xpool::devkit::timeline {

/// Own one registered Device source, packing frame, pinned frame and Host
/// batch slab. Context/IPC/calibration owners share the same respective budgets.
/// A self-retainer preserves uncertain CUDA ownership after errors or timeout;
/// only proven producer retirement and a completed drain release it.
class DeviceChannel : public std::enable_shared_from_this<DeviceChannel> {
  class ConstructionKey {
    friend class DeviceChannel;
    ConstructionKey() = default;
  };

public:
  /// Admit actual geometry before CUDA allocation. Initialization synchronizes
  /// only this new channel's private management Stream, before production use
  /// or Graph capture. Invalid sizing/admission rejects before CUDA use.
  /// \throws CollectionError with resource_budget for invalid/overflow geometry
  /// or admission, metadata_unavailable for registration, allocation for setup.
  static std::shared_ptr<DeviceChannel> create(std::shared_ptr<MemoryBudget> host, std::shared_ptr<MemoryBudget> device,
                                               BatchSource source, BatchInfo metadata, int device_index,
                                               std::size_t slots, std::size_t batches, std::size_t events_per_batch,
                                               std::optional<FabricRecordingConfig> fabric = std::nullopt,
                                               std::size_t transport_capacity = 0);
  /// Admit/construct all Host control, delivery and pinned/Device reservations
  /// without any CUDA call. The process can attach Writer/reconstruction owners
  /// and check their coexistence peak before activate() creates Device storage.
  static std::shared_ptr<DeviceChannel>
  prepare(std::shared_ptr<MemoryBudget> host, std::shared_ptr<MemoryBudget> device, BatchSource source,
          BatchInfo metadata, int device_index, std::size_t slots, std::size_t batches, std::size_t events_per_batch,
          std::optional<FabricRecordingConfig> fabric = std::nullopt, std::size_t transport_capacity = 0);
  /// Initialize one prepared channel once, before service/production launches.
  /// Failure retains uncertain CUDA ownership; the same object cannot retry.
  void activate();
  /// Host control/slab retirement; active/uncertain CUDA resources remain owned
  /// by the self-retainer and cannot reach this destructor prematurely.
  ~DeviceChannel();
  DeviceChannel(const DeviceChannel &) = delete;
  DeviceChannel &operator=(const DeviceChannel &) = delete;
  /// Initialized descriptor. Runtime retains a shared channel until all kernels
  /// and Graph references using it have actually retired.
  BufferView buffer_view() const noexcept { return view_; }
  /// Optional Fabric tables share this exact Source and its retirement owner.
  /// An empty result means this channel has no Fabric recording storage.
  FabricRecordingView fabric_view() const noexcept { return fabric_view_; }
  /// Fixed Transport current-state table on this same Source and owner.
  TransportRecordingView transport_view() const noexcept { return transport_view_; }
  /// Serialize collection/Writer lanes independently over this Host handle.
  Collector &collector() noexcept { return *collector_; }
  /// Physical CUDA ordinal chosen by the owning runtime.
  int device_index() const noexcept { return device_index_; }
  /// Progress on the collection lane. producer_retired proves all production
  /// Device calls, kernels and Graph references have actually retired at the
  /// original runtime boundary. Stop/timeouts alone never establish that proof.
  /// Free CUDA resources only after an empty final pass and its completion.
  /// \throws CollectionError; uncertain failures retain resources and budgets.
  /// Later ticks reject before CUDA submission or cleanup retry, including a
  /// failed Device selection outside the transfer driver's own state machine.
  bool tick(bool producer_retired, MetadataResolver resolver = {}, ReconstructionSink reconstruction = {});
  /// Collection-lane request and completion for the original runtime's
  /// temporary producer-quiescence boundary. Completion includes a fresh empty
  /// scan and actual reclamation; it does not retire the Source or its storage.
  void request_drain(std::uint64_t identity) noexcept { transfer_->request_drain(identity); }
  /// Read only on the collection lane, after progressing tick().
  bool drained(std::uint64_t identity) const noexcept { return transfer_->drained(identity); }
  /// Request asynchronous admission closure at the next collection tick.
  void stop() noexcept { stopping_.store(true, std::memory_order_release); }
  /// Collection-lane completion polling, independent of the next source scan.
  bool in_flight() const noexcept { return transfer_ && transfer_->in_flight(); }
  /// Whether CUDA storage retirement completed, independent of Writer files.
  bool retired() const noexcept { return retired_.load(std::memory_order_acquire); }
  /// Serialized lifecycle-owner query; once true, cold cancellation is unsafe.
  /// Failed activation still counts as an attempt until its resources retire.
  bool activation_attempted() const noexcept { return activation_attempted_; }
  /// Allocator-only constructor; the private key enforces create().
  DeviceChannel(ConstructionKey, std::shared_ptr<MemoryBudget> host, std::shared_ptr<MemoryBudget> device,
                BatchSource source, BatchInfo metadata, int device_index, SourceLayout source_layout,
                BatchPoolLayout batch_layout, TransferLayout transfer_layout, std::size_t slots, std::size_t batches,
                std::size_t events_per_batch, std::optional<FabricRecordingConfig> fabric,
                std::size_t transport_capacity);

private:
  friend class CollectionService;
  void initialize();
  void check(cudaError_t status, CollectionErrorCode code, const char *operation);
  bool release_storage() noexcept;
  const std::shared_ptr<MemoryBudget> host_;
  const std::shared_ptr<MemoryBudget> device_budget_;
  const BatchSource source_;
  const BatchInfo metadata_;
  const int device_index_;
  const SourceLayout source_layout_;
  const BatchPoolLayout batch_layout_;
  const TransferLayout transfer_layout_;
  const std::optional<FabricRecordingConfig> fabric_config_;
  const std::optional<FabricRecordingLayout> fabric_layout_;
  const std::size_t transport_capacity_;
  const std::size_t transport_bytes_;
  TransportRecordingView transport_view_{};
  std::byte *transport_storage_ = nullptr;
  MemoryBudget::Reservation transport_charge_;
  BufferView view_{};
  FabricRecordingView fabric_view_{};
  std::byte *fabric_storage_ = nullptr;
  MemoryBudget::Reservation fabric_charge_;
  std::byte *source_storage_ = nullptr;
  std::byte *device_frame_ = nullptr;
  std::byte *host_frame_ = nullptr;
  std::byte *batch_storage_ = nullptr;
  MemoryBudget::Reservation source_charge_;
  MemoryBudget::Reservation frame_charge_;
  MemoryBudget::Reservation pinned_charge_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t copied_ = nullptr;
  cudaEvent_t reclaimed_ = nullptr;
  std::optional<HostBatchPool> batches_;
  std::optional<Collector> collector_;
  std::optional<DeviceTransfer> transfer_;
  std::shared_ptr<DeviceChannel> retained_;
  std::atomic<bool> stopping_{false};
  std::atomic<bool> retired_{false};
  bool cuda_idle_ = true;
  bool activation_attempted_ = false;
  bool initialized_ = false;
  bool transfer_uncertain_ = false;
  cudaError_t cleanup_error_ = cudaSuccess;
  const char *cleanup_operation_ = nullptr;
};

} // namespace xpool::devkit::timeline
