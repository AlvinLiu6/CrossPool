#pragma once

/// \file xpool/transport/atnagent.hpp
/// \brief Process-wide AtnAgent Transport Resident and arena lifecycle.

#include <cstddef>
#include <map>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include <c10/core/Device.h>
#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <xpool/fabric/arena.hpp>
#include <xpool/transport/arena.hpp>

namespace xpool::transport {

/// Launch one cooperative Transport Resident over an immutable arena array.
/// \pre arenas is non-empty and state and stream are non-null.
/// \pre observations is empty when disabled; otherwise its entries match
/// arenas in count and order, and their owners outlive the Resident.
/// \pre Cooperative launch is supported and the complete grid is concurrently
/// resident; capability and occupancy failures occur before launch.
/// \throws c10::Error when capability, occupancy, or launch checks fail.
void launch_resident_kernel(cuda::std::span<const ArenaView> arenas, xpool::fabric::ArenaView fabric_arena,
                            ResidentState *state, cudaStream_t stream,
                            cuda::std::span<const ObservationView> observations = {});

/// Actual aligned Device allocations of an observed Resident: Arena pointers,
/// drain state and the private observation table. Preparation adds this once
/// to the Region pool demand before starting CUDA. This query performs no CUDA.
/// \throws timeline::CollectionError resource_budget on zero/overflow.
std::size_t resident_observation_device_bytes(std::size_t arena_count);

/// Process-lifetime owner of all Transport arenas and one AtnAgent Resident.
///
/// Arena creation and preactivation rollback remain incremental. Activation
/// freezes the complete arena set and launches exactly one cooperative block
/// per arena. Health and drain are process-wide and never accept handles.
/// Public operations enforce this lifecycle and surface CUDA failures as
/// c10::Error.
class AtnAgentRuntime {
public:
  /// Return the sole process-lifetime AtnAgent Transport runtime.
  static AtnAgentRuntime &singleton() {
    static AtnAgentRuntime runtime;
    return runtime;
  }

  AtnAgentRuntime(const AtnAgentRuntime &) = delete;
  AtnAgentRuntime &operator=(const AtnAgentRuntime &) = delete;
  AtnAgentRuntime(AtnAgentRuntime &&) = delete;
  AtnAgentRuntime &operator=(AtnAgentRuntime &&) = delete;

  /// Allocate and register one AtnAgent-owned Transport arena.
  /// The returned CUDA IPC handle remains valid until destroy_arenas().
  ArenaHandle create_arena(c10::DeviceIndex device, std::size_t instance_index, std::size_t instance_rank,
                           std::size_t payload_row_capacity, std::size_t hidden_size, c10::ScalarType payload_dtype,
                           std::size_t atn_tp_rank, std::size_t atn_tp_size, std::size_t atn_dp_rank,
                           std::size_t atn_dp_size, std::shared_ptr<ObservationPool> observation_pool = {},
                           std::uint64_t generation = 0);
  /// Return the immutable observation descriptor belonging to this exact Arena.
  /// Plain Arenas return null; unknown handles reject. No live Device read.
  std::optional<ObservationRegion> observation_region(const ArenaHandle &handle) const;

  /// Freeze the arena set and launch its sole cooperative Resident.
  ///
  /// This method returns only after every mailbox is acquire-observed as Idle.
  void activate();

  /// Reject premature Resident completion or CUDA stream failure.
  ///
  /// Completion caused by a retained canonical generation failure is expected;
  /// completion with neither that fact nor a host drain request is an error.
  void check_health() const;

  /// Publish the process-local Resident drain command without synchronizing.
  /// Repeated calls during or after completed drain are no-ops.
  void drain_async();

  /// Poll process-wide Resident completion and release its device resources.
  /// Returns true while the Resident or its control write remains pending.
  bool drain_pending();

  /// Destroy known arenas during preactivation rollback or after Resident drain.
  /// \throws c10::Error after attempting every requested cleanup when any
  /// release fails.
  /// Observed allocations also require the caller's confirmed closed lease
  /// admission and actual peer detach. Missing proof retains every allocation.
  void destroy_arenas(std::span<const ArenaHandle> handles, bool peer_mappings_detached = false);

private:
  class Resident : public std::enable_shared_from_this<Resident> {
  public:
    // Host-only preparation; install the stable owner before initialize().
    Resident(c10::DeviceIndex device, const std::map<ArenaHandle, Arena> &arenas,
             std::shared_ptr<ObservationPool> pool);
    ~Resident() = default;

    Resident(const Resident &) = delete;
    Resident &operator=(const Resident &) = delete;
    Resident(Resident &&) = delete;
    Resident &operator=(Resident &&) = delete;

    bool pending() const;
    bool host_drain_requested() const { return host_drain_requested_; }
    bool released() const { return released_; }
    void initialize(xpool::fabric::ArenaView fabric_arena);
    void quarantine() const;
    void request_drain();
    void release();

  private:
    c10::DeviceIndex device_;
    std::shared_ptr<ObservationPool> pool_;
    // Installed before CUDA; only proven release breaks this retention.
    std::shared_ptr<Resident> retained_;
    xpool::devkit::timeline::MemoryBudget::Reservation device_control_;
    std::pmr::vector<ArenaView> host_arenas_;
    std::pmr::vector<ObservationView> host_observations_;
    ArenaView *arenas_ = nullptr;
    ObservationView *observations_ = nullptr;
    ResidentState *state_ = nullptr;
    ResidentState initial_state_{};
    std::size_t arena_bytes_ = 0;
    std::size_t state_bytes_ = 0;
    std::size_t observation_bytes_ = 0;
    // Raw handles stay in this stable owner even if creation reports an error.
    cudaStream_t control_stream_ = nullptr;
    cudaStream_t resident_stream_ = nullptr;
    bool host_drain_requested_ = false;
    bool initialized_ = false;
    bool released_ = false;
    mutable bool uncertain_ = false;
  };

  AtnAgentRuntime() = default;
  ~AtnAgentRuntime() = default;

  bool generation_failed() const;
  Arena &arena(const ArenaHandle &handle);
  const Arena &arena(const ArenaHandle &handle) const;

  mutable std::mutex mutex_;
  std::optional<c10::DeviceIndex> device_;
  std::map<ArenaHandle, Arena> arenas_;
  Arena pending_arena_;
  std::shared_ptr<xpool::devkit::timeline::MemoryBudget> pending_host_;
  xpool::devkit::timeline::MemoryBudget::Reservation pending_control_;
  std::shared_ptr<Resident> resident_;
};

} // namespace xpool::transport
