#pragma once

/// \file xpool/devkit/timeline/identity.hpp
/// \brief Bounded launcher registration of immutable observation identities.

#include <array>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <xpool/devkit/timeline/batch.hpp>
#include <xpool/devkit/timeline/counter.hpp>
#include <xpool/devkit/timeline/resource.hpp>
#include <xpool/runtime.hpp>

namespace xpool::devkit::timeline {

/// UUID bytes in canonical UUID byte order, never CUDA IPC handle bytes.
using IdentityUuid = std::array<std::byte, 16>;

/// Decode canonical lowercase UUID text without changing its UUID byte order.
/// \throws std::invalid_argument for other lengths, grouping or hex spelling.
IdentityUuid parse_identity_uuid(std::string_view text);
/// Produce exactly 36 canonical UUID characters in caller-owned fixed storage.
std::array<char, 36> format_identity_uuid(const IdentityUuid &bytes) noexcept;

/// Actual process facts supplied by the existing launch/registration owner.
/// Creation UUID is fresh per process creation; pid is diagnostic, not identity.
struct ProcessIdentity {
  /// Launcher UUID unique to this actual process creation.
  IdentityUuid creation_id{};
  /// Machine-creation UUID defining the Host monotonic clock lifetime.
  IdentityUuid host_creation_id{};
  /// Positive diagnostic OS PID; never the identity directory key.
  std::uint64_t pid = 0;
  /// Actual Native process role, separate from each Event's observation role.
  xpool::RuntimeRole runtime_role = xpool::RuntimeRole::Daemon;
  /// Compare the complete immutable process declaration.
  bool operator==(const ProcessIdentity &) const = default;
};

/// One process's unique Host Producer under its actual Host clock domain.
struct RegisteredProcess {
  /// Retained launch facts; actual process liveness remains with the launcher.
  ProcessIdentity process;
  /// Run, unique Host Producer and its registered monotonic clock.
  BatchSource host;
  /// Launcher-only exact OS creation key; zero in imported metadata values.
  /// This lookup fact is never an identity or a C0 metadata field.
  std::uint64_t start_key = 0;
};

/// One machine-creation Host monotonic domain shared by its processes.
struct RegisteredHostClock {
  /// Positive Run-local metadata reference.
  std::uint64_t id;
  /// Actual machine-creation UUID, shared across its registered processes.
  IdentityUuid host_creation_id;
};

/// A Device Producer's fixed clock identity. Physical mapping and epoch are
/// copied from actual preparation facts, not inferred from a device index.
struct RegisteredDevice {
  /// Creation identity of its previously registered owning process.
  IdentityUuid process_creation_id;
  /// Run, unique Device Producer and its registered raw clock domain.
  BatchSource source;
  /// Zero-based mapping inside the owning process's actual CUDA visibility.
  std::uint32_t logical_device;
  /// Positive lifetime of the prepared raw Device clock.
  std::uint64_t raw_epoch;
  /// Actual physical mapping string; immutable and charged to the Host ledger.
  std::pmr::string physical_device_uuid;
};

/// Positive metadata reference to the exact production Generation string.
struct RegisteredGeneration {
  /// Positive Run-local reference, independent of Producer and clock counters.
  std::uint64_t id;
  /// Exact production identity, with Host-owned charged string storage.
  std::pmr::string fabric_generation;
};

class IdentityDirectory;

/// Stable immutable dictionary prefix. New registration may append disjoint
/// entries, while these spans and their strings remain unchanged. Retain this
/// value through serialization; it keeps the directory and Host budget alive.
struct IdentityDefinitions {
  /// Keeps every referenced definition, string and Host budget alive.
  std::shared_ptr<const IdentityDirectory> owner;
  /// Complete process/Host-Producer prefix at capture.
  std::span<const RegisteredProcess> processes;
  /// All Host domains referenced by that process prefix.
  std::span<const RegisteredHostClock> host_clocks;
  /// Complete Device-Producer/clock prefix at capture.
  std::span<const RegisteredDevice> devices;
  /// Production Generation references defined by that prefix.
  std::span<const RegisteredGeneration> generations;
};

/// Launcher-owned bounded identity directory for one Run. All registrations
/// are cold preparation calls before related Capture/Event use. Definitions
/// stay immutable for the Run; retired processes/Generations are never evicted
/// to reuse IDs. Capacity exhaustion rejects new preparation explicitly.
/// This directory establishes identity, not metadata JSON schema, topology,
/// actual process liveness, CUDA mapping ownership or peer-detach proof.
class IdentityDirectory : public std::enable_shared_from_this<IdentityDirectory> {
  class ConstructionKey {
    friend class IdentityDirectory;
    ConstructionKey() = default;
  };

public:
  /// Preallocate bounded definition tables under the launcher's Host ledger.
  /// Strings have their actual cold-registration allocations charged before
  /// publication. Run/creation UUIDs come from the launcher, never a PID.
  /// \throws CollectionError for zero capacities, overflow or failed admission.
  static std::shared_ptr<IdentityDirectory> create(std::shared_ptr<MemoryBudget> host, IdentityUuid run,
                                                   std::size_t processes = 64, std::size_t devices = 64,
                                                   std::size_t generations = 64);
  IdentityDirectory(const IdentityDirectory &) = delete;
  IdentityDirectory &operator=(const IdentityDirectory &) = delete;
  /// Assign a fresh random process-creation UUID once per actual OS creation.
  /// The launcher verifies liveness and supplies the exact nonzero creation
  /// key (for example, the bits of psutil's retained create_time). Repeating
  /// the same Host/PID/key/role returns the prior assignment. PID reuse with
  /// a different key allocates a new UUID and Host Producer. Randomness failure
  /// rejects preparation before publishing any definition.
  /// \throws CollectionError for invalid facts, conflicts or failed admission.
  RegisteredProcess assign_process(IdentityUuid host_creation, std::uint64_t pid, std::uint64_t start_key,
                                   xpool::RuntimeRole role);
  /// Register a process/Device Source after its process registration. Repeated
  /// identical mapping/epoch returns the same source; changed facts cannot
  /// alter a Producer's clock. Runtime context replacement needs a fresh
  /// supported preparation lifetime, not mutation of this definition.
  /// \throws CollectionError for missing process, invalid/conflicting facts,
  /// capacity, identity exhaustion or failed Host string admission.
  BatchSource register_device(IdentityUuid process, std::uint32_t logical_device, std::string_view physical_device_uuid,
                              std::uint64_t raw_epoch);
  /// Intern the exact production Generation string independently of Producer
  /// and clock counters; topology definitions remain with metadata preparation.
  /// \throws CollectionError for empty string, capacity or identity exhaustion.
  std::uint64_t register_generation(std::string_view fabric_generation);
  /// Capture a complete registered prefix under the cold mutex. No allocation
  /// occurs; later append cannot invalidate these fixed-capacity spans.
  IdentityDefinitions definitions() const;
  /// Close new identity admission; existing identical definitions remain valid.
  void stop();
  /// Whether a positive u64 namespace exhausted and closed new preparation.
  bool identity_exhausted() const;
  /// Fixed Run bytes shared by every source in this directory.
  const IdentityUuid &run_id() const noexcept { return run_; }
  /// Allocator-only constructor; use create() for complete control admission.
  IdentityDirectory(ConstructionKey, std::shared_ptr<MemoryBudget> host, IdentityUuid run, std::size_t processes,
                    std::size_t devices, std::size_t generations);

private:
  void require_admission() const;
  std::uint64_t next(AttemptSequence &sequence);
  const std::shared_ptr<MemoryBudget> host_;
  const IdentityUuid run_;
  BudgetMemoryResource resource_;
  std::pmr::vector<RegisteredProcess> processes_;
  std::pmr::vector<RegisteredHostClock> host_clocks_;
  std::pmr::vector<RegisteredDevice> devices_;
  std::pmr::vector<RegisteredGeneration> generations_;
  const std::size_t process_capacity_;
  const std::size_t device_capacity_;
  const std::size_t generation_capacity_;
  AttemptSequence producers_{};
  AttemptSequence clocks_{};
  AttemptSequence generation_ids_{};
  mutable std::mutex mutex_;
  bool stopped_ = false;
  bool exhausted_ = false;
};

} // namespace xpool::devkit::timeline
