#pragma once

/// \file xpool/devkit/timeline/policy.hpp
/// \brief Immutable Host policy installed before Native initialization.

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>

#include <xpool/debug/options.hpp>
#include <xpool/devkit/timeline/options.hpp>

namespace xpool::devkit::timeline {

/// Numeric copy of the authoritative Options; filesystem ownership stays with
/// xpool.config and later budgeted Source/Writer registration.
struct PolicyLimits {
  /// Process/Device allocation limit.
  std::uint64_t device_bytes;
  /// Complete process collection allocation limit.
  std::uint64_t host_bytes;
  /// Complete chunk size, including header.
  std::uint64_t chunk_bytes;
  /// Collection cadence.
  std::uint64_t collection_interval_ms;
  /// Process-wide artifact quota.
  std::uint64_t disk_bytes;
  /// Protected metadata/close allowance within the quota.
  std::uint64_t metadata_reserve_bytes;
  /// Extra wait within the remaining production deadline.
  std::uint64_t shutdown_wait_ms;
  /// Fixed calibration attempts per block.
  std::uint64_t calibration_block_attempts;
  /// Immutable Device Event slots.
  std::uint64_t device_event_slot_capacity;
  /// Simultaneously open IPC identity regions.
  std::uint64_t endpoint_capacity;
  /// Graph-owner Context slots.
  std::uint64_t context_slot_capacity;
  /// Raw Device-clock sampler slots.
  std::uint64_t sampler_slot_capacity;
  bool operator==(const PolicyLimits &) const = default;
};

/// Fixed-size immutable projection; returned copies cannot alter installed state.
struct CollectionPolicy {
  /// Authoritative validated numeric geometry.
  PolicyLimits limits;
  /// Original legacy getter/view flags and capacities.
  xpool::debug::Options legacy;
  /// Unified raw collection enables all four Event classes.
  bool unified;
  /// An enabled consumer explicitly requested shared Transport identity.
  bool explicit_transport_identity_consumer;
  /// Fingerprint of the complete resolved debug config, including legacy paths.
  std::array<std::byte, 32> configuration_fingerprint;
  /// Native hash of Options.outdir, avoiding a retained unbudgeted path copy.
  std::optional<std::array<std::byte, 32>> unified_outdir_fingerprint;
  /// Whether this process requires any observation resources.
  bool enabled() const noexcept;
  /// Raw class enablement; legacy snapshots still use legacy's own flags.
  bool transport_events() const noexcept { return unified || legacy.transport_observer.enable; }
  /// Raw Fabric enablement.
  bool fabric_events() const noexcept { return unified || legacy.fabric_observer.enable; }
  /// Raw Graph enablement.
  bool graph_events() const noexcept { return unified || legacy.graph_observer.enable; }
  /// Raw routing enablement.
  bool routing_events() const noexcept { return unified || legacy.ffn_routing_observer.enable; }
  /// C0 immutable shared identity policy for both Transport participants.
  bool transport_identity_required() const noexcept;
  bool operator==(const CollectionPolicy &) const = default;
};

/// Prepare an immutable value projection without installing process policy.
/// Actual production registration and capabilities remain with their owner.
CollectionPolicy make_collection_policy(const Options &options, const xpool::debug::Options &legacy,
                                        std::array<std::byte, 32> fingerprint = {}, bool identity_consumer = false);

} // namespace xpool::devkit::timeline
