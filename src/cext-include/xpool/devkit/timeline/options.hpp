#pragma once

/// \file xpool/devkit/timeline/options.hpp
/// \brief C0 preparation policy and checked, explicitly complete Host demands.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>

namespace xpool::devkit::timeline {

/// Actual owner-declared Host allocation and peak-read requirements, in bytes.
/// Each field must be supplied, including explicit zero for an inactive owner.
/// Unknown sizes remain null and reject admission. Totals describe distinct
/// physical storage, including rounding; inclusive pool totals are not added.
struct HostRequirements {
  /// Complete Host Event-pool layouts, including their own control/rounding.
  std::optional<std::size_t> events;
  /// Persistent control storage not already charged in another allocation.
  std::optional<std::size_t> control;
  /// Pinned Host receive frames and associated Host-only state.
  std::optional<std::size_t> transfer;
  /// Complete delivery pool, including batches currently held by Writer.
  std::optional<std::size_t> delivery;
  /// Writer-owned encoding scratch and persistent publication state.
  std::optional<std::size_t> writer;
  /// Bounded registered domains, names and exact metadata bytes.
  std::optional<std::size_t> metadata;
  /// Explicit worker stacks and guards, excluding already counted control.
  std::optional<std::size_t> threads;
  /// All coexisting legacy reconstruction slabs.
  std::optional<std::size_t> legacy;
  /// Canonical retained provenance, excluding report read copies.
  std::optional<std::size_t> provenance;
  /// Peak retained Native/binding snapshot and report materializations.
  std::optional<std::size_t> read_staging;
  /// Peak serialization workspace, distinct from Writer's encoding allocation.
  std::optional<std::size_t> serialization;

  /// Sum all declared requirements, rejecting missing fields and size overflow.
  /// \throws CollectionError resource_budget with the requesting Producer.
  std::size_t required_bytes(std::uint64_t producer = 0) const;
  /// Check a complete declaration against one shared process Host limit.
  /// This query allocates nothing and changes no live budget reservations.
  /// \throws CollectionError resource_budget, including incomplete declarations.
  std::size_t admit(std::size_t limit, std::uint64_t producer = 0) const;
};

/// One original legacy Record family, independent of Event wire values.
enum class LegacyKind { Transport, Fabric };

/// Distinct persistent and peak-read parts of one legacy admission demand.
struct LegacyMemorySizes {
  /// Persistent C0 reconstruction slots across all live domains.
  std::size_t slabs = 0;
  /// Persistent finite canonical provenance declarations.
  std::size_t provenance = 0;
  /// Simultaneously retained Record vectors and report values.
  std::size_t read_staging = 0;
};

/// Coexisting domains and retained runtime read windows, not a public reader
/// limit. Owners declare their actual provenance/report types; unimplemented
/// types cannot be passed as zero. Caller-owned copies must not be silently
/// excluded when deciding which windows the runtime still owns.
struct LegacyMemoryDemand {
  /// Family whose existing Native Record/vector declaration determines staging.
  LegacyKind kind = LegacyKind::Transport;
  /// Simultaneously retained Endpoint or Producer/Generation domains.
  std::size_t domains = 0;
  /// Earliest-N operations retained independently in each domain.
  std::size_t capacity = 8192;
  /// Declared runtime-held read peak in every domain, including older cutoffs.
  std::size_t readers_per_domain = 1;
  /// Actual persistent provenance declaration per eligible operation.
  std::size_t provenance_bytes_per_operation = 0;
  /// Actual report read declaration per eligible operation.
  std::size_t report_bytes_per_operation = 0;
  /// Actual fixed report read storage, excluding the Record vector.
  std::size_t report_fixed_bytes = 0;

  /// Compute C0 256/384-byte slabs and actual Native Record/vector read sizes,
  /// including all declared report/provenance storage. Common domain metadata
  /// and serialization scratch belong separately in HostRequirements.
  /// \throws CollectionError resource_budget for missing/overflowing demands.
  LegacyMemorySizes sizes(std::uint64_t producer = 0) const;
};

/// Native preparation projection of the resolved debug.timeline policy.
/// Validation is schema/known-minimum checking, not proof that all providers or
/// physical allocations are installed. Native code does not resolve env vars.
struct Options {
  /// Request complete unified collection; providers must also be prepared.
  bool enable = false;
  /// Unified artifact root, absent when unified mode is disabled.
  std::optional<std::filesystem::path> outdir = std::nullopt;
  /// Collection byte limit per process/Device, including every active allocation.
  std::size_t device_bytes = 8388608;
  /// Process-wide Host collection byte limit across observers and readers.
  std::size_t host_bytes = 33554432;
  /// Maximum complete chunk size including its 128-byte header.
  std::size_t chunk_bytes = 1048576;
  /// Background collection cadence, in milliseconds.
  std::uint64_t collection_interval_ms = 100;
  /// Process-wide runtime disk byte limit across directories.
  std::size_t disk_bytes = 1073741824;
  /// Protected metadata/close allowance within the disk limit.
  std::size_t metadata_reserve_bytes = 8388608;
  /// Additional observation wait in milliseconds, within the production deadline.
  std::uint64_t shutdown_wait_ms = 5000;
  /// Fixed C0 v4 block length of ten sampling attempts.
  std::size_t calibration_block_attempts = 10;
  /// Immutable Event slots per Device Producer.
  std::size_t device_event_slot_capacity = 16384;
  /// Maximum simultaneously open Operation IPC regions.
  std::size_t endpoint_capacity = 8;
  /// Process-wide dynamic Context slots in the actual Graph owner.
  std::size_t context_slot_capacity = 64;
  /// Raw Device-clock sample slots in this process.
  std::size_t sampler_slot_capacity = 1;

  /// Check pairing, positive widths, v4 bounds and the known unified minimum.
  /// Outdir is an absolute resolved path supplied by xpool.config.
  /// \throws CollectionError resource_budget; performs no I/O/CUDA/allocation.
  void validate() const;
  /// Fixed C0 full-category logical lower bound before per-allocation rounding,
  /// transfer scratch or other implementation storage. Never full admission.
  /// \throws CollectionError resource_budget on size overflow.
  std::size_t logical_device_minimum() const;
  /// Validate both participants and reject any differing immutable policy.
  /// \throws CollectionError resource_budget on invalid or inconsistent policy.
  void require_same_policy(const Options &peer) const;
  bool operator==(const Options &) const = default;
};

} // namespace xpool::devkit::timeline
