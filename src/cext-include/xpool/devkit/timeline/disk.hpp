#pragma once

/// \file xpool/devkit/timeline/disk.hpp
/// \brief Process-wide disk admission including retained failure evidence.

#include <cstdint>
#include <mutex>

#include <xpool/devkit/timeline/error.hpp>

namespace xpool::devkit::timeline {

/// Metadata and close summaries can use the protected reserve; Event and
/// attachment files must leave it available. One ledger covers all directories.
enum class DiskKind { Data, Metadata };

/// Exact known file extents and outstanding admission in one live process.
struct DiskReport {
  /// Immutable process limit, including temporary and metadata files.
  std::uint64_t limit;
  /// Protected metadata/close allowance, part of the limit.
  std::uint64_t metadata_reserve;
  /// Retained known bytes, including failed temporary files.
  std::uint64_t used;
  /// Admitted bytes not yet finalized to actual retained extents.
  std::uint64_t reserved;
  /// Largest simultaneous used plus reserved total.
  std::uint64_t peak;
};

/// Own one ledger per process and share it across every collection Writer and
/// attachment producer. Existing artifacts are charged with their actual size
/// during admission; files are never deleted to recover quota.
class DiskBudget {
public:
  /// \throws CollectionError with resource_budget for invalid limits.
  DiskBudget(std::uint64_t limit, std::uint64_t metadata_reserve);
  DiskBudget(const DiskBudget &) = delete;
  DiskBudget &operator=(const DiskBudget &) = delete;

  class Reservation;

  /// Exclusive accounting token for a surviving artifact. Destruction keeps
  /// its disk charge: losing an owner never proves the file was removed. Only
  /// a proven atomic replacement may retire the superseded file's bytes.
  class Extent {
  public:
    /// No previously published file.
    Extent() = default;
    Extent(const Extent &) = delete;
    Extent &operator=(const Extent &) = delete;
    /// Transfer accounting authority; bytes remain charged throughout.
    Extent(Extent &&other) noexcept;
    /// Transfer authority; any abandoned old charge remains conservative.
    Extent &operator=(Extent &&other) noexcept;
    /// Known charged bytes of this file.
    std::uint64_t bytes() const noexcept { return bytes_; }
    /// After successful atomic rename over this exact owned file, replace its
    /// token and retire only the superseded extent. New bytes were separately
    /// admitted while old bytes still existed. Returns false without reducing
    /// either ledger on a mismatched/empty replacement token.
    bool replaced_by(Extent &&replacement) noexcept;

  private:
    friend class Reservation;
    Extent(DiskBudget &budget, std::uint64_t bytes) : budget_(&budget), bytes_(bytes) {}
    DiskBudget *budget_ = nullptr;
    std::uint64_t bytes_ = 0;
  };

  /// One move-only pre-creation admission. The ledger outlives it. On failure,
  /// retain() records the actual surviving temporary extent before destruction.
  class Reservation {
  public:
    /// Empty admission.
    Reservation() = default;
    /// Return unused admission; a file owner retains surviving bytes first.
    ~Reservation();
    Reservation(const Reservation &) = delete;
    Reservation &operator=(const Reservation &) = delete;
    /// Transfer outstanding admission once.
    Reservation(Reservation &&other) noexcept;
    /// Return this unused admission and take another.
    Reservation &operator=(Reservation &&other) noexcept;
    /// Finalize actual extents once; retained bytes may not exceed admission.
    /// \throws CollectionError with resource_budget for an invalid extent.
    void retain(std::uint64_t bytes);
    /// Finalize a surviving extent and return exclusive replacement authority.
    /// Failure temporaries can use retain() when they will never be replaced.
    /// \throws CollectionError for an extent beyond this admission.
    Extent retain_extent(std::uint64_t bytes);

  private:
    friend class DiskBudget;
    Reservation(DiskBudget &budget, std::uint64_t bytes) noexcept;
    void cancel() noexcept;
    DiskBudget *budget_ = nullptr;
    std::uint64_t bytes_ = 0;
  };

  /// Admit exact eventual bytes before create/extend. Failure changes no totals.
  /// \throws CollectionError with resource_budget; callers set disk_quota using
  /// this typed category, independently of I/O errors.
  Reservation reserve(std::uint64_t bytes, DiskKind kind, std::uint64_t producer = 0);
  /// Register preexisting extents once before any new files use this ledger.
  /// \throws CollectionError with resource_budget if the total exceeds quota.
  void charge_existing(std::uint64_t bytes);
  /// Consistent report while independent Writers admit files concurrently.
  DiskReport report() const;

private:
  const std::uint64_t limit_;
  const std::uint64_t metadata_reserve_;
  std::uint64_t used_ = 0;
  std::uint64_t reserved_ = 0;
  std::uint64_t peak_ = 0;
  mutable std::mutex mutex_;
};

} // namespace xpool::devkit::timeline
