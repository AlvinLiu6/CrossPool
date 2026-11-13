#pragma once

/// \file xpool/devkit/timeline/graphlog.hpp
/// \brief Bounded legacy Host Graph records on the process collection workers.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <optional>
#include <string>
#include <string_view>

#include <xpool/devkit/timeline/counter.hpp>
#include <xpool/devkit/timeline/disk.hpp>
#include <xpool/devkit/timeline/json.hpp>
#include <xpool/devkit/timeline/resource.hpp>

namespace xpool::devkit::timeline {

/// Private compatibility-log outcomes, independent of C0 Event counters.
/// Values are exact after producer, collection and Writer calls have retired;
/// live values are cumulative lower bounds. Saturation is explicitly reported.
struct GraphLogReport {
  /// Calls admitted before closure, including full and invalid-input outcomes.
  std::uint64_t attempted = 0;
  /// Complete records published to producer slots.
  std::uint64_t accepted = 0;
  /// The observed queue head still belonged to an earlier occupied position.
  std::uint64_t dropped_full = 0;
  /// The observed head advanced or its one claim lost to another producer.
  std::uint64_t dropped_contention = 0;
  /// Invalid phase/kind/backend inputs omitted as complete records.
  std::uint64_t dropped_input = 0;
  /// Complete records copied to bounded Writer delivery frames.
  std::uint64_t copied = 0;
  /// Complete JSONL lines accepted by successful file write calls.
  std::uint64_t written = 0;
  /// Copied records omitted after an explicit Writer failure.
  std::uint64_t lost = 0;
  /// A cumulative counter saturated or the private queue positions exhausted.
  bool counter_exhausted = false;
  /// Producer admission has stopped; previously admitted calls may retire.
  bool closed = false;
  /// All accepted work and the summary attempt retired on the Writer lane.
  bool finished = false;
  /// First file/quota failure, even when resource retirement finished.
  std::optional<CollectionErrorCode> error = std::nullopt;
};

/// One process's original SGLang Graph JSONL stream. All retained controls,
/// producer slots, delivery frames and paths share the process Host ledger.
/// Producers call record(); exactly one collection lane calls collect();
/// exactly one Writer calls write_one()/finish(). Those lanes may overlap.
/// No method creates a worker or initializes CUDA.
class GraphLogStream {
  class ConstructionKey {
    friend class GraphLogStream;
    ConstructionKey() = default;
  };

public:
  /// Admit fixed storage before service. path is an absolute process-owned
  /// legacy destination; pid names its actual process. producer is a registered
  /// Host Source when present, or zero in compatibility-only mode.
  /// Output creation happens on the Writer lane with O_EXCL, preserving any
  /// preexisting file. Backend labels have a 127-byte UTF-8 payload limit.
  /// At least two producer slots distinguish free and ready sequence labels.
  /// \throws CollectionError for geometry, identity or Host admission.
  static std::shared_ptr<GraphLogStream> create(std::shared_ptr<MemoryBudget> host, std::shared_ptr<DiskBudget> disk,
                                                std::string_view path, std::uint64_t producer, std::uint64_t pid,
                                                std::size_t slots = 256);
  ~GraphLogStream();
  GraphLogStream(const GraphLogStream &) = delete;
  GraphLogStream &operator=(const GraphLogStream &) = delete;
  /// Copy one original lifecycle record with one bounded queue-head CAS and no
  /// retry. phase is decode=1/prefill=2; kind is capture begin/end/error=1/2/3
  /// or execute begin/end/error=4/5/6. raw_time_ns is CLOCK_MONOTONIC nanoseconds.
  /// Full, a lost concurrent head claim or invalid input drops the new record
  /// with its own counter and returns false; closed
  /// admission returns false without starting another attempt. No allocation,
  /// formatting, file operation, mutex wait or CUDA operation occurs.
  bool record(std::uint32_t phase, std::uint32_t kind, std::string_view backend, std::uint64_t raw_time_ns) noexcept;
  /// Close admission. Existing calls may finish; their actual retirement is
  /// established separately by the process lifecycle before final collection.
  void stop() noexcept;
  /// Copy the ready FIFO prefix into one bounded frame, releasing each source
  /// slot after its whole Host copy. A pending earlier producer or busy delivery
  /// pool returns promptly while retaining later records in order.
  std::size_t collect() noexcept;
  /// Whether every admitted queue position has been copied. Exact only after
  /// all producer calls retire; pending delivery frames are independent.
  bool producer_empty() const noexcept;
  /// Append at most one collected frame on the existing Writer lane. Exact
  /// disk admission precedes extension. Failures stop admission, preserve the
  /// actual file prefix and account remaining accepted records as lost.
  /// Returns whether a frame was consumed, including loss after failure.
  bool write_one() noexcept;
  /// Writer-only finalization after actual producer retirement and collection
  /// completion. Returns false while delivery frames remain. Closes the file
  /// and attempts a separate .quality.json summary from protected disk quota.
  /// finished means all work retired; report().error still distinguishes loss.
  bool finish() noexcept;
  /// Cumulative outcome evidence; reading performs no I/O or preparation.
  GraphLogReport report() const noexcept;
  /// Registered Host Producer, or zero for compatibility-only logging.
  std::uint64_t producer() const noexcept { return producer_; }
  /// Allocator-only constructor; create() charges the actual shared control.
  GraphLogStream(ConstructionKey, std::shared_ptr<MemoryBudget> host, std::shared_ptr<DiskBudget> disk,
                 std::string_view path, std::uint64_t producer, std::uint64_t pid, std::size_t slots);

private:
  friend class CollectionService;
  struct Record {
    std::uint64_t raw_time_ns = 0;
    std::uint32_t phase = 0;
    std::uint32_t kind = 0;
    std::array<char, 128> backend{};
    std::uint32_t backend_bytes = 0;
  };
  struct Slot {
    std::atomic<std::uint64_t> sequence{0};
    Record record{};
  };
  struct Frame {
    std::atomic<std::uint32_t> state{0};
    std::size_t count = 0;
    std::array<Record, 64> records{};
  };
  void ensure_file();
  void append(const Record &record);
  void fail(CollectionErrorCode error) noexcept;
  std::size_t serialize(const Record &record, JsonSink sink) const;
  std::size_t serialize_report(JsonSink sink) const;
  void write_report();
  const std::shared_ptr<MemoryBudget> host_;
  const std::shared_ptr<DiskBudget> disk_;
  const std::uint64_t producer_;
  const std::uint64_t pid_;
  const std::size_t capacity_;
  BudgetMemoryResource paths_;
  std::pmr::string path_;
  std::pmr::string report_path_;
  std::pmr::string report_temporary_;
  Slot *slots_ = nullptr;
  std::array<Frame, 2> frames_{};
  std::atomic<std::uint64_t> reserve_position_{0};
  std::uint64_t consume_position_ = 0;
  std::atomic<std::uint64_t> collected_position_{0};
  std::atomic<bool> positions_exhausted_{false};
  std::size_t collect_frame_ = 0;
  std::size_t write_frame_ = 0;
  std::atomic<bool> closed_{false};
  std::atomic<bool> finished_{false};
  std::atomic<std::uint32_t> error_{0};
  ConcurrentCounter attempted_;
  ConcurrentCounter accepted_;
  ConcurrentCounter dropped_full_;
  ConcurrentCounter dropped_contention_;
  ConcurrentCounter dropped_input_;
  ConcurrentCounter copied_;
  ConcurrentCounter written_;
  ConcurrentCounter lost_;
  int descriptor_ = -1;
};

} // namespace xpool::devkit::timeline
