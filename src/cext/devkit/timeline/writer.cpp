#include <xpool/devkit/timeline/writer.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <limits>
#include <string_view>

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <xpool/devkit/timeline/digest.hpp>

namespace xpool::devkit::timeline {

namespace {

[[noreturn]] void io_error(std::uint64_t producer, const char *path, int cause) {
  throw CollectionError(CollectionErrorCode::Io, producer, "xpool raw artifact publication failed", path,
                        std::strerror(cause));
}

bool same_source(const BatchSource &left, const BatchSource &right) {
  return left.producer == right.producer && left.clock_domain_id == right.clock_domain_id &&
         left.run_id == right.run_id;
}

// Fixed C0 field count and integer widths fit this stack-owned formatter. The
// process worker accounts its complete stack separately from Writer scratch.
class QualityJson {
public:
  void text(std::string_view value) {
    if (value.size() > storage_.size() - used_) {
      throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool quality formatting capacity exceeded");
    }
    std::ranges::copy(value, storage_.begin() + static_cast<std::ptrdiff_t>(used_));
    used_ += value.size();
  }
  void number(std::uint64_t value) {
    const auto result = std::to_chars(storage_.data() + used_, storage_.data() + storage_.size(), value);
    if (result.ec != std::errc{}) {
      throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool quality integer formatting failed");
    }
    used_ = static_cast<std::size_t>(result.ptr - storage_.data());
  }
  void field(std::string_view name, std::uint64_t value) {
    text("\"");
    text(name);
    text("\":");
    number(value);
  }
  std::span<const std::byte> bytes() const { return std::as_bytes(std::span{storage_}.first(used_)); }

private:
  std::array<char, 4096> storage_{};
  std::size_t used_ = 0;
};

} // namespace

RawWriter::RawWriter(BatchSource source, std::filesystem::path outdir, MemoryBudget &host, DiskBudget &disk,
                     std::size_t chunk_bytes, std::size_t metadata_capacity)
    : source_(source), host_(host), disk_(disk), chunk_bytes_(chunk_bytes), metadata_capacity_(metadata_capacity),
      path_resource_(host, MemoryKind::Writer, source.producer), outdir_(outdir.native(), &path_resource_),
      final_path_(&path_resource_), temporary_path_(&path_resource_),
      registry_resource_(host, MemoryKind::Metadata, source.producer), metadata_(&registry_resource_) {
  if (source.producer == 0 || source.clock_domain_id == 0 || outdir_.empty() || outdir_.front() != '/' ||
      outdir_.size() > 1048576 || chunk_bytes < Chunk::required_bytes(1) || chunk_bytes > Chunk::maximum_bytes ||
      metadata_capacity == 0) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer, "xpool raw Writer sizing is invalid");
  }
  final_path_.reserve(outdir_.size() + 128);
  temporary_path_.reserve(outdir_.size() + 132);
  metadata_.reserve(metadata_capacity_);
  scratch_ = static_cast<std::byte *>(
      host_.allocate_storage(chunk_bytes_, alignof(std::max_align_t), MemoryKind::Writer, source_.producer));
}

RawWriter::~RawWriter() {
  host_.deallocate_storage(scratch_, chunk_bytes_, alignof(std::max_align_t), MemoryKind::Writer);
}

void RawWriter::begin_path(std::string_view prefix) {
  final_path_ = outdir_;
  final_path_ += '/';
  final_path_ += prefix;
}
void RawWriter::append_number(std::uint64_t value) {
  auto digits = std::array<char, 20>{};
  const auto converted = std::to_chars(digits.data(), digits.data() + digits.size(), value);
  final_path_.append(digits.data(), converted.ptr);
}
void RawWriter::publish_file(std::span<const std::byte> bytes, DiskKind kind) {
  auto admission = disk_.reserve(bytes.size(), kind, source_.producer);
  temporary_path_ = final_path_;
  temporary_path_ += ".tmp";
  // All path buffers were admitted before service. Directory preparation runs
  // here on the Writer, using that same fixed buffer and no filesystem owner.
  for (auto offset = std::size_t{1}; offset < final_path_.size(); ++offset) {
    if (final_path_[offset] != '/')
      continue;
    final_path_[offset] = '\0';
    const auto result = ::mkdir(final_path_.c_str(), 0700);
    const auto cause = errno;
    final_path_[offset] = '/';
    if (result != 0 && cause != EEXIST)
      io_error(source_.producer, temporary_path_.c_str(), cause);
  }
  auto descriptor = ::open(temporary_path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (descriptor < 0) {
    io_error(source_.producer, temporary_path_.c_str(), errno);
  }
  auto written = std::size_t{0};
  auto failure = 0;
  while (written < bytes.size()) {
    const auto count = ::write(descriptor, bytes.data() + written, bytes.size() - written);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      failure = count == 0 ? EIO : errno;
      break;
    }
    written += static_cast<std::size_t>(count);
  }
  const auto closed = ::close(descriptor);
  if (closed != 0 && failure == 0) {
    failure = errno;
  }
  // The complete file admission becomes its surviving extent on every path.
  // rename only changes this file's name and needs no second data charge.
  admission.retain(written);
  if (failure != 0) {
    io_error(source_.producer, temporary_path_.c_str(), failure);
  }
  if (::syscall(SYS_renameat2, AT_FDCWD, temporary_path_.c_str(), AT_FDCWD, final_path_.c_str(), RENAME_NOREPLACE) !=
      0) {
    io_error(source_.producer, temporary_path_.c_str(), errno);
  }
}

MetadataPublication RawWriter::publish_metadata(std::uint64_t revision, std::span<const std::byte> bytes) {
  if (revision == 0 || (!metadata_.empty() && revision <= metadata_.back().revision) ||
      metadata_.size() == metadata_capacity_ || bytes.empty() || bytes.back() != std::byte{'\n'}) {
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                          "xpool metadata needs bounded increasing revisions and a final newline");
  }
  const auto result = MetadataPublication{revision, sha256(bytes)};
  begin_path("metadata.");
  append_number(source_.producer);
  final_path_ += '.';
  append_number(revision);
  final_path_ += ".json";
  publish_file(bytes, DiskKind::Metadata);
  // Capacity is reserved before service and this trivial value cannot throw.
  metadata_.push_back(result);
  return result;
}

void RawWriter::publish(BatchView batch) {
  const auto metadata = std::ranges::find(metadata_, batch.info.metadata_revision, &MetadataPublication::revision);
  if (!same_source(batch.source, source_) || metadata == metadata_.end() ||
      !std::equal(batch.info.metadata_sha256_prefix.begin(), batch.info.metadata_sha256_prefix.end(),
                  metadata->sha256.begin())) {
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                          "xpool chunk requires its exact published source metadata");
  }
  const auto sequence = chunks_.try_next();
  if (sequence == 0) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, source_.producer, "xpool chunk identities exhausted");
  }
  auto encoded = std::span<const std::byte>{};
  try {
    encoded = batch.chunk(sequence).encode({scratch_, chunk_bytes_});
  } catch (const std::exception &cause) {
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                          "xpool chunk encoding rejected its input", std::nullopt, cause.what());
  }
  begin_path("chunks/");
  append_number(source_.producer);
  final_path_ += '/';
  append_number(sequence);
  final_path_ += ".xpool";
  publish_file(encoded, DiskKind::Data);
}

void RawWriter::publish_pending_metadata(const MetadataCatalog &catalog) {
  const auto count = catalog.size();
  if (count != 0)
    ensure_metadata(catalog.at(count - 1)->revision, catalog);
}

void RawWriter::ensure_metadata(std::uint64_t revision, const MetadataCatalog &catalog) {
  if (!same_source(source_, catalog.source()))
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                          "xpool writer metadata catalog belongs to another source");
  for (auto index = std::size_t{0}; index < catalog.size(); ++index) {
    const auto document = *catalog.at(index);
    if (document.revision > revision)
      break;
    const auto published = std::ranges::find(metadata_, document.revision, &MetadataPublication::revision);
    if (published == metadata_.end())
      publish_metadata(document.revision, document.bytes);
    else if (published->sha256 != document.sha256)
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                            "xpool immutable metadata revision has different bytes");
    if (document.revision == revision)
      return;
  }
  throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                        "xpool writer cannot publish the batch's frozen metadata revision");
}

bool RawWriter::write_one(Collector &collector, const MetadataCatalog *catalog) {
  if (!same_source(collector.source(), source_)) {
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                          "xpool Writer and Collector registrations differ");
  }
  if (chunks_.quiescent_value() == std::numeric_limits<std::uint64_t>::max()) {
    collector.fail(StopReason::IdentityExhausted);
  }
  if (collector.stop_requested()) {
    collector.discard_ready();
    return false;
  }
  const auto maximum = static_cast<std::uint32_t>((chunk_bytes_ - Chunk::header_bytes) / Event::encoded_bytes);
  const auto result = poll(collector, maximum);
  if (!result.token) {
    return false;
  }
  auto metadata_stage = catalog != nullptr;
  try {
    const auto batch = collector.view(*result.token);
    if (catalog)
      ensure_metadata(batch.info.metadata_revision, *catalog);
    metadata_stage = false;
    publish(batch);
  } catch (const CollectionError &cause) {
    const auto reason = cause.code() == CollectionErrorCode::ResourceBudget
                            ? StopReason::DiskQuota
                            : (metadata_stage || cause.code() == CollectionErrorCode::MetadataUnavailable
                                   ? StopReason::MetadataUnavailable
                                   : StopReason::IoError);
    collector.fail(reason);
    collector.report(*result.token, {.post_commit_lost = result.events.size()});
    release(collector, *result.token);
    throw;
  } catch (const std::exception &cause) {
    collector.fail(StopReason::MetadataUnavailable);
    collector.report(*result.token, {.post_commit_lost = result.events.size()});
    release(collector, *result.token);
    throw CollectionError(CollectionErrorCode::Allocation, source_.producer,
                          "xpool Writer failed before complete publication", std::nullopt, cause.what());
  }
  collector.report(*result.token, {.exported = result.events.size()});
  release(collector, *result.token);
  return true;
}

void RawWriter::publish_quality(const QualitySnapshot &snapshot) {
  if (snapshot.contract_version != kContractVersion || snapshot.run_id != source_.run_id ||
      snapshot.producer_id != source_.producer || snapshot.revision == 0 || snapshot.revision <= quality_revision_) {
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                          "xpool quality publication needs matching increasing identities");
  }
  auto output = QualityJson{};
  output.text("{");
  output.field("contract_version", snapshot.contract_version);
  output.text(",\"run_id\":\"");
  constexpr auto hex = std::string_view{"0123456789abcdef"};
  for (auto index = std::size_t{0}; index < source_.run_id.size(); ++index) {
    if (index == 4 || index == 6 || index == 8 || index == 10) {
      output.text("-");
    }
    const auto byte = std::to_integer<unsigned>(source_.run_id[index]);
    output.text(hex.substr(byte >> 4, 1));
    output.text(hex.substr(byte & 15, 1));
  }
  output.text("\",");
  output.field("producer_id", snapshot.producer_id);
  output.text(",");
  output.field("revision", snapshot.revision);
  output.text(",");
  output.field("host_time_ns", snapshot.host_time_ns);
  constexpr auto collection_names = std::array{"active", "stopped", "closed"};
  constexpr auto export_names = std::array{"active", "failed", "closed"};
  constexpr auto stop_names =
      std::array{"disk_quota", "io_error", "metadata_unavailable", "identity_exhausted", "shutdown_deadline"};
  constexpr auto tail_names = std::array{"complete", "pending", "unknown"};
  output.text(",\"collection_state\":\"");
  output.text(collection_names.at(static_cast<std::size_t>(snapshot.collection_state)));
  output.text("\",\"export_state\":\"");
  output.text(export_names.at(static_cast<std::size_t>(snapshot.export_state)));
  output.text("\",\"stop_reason\":");
  if (snapshot.stop_reason) {
    output.text("\"");
    output.text(stop_names.at(static_cast<std::size_t>(*snapshot.stop_reason)));
    output.text("\"");
  } else {
    output.text("null");
  }
  output.text(",");
  output.field("last_attempted_sequence", snapshot.last_attempted_sequence);
  output.text(",\"counters\":{");
  constexpr auto fields =
      std::array{std::pair{"attempted", &QualityCounters::attempted},
                 std::pair{"committed", &QualityCounters::committed},
                 std::pair{"dropped_full", &QualityCounters::dropped_full},
                 std::pair{"abandoned", &QualityCounters::abandoned},
                 std::pair{"stale_token", &QualityCounters::stale_token},
                 std::pair{"duplicate_commit", &QualityCounters::duplicate_commit},
                 std::pair{"copied", &QualityCounters::copied},
                 std::pair{"exported", &QualityCounters::exported},
                 std::pair{"post_commit_lost", &QualityCounters::post_commit_lost},
                 std::pair{"snapshot_operations_attempted", &QualityCounters::snapshot_operations_attempted},
                 std::pair{"snapshot_operations_dropped", &QualityCounters::snapshot_operations_dropped},
                 std::pair{"attachment_attempted", &QualityCounters::attachment_attempted},
                 std::pair{"attachment_capacity_exhausted", &QualityCounters::attachment_capacity_exhausted},
                 std::pair{"attachment_payload_oversize", &QualityCounters::attachment_payload_oversize},
                 std::pair{"attachment_not_collected", &QualityCounters::attachment_not_collected},
                 std::pair{"attachment_publish_failed", &QualityCounters::attachment_publish_failed},
                 std::pair{"sample_attempted", &QualityCounters::sample_attempted},
                 std::pair{"sample_failed", &QualityCounters::sample_failed},
                 std::pair{"sample_skipped_busy", &QualityCounters::sample_skipped_busy},
                 std::pair{"context_unavailable", &QualityCounters::context_unavailable}};
  auto first = true;
  for (const auto &[name, member] : fields) {
    if (!first) {
      output.text(",");
    }
    first = false;
    output.field(name, snapshot.counters.*member);
  }
  output.text("},\"pending\":");
  if (snapshot.pending) {
    output.number(*snapshot.pending);
  } else {
    output.text("null");
  }
  output.text(",\"tail\":\"");
  output.text(tail_names.at(static_cast<std::size_t>(snapshot.tail)));
  output.text("\",\"counter_exhausted\":");
  output.text(snapshot.counter_exhausted ? "true" : "false");
  output.text(",\"exact_totals\":");
  output.text(snapshot.exact_totals ? "true" : "false");
  output.text("}\n");
  begin_path("quality.");
  append_number(source_.producer);
  final_path_ += '.';
  append_number(snapshot.revision);
  final_path_ += ".json";
  publish_file(output.bytes(), DiskKind::Metadata);
  quality_revision_ = snapshot.revision;
}

} // namespace xpool::devkit::timeline
