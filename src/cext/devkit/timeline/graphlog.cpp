#include <xpool/devkit/timeline/graphlog.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace xpool::devkit::timeline {
namespace {
static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);

[[noreturn]] void io_error(std::uint64_t producer, const char *path, int cause) {
  throw CollectionError(CollectionErrorCode::Io, producer, "xpool graph log write failed", path, std::strerror(cause));
}

struct FileOutput {
  int descriptor;
  const char *path;
  std::uint64_t producer;
  std::uint64_t written = 0;
  static void write(void *context, std::string_view bytes) {
    auto &output = *static_cast<FileOutput *>(context);
    while (!bytes.empty()) {
      const auto count = ::write(output.descriptor, bytes.data(), bytes.size());
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0)
        io_error(output.producer, output.path, count == 0 ? EIO : errno);
      output.written += static_cast<std::size_t>(count);
      bytes.remove_prefix(static_cast<std::size_t>(count));
    }
  }
};
} // namespace

std::shared_ptr<GraphLogStream> GraphLogStream::create(std::shared_ptr<MemoryBudget> host,
                                                       std::shared_ptr<DiskBudget> disk, std::string_view path,
                                                       std::uint64_t producer, std::uint64_t pid, std::size_t slots) {
  if (!host || !disk || pid == 0 || slots < 2 || slots > std::numeric_limits<std::size_t>::max() / sizeof(Slot) ||
      path.empty() || path.front() != '/' || path.back() == '/' || path.find('\0') != path.npos)
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer,
                          "xpool graph log geometry or identity is invalid");
  return make_budgeted_shared<GraphLogStream>(host, MemoryKind::Writer, producer, ConstructionKey{}, host,
                                              std::move(disk), path, producer, pid, slots);
}

GraphLogStream::GraphLogStream(ConstructionKey, std::shared_ptr<MemoryBudget> host, std::shared_ptr<DiskBudget> disk,
                               std::string_view path, std::uint64_t producer, std::uint64_t pid, std::size_t slots)
    : host_(std::move(host)), disk_(std::move(disk)), producer_(producer), pid_(pid), capacity_(slots),
      paths_(*host_, MemoryKind::Writer, producer), path_(path, &paths_), report_path_(path, &paths_),
      report_temporary_(path, &paths_) {
  report_path_ += ".quality.json";
  report_temporary_ += ".quality.json.tmp";
  // Cold, once-only registration of the process-owned legacy destinations.
  // Existing artifacts are retained even if Writer creation later rejects them.
  for (const auto *name : {path_.c_str(), report_path_.c_str(), report_temporary_.c_str()}) {
    struct stat status{};
    if (::lstat(name, &status) == 0) {
      if (S_ISREG(status.st_mode))
        disk_->charge_existing(static_cast<std::uint64_t>(status.st_size));
    } else if (errno != ENOENT && errno != ENOTDIR) {
      io_error(producer_, name, errno);
    }
  }
  slots_ = static_cast<Slot *>(
      host_->allocate_storage(capacity_ * sizeof(Slot), alignof(Slot), MemoryKind::Events, producer));
  std::uninitialized_value_construct_n(slots_, capacity_);
  for (auto index = std::size_t{0}; index < capacity_; ++index)
    slots_[index].sequence.store(index, std::memory_order_relaxed);
}

GraphLogStream::~GraphLogStream() {
  // The service retains this owner while any producer, worker or file call
  // remains live. Final destruction closes only its own already-idle file.
  if (descriptor_ >= 0)
    ::close(descriptor_);
  std::destroy_n(slots_, capacity_);
  host_->deallocate_storage(slots_, capacity_ * sizeof(Slot), alignof(Slot), MemoryKind::Events);
}

bool GraphLogStream::record(std::uint32_t phase, std::uint32_t kind, std::string_view backend,
                            std::uint64_t raw_time_ns) noexcept {
  if (closed_.load(std::memory_order_acquire))
    return false;
  auto position = reserve_position_.load(std::memory_order_relaxed);
  if (position >= UINT64_MAX - capacity_) {
    positions_exhausted_.store(true, std::memory_order_release);
    stop();
    return false;
  }
  attempted_.add(1);
  if (phase < 1 || phase > 2 || kind < 1 || kind > 6 || backend.empty() || backend.size() >= Record{}.backend.size() ||
      backend.find('\0') != backend.npos) {
    dropped_input_.add(1);
    return false;
  }
  auto &slot = slots_[position % capacity_];
  const auto sequence = slot.sequence.load(std::memory_order_acquire);
  if (sequence != position) {
    if (sequence > position)
      dropped_contention_.add(1);
    else
      dropped_full_.add(1);
    return false;
  }
  // Claim once. A losing caller drops without reserving a queue position, so
  // there is no hole for the consumer to skip and no producer retry loop.
  if (!reserve_position_.compare_exchange_strong(position, position + 1, std::memory_order_relaxed)) {
    dropped_contention_.add(1);
    return false;
  }
  slot.record = {.raw_time_ns = raw_time_ns, .phase = phase, .kind = kind};
  std::memcpy(slot.record.backend.data(), backend.data(), backend.size());
  slot.record.backend_bytes = static_cast<std::uint32_t>(backend.size());
  accepted_.add(1);
  slot.sequence.store(position + 1, std::memory_order_release);
  return true;
}

void GraphLogStream::stop() noexcept { closed_.store(true, std::memory_order_release); }

std::size_t GraphLogStream::collect() noexcept {
  auto &frame = frames_[collect_frame_];
  if (frame.state.load(std::memory_order_acquire) != 0)
    return 0;
  frame.state.store(1, std::memory_order_relaxed);
  frame.count = 0;
  while (frame.count < frame.records.size()) {
    auto &slot = slots_[consume_position_ % capacity_];
    if (slot.sequence.load(std::memory_order_acquire) != consume_position_ + 1)
      break;
    frame.records[frame.count++] = slot.record;
    slot.sequence.store(consume_position_ + capacity_, std::memory_order_release);
    ++consume_position_;
  }
  copied_.add(frame.count);
  frame.state.store(frame.count ? 3 : 0, std::memory_order_release);
  collected_position_.store(consume_position_, std::memory_order_release);
  if (frame.count)
    collect_frame_ = (collect_frame_ + 1) % frames_.size();
  return frame.count;
}

bool GraphLogStream::producer_empty() const noexcept {
  return reserve_position_.load(std::memory_order_acquire) == collected_position_.load(std::memory_order_acquire);
}

void GraphLogStream::fail(CollectionErrorCode error) noexcept {
  // Only the Writer updates its first error. Readers inspect the atomic code.
  if (error_.load(std::memory_order_relaxed) == 0)
    error_.store(static_cast<std::uint32_t>(error) + 1, std::memory_order_release);
  stop();
}

void GraphLogStream::ensure_file() {
  if (descriptor_ >= 0)
    return;
  for (auto offset = std::size_t{1}; offset < path_.size(); ++offset) {
    if (path_[offset] != '/')
      continue;
    path_[offset] = '\0';
    const auto result = ::mkdir(path_.c_str(), 0700);
    const auto cause = errno;
    path_[offset] = '/';
    if (result != 0 && cause != EEXIST)
      io_error(producer_, path_.c_str(), cause);
  }
  descriptor_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (descriptor_ < 0)
    io_error(producer_, path_.c_str(), errno);
}

std::size_t GraphLogStream::serialize(const Record &record, JsonSink sink) const {
  constexpr auto kinds =
      std::array{"capture_begin", "capture_end", "capture_error", "execute_begin", "execute_end", "execute_error"};
  auto out = detail::JsonOutput{sink};
  out.text("{\"backend_class\":");
  out.quoted({record.backend.data(), record.backend_bytes});
  out.text(",\"event\":");
  out.quoted(kinds[record.kind - 1]);
  out.text(",\"forward_phase\":");
  out.quoted(record.phase == 1 ? "decode" : "prefill");
  out.text(",\"pid\":");
  out.number(pid_);
  out.text(",\"time_ns\":");
  out.number(record.raw_time_ns);
  out.text("}\n");
  return out.finish();
}

void GraphLogStream::append(const Record &record) {
  const auto bytes = serialize(record, {});
  auto admission = disk_->reserve(bytes, DiskKind::Data, producer_);
  ensure_file();
  auto output = FileOutput{descriptor_, path_.c_str(), producer_};
  try {
    serialize(record, {&output, FileOutput::write});
  } catch (...) {
    admission.retain(output.written);
    throw;
  }
  admission.retain(output.written);
}

bool GraphLogStream::write_one() noexcept {
  auto &frame = frames_[write_frame_];
  if (frame.state.load(std::memory_order_acquire) != 3)
    return false;
  frame.state.store(5, std::memory_order_relaxed);
  for (auto index = std::size_t{0}; index < frame.count; ++index) {
    if (error_.load(std::memory_order_acquire) != 0) {
      lost_.add(frame.count - index);
      break;
    }
    try {
      append(frame.records[index]);
      written_.add(1);
    } catch (const CollectionError &error) {
      fail(error.code());
      lost_.add(frame.count - index);
      break;
    } catch (...) {
      fail(CollectionErrorCode::Io);
      lost_.add(frame.count - index);
      break;
    }
  }
  frame.state.store(0, std::memory_order_release);
  write_frame_ = (write_frame_ + 1) % frames_.size();
  return true;
}

GraphLogReport GraphLogStream::report() const noexcept {
  auto output = GraphLogReport{};
  for (const auto &[counter, destination] :
       std::array{std::pair{&attempted_, &output.attempted}, std::pair{&accepted_, &output.accepted},
                  std::pair{&dropped_full_, &output.dropped_full}, std::pair{&dropped_input_, &output.dropped_input},
                  std::pair{&dropped_contention_, &output.dropped_contention}, std::pair{&copied_, &output.copied},
                  std::pair{&written_, &output.written}, std::pair{&lost_, &output.lost}}) {
    const auto value = counter->lower_bound();
    *destination = value.value;
    output.counter_exhausted = output.counter_exhausted || value.exhausted;
  }
  output.closed = closed_.load(std::memory_order_acquire);
  output.finished = finished_.load(std::memory_order_acquire);
  output.counter_exhausted = output.counter_exhausted || positions_exhausted_.load(std::memory_order_acquire);
  if (const auto error = error_.load(std::memory_order_acquire))
    output.error = static_cast<CollectionErrorCode>(error - 1);
  return output;
}

std::size_t GraphLogStream::serialize_report(JsonSink sink) const {
  const auto value = report();
  auto out = detail::JsonOutput{sink};
  out.text("{\"schema\":\"xpool.graph_log_quality.v1\",\"producer\":");
  out.number(producer_);
  out.text(",\"pid\":");
  out.number(pid_);
  for (const auto &[name, count] :
       std::array{std::pair{"attempted", value.attempted}, std::pair{"accepted", value.accepted},
                  std::pair{"dropped_full", value.dropped_full}, std::pair{"dropped_input", value.dropped_input},
                  std::pair{"dropped_contention", value.dropped_contention}, std::pair{"copied", value.copied},
                  std::pair{"written", value.written}, std::pair{"lost", value.lost}}) {
    out.text(",");
    out.quoted(name);
    out.text(":");
    out.number(count);
  }
  out.text(",\"counter_exhausted\":");
  out.text(value.counter_exhausted ? "true" : "false");
  out.text(",\"error_code\":");
  if (value.error)
    out.number(static_cast<std::uint32_t>(*value.error));
  else
    out.text("null");
  out.text("}\n");
  return out.finish();
}

void GraphLogStream::write_report() {
  const auto bytes = serialize_report({});
  auto admission = disk_->reserve(bytes, DiskKind::Metadata, producer_);
  const auto descriptor = ::open(report_temporary_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (descriptor < 0)
    io_error(producer_, report_temporary_.c_str(), errno);
  auto output = FileOutput{descriptor, report_temporary_.c_str(), producer_};
  try {
    serialize_report({&output, FileOutput::write});
  } catch (...) {
    ::close(descriptor);
    admission.retain(output.written);
    throw;
  }
  const auto closed = ::close(descriptor);
  const auto cause = errno;
  admission.retain(output.written);
  if (closed != 0)
    io_error(producer_, report_temporary_.c_str(), cause);
  if (::syscall(SYS_renameat2, AT_FDCWD, report_temporary_.c_str(), AT_FDCWD, report_path_.c_str(), RENAME_NOREPLACE) !=
      0)
    io_error(producer_, report_temporary_.c_str(), errno);
}

bool GraphLogStream::finish() noexcept {
  if (finished_.load(std::memory_order_acquire))
    return true;
  if (!closed_.load(std::memory_order_acquire) || !producer_empty())
    return false;
  for (const auto &frame : frames_)
    if (frame.state.load(std::memory_order_acquire) != 0)
      return false;
  try {
    if (error_.load(std::memory_order_acquire) == 0)
      ensure_file();
  } catch (const CollectionError &error) {
    fail(error.code());
  } catch (...) {
    fail(CollectionErrorCode::Io);
  }
  if (descriptor_ >= 0) {
    if (::close(descriptor_) != 0)
      fail(CollectionErrorCode::Io);
    descriptor_ = -1;
  }
  try {
    write_report();
  } catch (const CollectionError &error) {
    fail(error.code());
  } catch (...) {
    fail(CollectionErrorCode::Io);
  }
  finished_.store(true, std::memory_order_release);
  return true;
}

} // namespace xpool::devkit::timeline
