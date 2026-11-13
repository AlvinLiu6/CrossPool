#include <xpool/devkit/timeline/snapshotwriter.hpp>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <xpool/devkit/timeline/digest.hpp>

namespace xpool::devkit::timeline {

namespace {

[[noreturn]] void io_error(std::uint64_t producer, const char *path, int cause) {
  throw CollectionError(CollectionErrorCode::Io, producer, "xpool snapshot pair publication failed", path,
                        std::strerror(cause));
}

void absent(const std::pmr::string &path) {
  struct stat status{};
  if (::lstat(path.c_str(), &status) == 0)
    io_error(0, path.c_str(), EEXIST);
  if (errno != ENOENT)
    io_error(0, path.c_str(), errno);
}

struct FileOutput {
  int descriptor;
  const char *path;
  std::uint64_t producer;
  std::uint64_t expected;
  std::uint64_t written = 0;
  Sha256 digest{};

  static void write(void *context, std::string_view bytes) {
    auto &file = *static_cast<FileOutput *>(context);
    if (bytes.size() > file.expected - file.written)
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, file.producer,
                            "xpool snapshot formatter exceeded its measured size", file.path);
    file.digest.update(std::as_bytes(std::span{bytes.data(), bytes.size()}));
    while (!bytes.empty()) {
      const auto count = ::write(file.descriptor, bytes.data(), bytes.size());
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0)
        io_error(file.producer, file.path, count == 0 ? EIO : errno);
      file.written += static_cast<std::size_t>(count);
      bytes.remove_prefix(static_cast<std::size_t>(count));
    }
  }
};

template <typename Serialize>
DiskBudget::Extent write_file(const std::pmr::string &path, DiskBudget::Reservation &admission, std::uint64_t producer,
                              std::uint64_t size, Serialize serialize,
                              std::optional<std::array<std::byte, 32>> expected_hash = std::nullopt) {
  const auto descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (descriptor < 0)
    io_error(producer, path.c_str(), errno);
  auto output = FileOutput{descriptor, path.c_str(), producer, size};
  try {
    serialize(JsonSink{&output, FileOutput::write});
    if (output.written != size || (expected_hash && output.digest.finish() != *expected_hash))
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, producer,
                            "xpool snapshot formatter changed across its immutable read", path.c_str());
  } catch (...) {
    // Preserve the operation's original error; close cannot establish success.
    ::close(descriptor);
    admission.retain(output.written);
    throw;
  }
  const auto closed = ::close(descriptor);
  const auto cause = errno;
  auto extent = admission.retain_extent(output.written);
  if (closed != 0)
    io_error(producer, path.c_str(), cause);
  return extent;
}

void replace_file(const std::pmr::string &temporary, const std::pmr::string &path, DiskBudget::Extent &old,
                  DiskBudget::Extent replacement, std::uint64_t producer) {
  if (::rename(temporary.c_str(), path.c_str()) != 0)
    io_error(producer, temporary.c_str(), errno);
  // Tokens are private to this owner and originate from its one shared ledger.
  // This retirement cannot fail or allocate after the successful rename.
  static_cast<void>(old.replaced_by(std::move(replacement)));
}

} // namespace

std::shared_ptr<SnapshotPublisher> SnapshotPublisher::create(std::shared_ptr<MemoryBudget> host,
                                                             std::shared_ptr<DiskBudget> disk, std::string_view path,
                                                             detail::LegacyScope scope) {
  return make_budgeted_shared<SnapshotPublisher>(host, MemoryKind::Writer, 0, ConstructionKey{}, host, std::move(disk),
                                                 path, scope);
}

SnapshotPublisher::SnapshotPublisher(ConstructionKey, std::shared_ptr<MemoryBudget> host,
                                     std::shared_ptr<DiskBudget> disk, std::string_view path, detail::LegacyScope scope)
    : host_(std::move(host)), disk_(std::move(disk)), scope_(scope), paths_(*host_, MemoryKind::Writer),
      snapshot_(path, &paths_), evidence_(path, &paths_), snapshot_temporary_(path, &paths_),
      evidence_temporary_(path, &paths_) {
  if (!disk_ || path.empty() || path.front() != '/' || path.back() == '/' || path.find('\0') != path.npos)
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool snapshot path or disk ledger is invalid");
  evidence_ += ".evidence.json";
  snapshot_temporary_ += ".tmp";
  evidence_temporary_ += ".evidence.json.tmp";
  for (const auto *file : {&snapshot_, &evidence_, &snapshot_temporary_, &evidence_temporary_})
    absent(*file);
}

void SnapshotPublisher::publish(const SnapshotRead &read, SnapshotDocument document) {
  const auto &report = *read.evidence();
  const auto producer = report.source.producer;
  if (failed_)
    io_error(producer, snapshot_.c_str(), ECANCELED);
  if (!report.sequence || !document.serialize)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, producer,
                          "xpool snapshot envelope needs exact original operation counts and a formatter");
  if (!read.belongs_to(scope_))
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, producer,
                          "xpool snapshot does not belong to its registered lifecycle path");
  try {
    auto digest = Sha256{};
    document.serialize(read, document.context,
                       {&digest, [](void *context, std::string_view bytes) {
                          static_cast<Sha256 *>(context)->update(std::as_bytes(std::span{bytes.data(), bytes.size()}));
                        }});
    const auto hash = digest.finish();
    const auto evidence_size = report.serialize({}, hash);
    auto evidence_admission = disk_->reserve(evidence_size, DiskKind::Data, producer);
    auto snapshot_admission = disk_->reserve(digest.size(), DiskKind::Data, producer);
    auto evidence_extent = write_file(evidence_temporary_, evidence_admission, producer, evidence_size,
                                      [&](JsonSink sink) { report.serialize(sink, hash); });
    replace_file(evidence_temporary_, evidence_, evidence_extent_, std::move(evidence_extent), producer);
    auto snapshot_extent = write_file(
        snapshot_temporary_, snapshot_admission, producer, digest.size(),
        [&](JsonSink sink) { document.serialize(read, document.context, sink); }, hash);
    replace_file(snapshot_temporary_, snapshot_, snapshot_extent_, std::move(snapshot_extent), producer);
  } catch (...) {
    failed_ = true;
    throw;
  }
}

} // namespace xpool::devkit::timeline
