#include <xpool/devkit/timeline/metadata.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

#include <xpool/devkit/timeline/digest.hpp>

namespace xpool::devkit::timeline {

std::shared_ptr<MetadataCatalog> MetadataCatalog::create(std::shared_ptr<MemoryBudget> budget, BatchSource source,
                                                         std::size_t capacity, std::size_t max_revision_bytes) {
  return make_budgeted_shared<MetadataCatalog>(budget, MemoryKind::Metadata, source.producer, ConstructionKey{}, budget,
                                               source, capacity, max_revision_bytes);
}

MetadataCatalog::MetadataCatalog(ConstructionKey, std::shared_ptr<MemoryBudget> budget, BatchSource source,
                                 std::size_t capacity, std::size_t max_revision_bytes)
    : budget_(std::move(budget)), source_(source), capacity_(capacity), max_revision_bytes_(max_revision_bytes) {
  if (source.producer == 0 || source.clock_domain_id == 0 || capacity == 0 || max_revision_bytes == 0 ||
      capacity > std::numeric_limits<std::size_t>::max() / sizeof(Slot) ||
      capacity > std::numeric_limits<std::size_t>::max() / max_revision_bytes)
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer,
                          "xpool metadata catalog geometry is invalid");
  slots_ = static_cast<Slot *>(
      budget_->allocate_storage(capacity * sizeof(Slot), alignof(Slot), MemoryKind::Metadata, source.producer));
  try {
    payload_ = static_cast<std::byte *>(budget_->allocate_storage(
        capacity * max_revision_bytes, alignof(std::max_align_t), MemoryKind::Metadata, source.producer));
  } catch (...) {
    budget_->deallocate_storage(slots_, capacity * sizeof(Slot), alignof(Slot), MemoryKind::Metadata);
    throw;
  }
  for (auto index = std::size_t{0}; index < capacity_; ++index)
    std::construct_at(slots_ + index);
}

MetadataCatalog::~MetadataCatalog() {
  budget_->deallocate_storage(payload_, capacity_ * max_revision_bytes_, alignof(std::max_align_t),
                              MemoryKind::Metadata);
  budget_->deallocate_storage(slots_, capacity_ * sizeof(Slot), alignof(Slot), MemoryKind::Metadata);
}

MetadataRevision MetadataCatalog::append(std::uint64_t revision, std::span<const std::byte> bytes) {
  return append_owned(revision, bytes, nullptr);
}

MetadataRevision MetadataCatalog::append_validated(std::uint64_t revision, std::span<const std::byte> bytes,
                                                   const void *authority) {
  return append_owned(revision, bytes, authority);
}

MetadataRevision MetadataCatalog::append_owned(std::uint64_t revision, std::span<const std::byte> bytes,
                                               const void *authority) {
  const auto lock = std::lock_guard{registration_};
  if ((coverage_.owner && authority != coverage_.owner.get()) || (!coverage_.owner && authority))
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                          "xpool semantic metadata append requires its registered owner");
  const auto count = size();
  if (stopped_.load(std::memory_order_acquire) || count == capacity_ || bytes.empty() ||
      bytes.back() != std::byte{'\n'} || bytes.size() > max_revision_bytes_ ||
      (count == 0 ? revision != 1 : revision <= slots_[count - 1].revision))
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                          "xpool metadata registration violates its fixed revision capacity or lifecycle");
  auto &slot = slots_[count];
  slot.revision = revision;
  slot.size = bytes.size();
  slot.sha256 = sha256(bytes);
  std::ranges::copy(bytes, payload_ + count * max_revision_bytes_);
  count_.store(count + 1, std::memory_order_release);
  return *at(count);
}

void MetadataCatalog::install_coverage(MetadataCoverage coverage) {
  const auto lock = std::lock_guard{registration_};
  if (!coverage.owner || !coverage.validate || coverage_.owner || stopped_.load(std::memory_order_acquire) ||
      size() == 0)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                          "xpool metadata coverage requires one live prepared semantic owner");
  coverage_ = std::move(coverage);
}

std::optional<MetadataRevision> MetadataCatalog::at(std::size_t index) const noexcept {
  if (index >= size())
    return std::nullopt;
  const auto &slot = slots_[index];
  return MetadataRevision{source_, slot.revision, slot.sha256, {payload_ + index * max_revision_bytes_, slot.size}};
}

BatchInfo MetadataCatalog::resolve(std::span<const Event> copied) const {
  const auto count = size();
  if (count == 0 ||
      std::ranges::any_of(copied, [&](const Event &event) { return event.clock_domain_id != source_.clock_domain_id; }))
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                          "xpool copied events have no matching complete metadata revision");
  const auto revision = *at(count - 1);
  if (coverage_.owner)
    coverage_.validate(coverage_.owner.get(), copied, revision.revision);
  auto result = BatchInfo{.metadata_revision = revision.revision};
  std::copy_n(revision.sha256.begin(), result.metadata_sha256_prefix.size(), result.metadata_sha256_prefix.begin());
  return result;
}

MetadataResolver MetadataCatalog::resolver() noexcept {
  return {this, [](void *context, std::span<const Event> copied) {
            return static_cast<MetadataCatalog *>(context)->resolve(copied);
          }};
}

void MetadataCatalog::require_registration(BatchSource source, const BatchInfo &initial) const {
  const auto first = at(0);
  if (source.producer != source_.producer || source.run_id != source_.run_id ||
      source.clock_domain_id != source_.clock_domain_id || !first || initial.metadata_revision != first->revision ||
      !std::equal(initial.metadata_sha256_prefix.begin(), initial.metadata_sha256_prefix.end(), first->sha256.begin()))
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                          "xpool channel registration disagrees with its immutable metadata catalog");
}

} // namespace xpool::devkit::timeline
