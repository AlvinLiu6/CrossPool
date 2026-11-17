#include <xpool/devkit/timeline/hostchannel.hpp>

#include <exception>
#include <memory>
#include <utility>

namespace xpool::devkit::timeline {

std::shared_ptr<HostChannel> HostChannel::create(std::shared_ptr<MemoryBudget> host, BatchSource source, BatchInfo info,
                                                 std::size_t slots, std::size_t batches, std::size_t events_per_batch) {
  if (!host || info.metadata_revision == 0)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                          "xpool Host channel has no metadata");
  try {
    return make_budgeted_shared<HostChannel>(host, MemoryKind::Control, source.producer, ConstructionKey{}, host,
                                             source, info, slots, batches, events_per_batch);
  } catch (const CollectionError &) {
    throw;
  } catch (const std::exception &cause) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer, "xpool Host channel geometry failed",
                          std::nullopt, cause.what());
  }
}

HostChannel::HostChannel(ConstructionKey, std::shared_ptr<MemoryBudget> host, BatchSource source, BatchInfo info,
                         std::size_t slots, std::size_t batches, std::size_t events_per_batch)
    : host_(std::move(host)), info_(info) {
  const auto layout = HostBatchPool::layout(batches, events_per_batch);
  delivery_ = host_->allocate_host(layout.total_bytes, 256, MemoryKind::Delivery, source.producer);
  scratch_ = host_->allocate_host(MemoryBudget::array_bytes(events_per_batch, sizeof(Event), 256), 256,
                                  MemoryKind::Transfer, source.producer);
  buffer_.emplace(*host_, source, slots);
  auto *base = delivery_.bytes().data();
  auto *events = reinterpret_cast<Event *>(base);
  auto *controls = reinterpret_cast<BatchControl *>(base + layout.control_offset);
  for (auto index = std::size_t{0}; index < batches * events_per_batch; ++index)
    std::construct_at(events + index);
  for (auto index = std::size_t{0}; index < batches; ++index)
    std::construct_at(controls + index);
  auto *state = std::construct_at(reinterpret_cast<BatchPoolState *>(base + layout.state_offset));
  batches_.emplace(source, events_per_batch, std::span{events, batches * events_per_batch},
                   std::span{controls, batches}, *state);
  auto *scratch = reinterpret_cast<Event *>(scratch_.bytes().data());
  for (auto index = std::size_t{0}; index < events_per_batch; ++index)
    std::construct_at(scratch + index);
  collector_.emplace(*batches_, std::span{scratch, events_per_batch});
}

} // namespace xpool::devkit::timeline
