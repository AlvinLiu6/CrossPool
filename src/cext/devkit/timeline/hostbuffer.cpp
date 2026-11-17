#include <xpool/devkit/timeline/hostbuffer.hpp>

#include <memory>
#include <stdexcept>

#include <xpool/devkit/timeline/error.hpp>

namespace xpool::devkit::timeline {

HostBuffer::HostBuffer(MemoryBudget &budget, BatchSource source, std::size_t capacity) {
  try {
    const auto geometry = layout(capacity);
    storage_ = budget.allocate_host(geometry.total_bytes, 256, MemoryKind::Events, source.producer);
    auto *base = storage_.bytes().data();
    auto *events = reinterpret_cast<Event *>(base);
    auto *slots = reinterpret_cast<SlotControl *>(base + geometry.slots.control_offset);
    for (auto index = std::size_t{0}; index < capacity; ++index) {
      std::construct_at(events + index);
      std::construct_at(slots + index);
    }
    auto *pool = std::construct_at(reinterpret_cast<SlotPoolState *>(base + geometry.slots.state_offset));
    auto *state = std::construct_at(reinterpret_cast<SourceState *>(base + geometry.source_state_offset));
    source_.emplace(source, std::span{events, capacity}, std::span{slots, capacity}, *pool, *state);
  } catch (const std::invalid_argument &cause) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer, "xpool host source setup failed",
                          std::nullopt, cause.what());
  } catch (const std::overflow_error &cause) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer, "xpool host source sizing overflow",
                          std::nullopt, cause.what());
  }
}

HostBuffer::HostBuffer(BatchSource source, std::span<Event> events, std::span<SlotControl> slots, SlotPoolState &pool,
                       SourceState &state) {
  try {
    source_.emplace(source, events, slots, pool, state);
  } catch (const std::invalid_argument &cause) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer, "xpool borrowed host source is invalid",
                          std::nullopt, cause.what());
  }
}

ReserveResult try_reserve_host(HostBuffer &buffer) {
  try {
    return buffer.source().try_reserve();
  } catch (const std::overflow_error &cause) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, buffer.source().source().producer,
                          "xpool host source reservation concurrency exhausted", std::nullopt, cause.what());
  }
}

CommitStatus commit_host(HostBuffer &buffer, WriteToken token, const Event &event) {
  try {
    return buffer.source().commit(token, event);
  } catch (const std::invalid_argument &cause) {
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, buffer.source().source().producer,
                          "xpool event source clock mismatch", std::nullopt, cause.what());
  }
}

AbandonStatus abandon_host(HostBuffer &buffer, WriteToken token) { return buffer.source().abandon(token); }

} // namespace xpool::devkit::timeline
