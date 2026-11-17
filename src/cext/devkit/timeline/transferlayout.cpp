#include <xpool/devkit/timeline/transferlayout.hpp>

#include <xpool/devkit/timeline/chunk.hpp>
#include <xpool/devkit/timeline/error.hpp>

namespace xpool::devkit::timeline {

TransferLayout TransferLayout::for_capacity(std::size_t capacity) {
  if (capacity == 0 || capacity > Chunk::maximum_events) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool transfer frame capacity is invalid");
  }
  const auto round = [](std::size_t size, std::size_t alignment) {
    return size + (alignment - size % alignment) % alignment;
  };
  // The count is checked against 4095 before arithmetic, so every actual
  // native-type product is bounded independently of a requested allocation.
  const auto events = round(sizeof(TransferHeader), alignof(Event));
  const auto host = round(events + capacity * sizeof(Event), 256);
  const auto tokens = round(host, alignof(ReadToken));
  const auto device = round(tokens + capacity * sizeof(ReadToken), 256);
  return {capacity, events, host, tokens, device};
}

} // namespace xpool::devkit::timeline
