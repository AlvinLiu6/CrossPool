#include <xpool/devkit/timeline/transportrecording.hpp>

#include <array>

#include <xpool/devkit/timeline/error.hpp>
#include <xpool/utils/layout.hpp>

namespace xpool::devkit::timeline {

std::size_t transport_recording_bytes(std::size_t capacity, std::uint64_t producer) {
  if (capacity == 0)
    return 0;
  try {
    using xpool::utils::layout::LayoutRegionSpec;
    const auto regions =
        std::array{LayoutRegionSpec::array<TransportRecordingEndpoint>("Transport current Endpoint facts", capacity)};
    return xpool::utils::layout::LayoutPlan{regions, 256}.total_bytes;
  } catch (const c10::Error &error) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool Transport recording layout overflows",
                          std::nullopt, error.what());
  }
}

} // namespace xpool::devkit::timeline
