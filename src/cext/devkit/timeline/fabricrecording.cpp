#include <xpool/devkit/timeline/fabricrecording.hpp>

#include <array>

#include <xpool/devkit/timeline/error.hpp>
#include <xpool/utils/layout.hpp>

namespace xpool::devkit::timeline {

FabricRecordingLayout FabricRecordingLayout::for_instances(std::size_t instances, std::uint64_t producer) {
  return for_config({.generation = 1, .pe = 0, .instances = instances, .lanes = 1, .atnagent = true}, producer);
}

FabricRecordingLayout FabricRecordingLayout::for_config(FabricRecordingConfig config, std::uint64_t producer) {
  if (config.instances == 0 || config.lanes == 0 || config.generation == 0 || config.pe < 0 ||
      (config.atnagent && config.coordinator))
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer,
                          "xpool Fabric recording requires a positive Instance table");
  try {
    using xpool::utils::layout::LayoutRegionSpec;
    const auto regions = std::to_array<LayoutRegionSpec>({
        LayoutRegionSpec::object<FabricRecordingState>("Fabric recording state"),
        LayoutRegionSpec::array<FabricAtnCurrent>("Fabric current AtnAgent operations",
                                                  config.atnagent ? config.instances : 0),
        LayoutRegionSpec::array<FabricCoordinatorCurrent>("Fabric current Coordinator operations",
                                                          config.coordinator ? config.instances : 0),
        LayoutRegionSpec::array<FabricFfnCurrent>("Fabric current FfnAgent operations",
                                                  config.atnagent ? 0 : config.lanes),
    });
    const auto layout = xpool::utils::layout::LayoutPlan{regions, 256};
    return {.current_offset = layout[1].offset,
            .coordinator_offset = layout[2].offset,
            .ffn_offset = layout[3].offset,
            .total_bytes = layout.total_bytes};
  } catch (const c10::Error &error) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool Fabric recording layout overflows",
                          std::nullopt, error.what());
  }
}

} // namespace xpool::devkit::timeline
