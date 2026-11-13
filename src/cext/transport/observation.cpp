#include <xpool/transport/observation.hpp>

#include <xpool/devkit/timeline/error.hpp>

namespace xpool::transport {

void validate_observation_region(const std::optional<ObservationRegion> &region, bool transport_identity_required,
                                 std::uint64_t generation, std::uint64_t owner_producer) {
  if (region.has_value() != transport_identity_required)
    throw xpool::devkit::timeline::CollectionError(
        xpool::devkit::timeline::CollectionErrorCode::MetadataUnavailable, owner_producer,
        "xpool Transport observation region differs from immutable preparation policy");
  if (!region)
    return;
  if (region->version != xpool::devkit::timeline::kContractVersion || owner_producer == 0 ||
      region->owner_producer != owner_producer || region->endpoint_sequence == 0 || generation == 0 ||
      region->generation != generation || region->allocation_bytes < sizeof(ObservationWord) ||
      region->allocation_bytes % alignof(ObservationWord) != 0)
    throw xpool::devkit::timeline::CollectionError(
        xpool::devkit::timeline::CollectionErrorCode::MetadataUnavailable, owner_producer,
        "xpool Transport observation region has invalid version, owner, Endpoint, extent or Generation");
}

} // namespace xpool::transport
