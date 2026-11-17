#pragma once

/// \file xpool/devkit/timeline/transportruntime.hpp
/// \brief Production Transport preparation, export and proven retirement.

#include <memory>
#include <string_view>

#include <xpool/devkit/timeline/metaregistry.hpp>
#include <xpool/transport/observation.hpp>

namespace xpool::devkit::transport_observer {

/// Prepare the registered Endpoint before publication/attach. AtnAgent borrows
/// its cold Fabric channel when present; Instance prepares its own Device Source.
/// The stable file path/envelope come from the original Python model identity.
/// Same descriptor retries are idempotent; replacement requires prior close.
void prepare_endpoint(std::shared_ptr<timeline::MetadataRegistry> metadata,
                      const xpool::transport::ObservationRegion &region, std::size_t instance_index, std::uint32_t rank,
                      int device, bool atnagent, std::string_view path);

/// Install prepared Endpoint entries after the shared Fabric channel activates,
/// before any resident kernels or Graph capture. No second Source is created.
void activate_borrowed();
/// Clear the borrowed binding at the real all-producer retirement boundary,
/// before Fabric retires the shared Device allocation. Repeated calls are safe.
void retire_borrowed();
/// After actual Endpoint close, join an owned Source or release borrowed Host
/// control after the Fabric Source retires. False retains resources on timeout.
bool finish_retirement();

} // namespace xpool::devkit::transport_observer
