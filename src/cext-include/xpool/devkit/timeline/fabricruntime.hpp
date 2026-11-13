#pragma once

/// \file xpool/devkit/timeline/fabricruntime.hpp
/// \brief Actual Fabric participant preparation and bounded lifecycle readout.

#include <chrono>
#include <memory>
#include <string_view>

#include <xpool/devkit/timeline/fabricrecording.hpp>
#include <xpool/devkit/timeline/metaregistry.hpp>
#include <xpool/devkit/timeline/snapshotexport.hpp>
#include <xpool/fabric/projection.hpp>

namespace xpool::devkit::fabric_observer {

/// Admit the process Fabric recording owner before Transport or Fabric allocation.
/// Metadata, source/clock, actual PE/Generation and geometry must agree. Paths
/// come from xpool.config. Disabled policy allocates nothing. One lifetime per
/// process; equal preparation is retained by the Python runtime, not recreated.
/// A cold process defers CUDA until FabricJoinPost. An already running Graph
/// service activates the complete Source before publishing its registry entry;
/// production bindings still wait for the real join. Cold retries reuse owners.
/// \throws CollectionError for missing identity or admission failure.
void prepare(std::shared_ptr<timeline::MetadataRegistry> metadata, std::uint64_t generation,
             const xpool::fabric::ArenaProjection &projection, int pe, int device, std::string_view outdir);

/// Original caller holds actual local Fabric drain/quiescence. Freeze one fresh
/// collection cutoff and queue its legacy JSON/evidence pair to the sole Writer.
/// Busy/closed publication rejects explicitly; no serialization runs here.
void export_snapshot();

/// Cancel only preparation that has never attempted Device activation. Caller
/// has stopped preparation/join calls. False retains an activated or uncertain
/// owner for its actual production retirement; this function performs no CUDA.
bool cancel_preparation();

/// Join the observation workers after actual Fabric finalization retired every
/// Device source. The original observation/production deadline is not renewed.
/// False retains Writer-held Host storage and charges; publication errors remain
/// available in export_report() after a successful join and owner release.
bool finish_retirement();

/// Last completed bounded diagnostic sample, independent of C0 quality fields.
/// Reading it never synchronizes CUDA; active counts remain lower bounds.
timeline::FabricProgress recording_progress();

/// Snapshot publication status, retained after worker join and owner release.
/// File success is separate from snapshot memory release and actual retirement.
timeline::SnapshotExportReport export_report();

} // namespace xpool::devkit::fabric_observer
