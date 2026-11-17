#pragma once

/// \file xpool/devkit/timeline/artifact.hpp
/// \brief Immutable attachment handoff to the process Writer.
#include <memory>

#include <xpool/devkit/timeline/writer.hpp>

namespace xpool::devkit::timeline {
/// A budget-owned payload submitted once after its production facts are immutable.
/// The Writer alone serializes and publishes it; no Producer performs file I/O.
class ArtifactJob {
public:
  virtual ~ArtifactJob() = default;
  /// Registered source selects the exact existing Writer and disk ledger.
  virtual BatchSource source() const noexcept = 0;
  /// Number of Routing attachment outcomes affected by a publication failure.
  virtual std::size_t routing_count() const noexcept { return 0; }
  /// Publish at most once. Typed failures retain the published prefix.
  virtual void write(RawWriter &writer) = 0;
};
} // namespace xpool::devkit::timeline
