#pragma once

/// \file xpool/devkit/timeline/transferlayout.hpp
/// \brief Actual Host/Device staging geometry for asynchronous event transfers.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <xpool/devkit/timeline/fabricrecording.hpp>
#include <xpool/devkit/timeline/source.hpp>

namespace xpool::devkit::timeline {

/// Private native transfer header, never a wire or IPC format. Device kernels
/// write it; Host inspection waits for the corresponding copy/completion event.
struct TransferHeader {
  /// Number of held committed generations packed in this frame.
  std::uint64_t count = 0;
  /// Failed reclamations, read only after the reclamation event completes.
  std::uint64_t release_failures = 0;
  /// Exact after proven producer retirement; otherwise cumulative lower bounds.
  SourceProgress progress{};
  /// Optional Fabric original-operation/diagnostic sample; never mutable facts.
  FabricProgress fabric{};
};

/// One fixed pinned frame and one Device frame, using actual native types.
struct TransferLayout {
  /// Fixed maximum Events packed by this frame.
  std::size_t capacity;
  /// Aligned Event array following TransferHeader.
  std::size_t event_offset;
  /// Rounded complete pinned frame and D2H byte count.
  std::size_t host_bytes;
  /// Device-only ReadToken array, following the entire copied region.
  std::size_t token_offset;
  /// Rounded complete Device frame including retained read tokens.
  std::size_t device_bytes;

  /// \throws CollectionError with resource_budget for zero/oversized capacity.
  /// This pure Host query allocates nothing and initializes no CUDA state.
  static TransferLayout for_capacity(std::size_t capacity);
};

static_assert(std::is_trivially_copyable_v<TransferHeader>);

} // namespace xpool::devkit::timeline
