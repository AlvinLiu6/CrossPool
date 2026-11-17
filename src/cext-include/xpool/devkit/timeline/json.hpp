#pragma once

/// \file xpool/devkit/timeline/json.hpp
/// \brief Bounded streaming JSON output for metadata and compatibility artifacts.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace xpool::devkit::timeline {

/// Synchronous borrowed destination. Each callback consumes its bytes before
/// returning; views expire on return. Exceptions propagate to the file owner,
/// which retains and charges any partial temporary file. A null callback counts
/// exact output bytes without retaining them, for disk admission before create.
struct JsonSink {
  /// Destination owner, alive through serialization.
  void *context = nullptr;
  /// Optional synchronous write operation; may throw on an incomplete write.
  void (*write)(void *, std::string_view) = nullptr;
};

namespace detail {

/// \cond INTERNAL
// A small stack buffer is part of the declared Writer worker stack. It does
// not grow with the snapshot and can also calculate its exact serialized size.
class JsonOutput {
public:
  explicit JsonOutput(JsonSink sink) : sink_(sink) {}
  void text(std::string_view value);
  void number(std::uint64_t value);
  void nullable(std::optional<std::uint64_t> value);
  void quoted(std::string_view value);
  void key(std::string_view value);
  std::size_t finish();

private:
  void flush();
  const JsonSink sink_;
  char storage_[4096]{};
  std::size_t used_ = 0;
  std::size_t total_ = 0;
};
/// \endcond

} // namespace detail
} // namespace xpool::devkit::timeline
