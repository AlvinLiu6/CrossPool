#pragma once

/// \file xpool/devkit/timeline/legacyaccess.hpp
/// \brief Internal Host materialization into unchanged legacy Record values.

#include <optional>

#include <xpool/devkit/fabric_observer.hpp>
#include <xpool/devkit/timeline/legacy.hpp>
#include <xpool/devkit/transport_observer.hpp>

namespace xpool::devkit::timeline::detail {

/// \cond INTERNAL
struct LegacyRecordAccess {
  // Defined only for the two existing native observer Record types. Input is
  // a same-cutoff, conflict-free observation with explicit availability masks.
  template <typename Record>
  static std::optional<Record> project(const LegacyOperation &operation, const LegacyScope &scope);
};
template <>
std::optional<xpool::devkit::transport_observer::Record>
LegacyRecordAccess::project<xpool::devkit::transport_observer::Record>(const LegacyOperation &, const LegacyScope &);
template <>
std::optional<xpool::devkit::fabric_observer::Record>
LegacyRecordAccess::project<xpool::devkit::fabric_observer::Record>(const LegacyOperation &, const LegacyScope &);
/// \endcond

} // namespace xpool::devkit::timeline::detail
