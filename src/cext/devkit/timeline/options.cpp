#include <xpool/devkit/timeline/options.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <vector>

#include <xpool/devkit/fabric_observer.hpp>
#include <xpool/devkit/timeline/chunk.hpp>
#include <xpool/devkit/timeline/error.hpp>
#include <xpool/devkit/transport_observer.hpp>
#include <xpool/utils/checked.hpp>

namespace xpool::devkit::timeline {

std::size_t HostRequirements::required_bytes(std::uint64_t producer) const {
  const auto values = std::array{events,  control, transfer,   delivery,     writer,       metadata,
                                 threads, legacy,  provenance, read_staging, serialization};
  const auto names = std::array{"events",  "control", "transfer",   "delivery",     "writer",       "metadata",
                                "threads", "legacy",  "provenance", "read_staging", "serialization"};
  auto total = std::size_t{0};
  for (auto index = std::size_t{0}; index < values.size(); ++index) {
    if (!values[index])
      throw CollectionError(CollectionErrorCode::ResourceBudget, producer,
                            std::string{"xpool missing host requirement: "} + names[index]);
    try {
      total = xpool::utils::checked::sum(total, *values[index]);
    } catch (const c10::Error &cause) {
      throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool host requirement sum overflows",
                            std::nullopt, cause.what_without_backtrace());
    }
  }
  return total;
}

std::size_t HostRequirements::admit(std::size_t limit, std::uint64_t producer) const {
  const auto total = required_bytes(producer);
  if (limit == 0 || total > limit)
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer,
                          "xpool host requirements exceed budget: required=" + std::to_string(total) +
                              " limit=" + std::to_string(limit));
  return total;
}

LegacyMemorySizes LegacyMemoryDemand::sizes(std::uint64_t producer) const {
  if (kind != LegacyKind::Transport && kind != LegacyKind::Fabric)
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool invalid legacy requirement kind");
  if (domains == 0)
    return {};
  if (capacity == 0 || readers_per_domain == 0 || provenance_bytes_per_operation == 0 ||
      report_bytes_per_operation == 0 || report_fixed_bytes == 0)
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool incomplete legacy requirements");
  try {
    const auto transport = kind == LegacyKind::Transport;
    const auto slot_bytes = transport ? std::size_t{256} : std::size_t{384};
    const auto record_bytes = transport ? sizeof(transport_observer::Record) : sizeof(fabric_observer::Record);
    const auto vector_bytes =
        transport ? sizeof(std::vector<transport_observer::Record>) : sizeof(std::vector<fabric_observer::Record>);
    const auto records = xpool::utils::checked::prod(capacity, record_bytes);
    const auto report = xpool::utils::checked::sum(report_fixed_bytes,
                                                   xpool::utils::checked::prod(capacity, report_bytes_per_operation));
    const auto one_read = xpool::utils::checked::sum(vector_bytes, records, report);
    return {.slabs = xpool::utils::checked::prod(domains, capacity, slot_bytes),
            .provenance = xpool::utils::checked::prod(domains, capacity, provenance_bytes_per_operation),
            .read_staging = xpool::utils::checked::prod(domains, readers_per_domain, one_read)};
  } catch (const c10::Error &cause) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool legacy requirement size overflows",
                          std::nullopt, cause.what_without_backtrace());
  }
}

std::size_t Options::logical_device_minimum() const {
  try {
    return xpool::utils::checked::sum(xpool::utils::checked::prod(device_event_slot_capacity, std::size_t{272}),
                                      xpool::utils::checked::prod(endpoint_capacity, std::size_t{8}),
                                      xpool::utils::checked::prod(context_slot_capacity, std::size_t{40}),
                                      xpool::utils::checked::prod(sampler_slot_capacity, std::size_t{8}),
                                      std::size_t{256});
  } catch (const c10::Error &cause) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool logical device minimum overflows",
                          std::nullopt, cause.what_without_backtrace());
  }
}

void Options::validate() const {
  const auto positive = std::array{device_bytes,
                                   host_bytes,
                                   chunk_bytes,
                                   disk_bytes,
                                   metadata_reserve_bytes,
                                   device_event_slot_capacity,
                                   endpoint_capacity,
                                   context_slot_capacity,
                                   sampler_slot_capacity};
  if (enable != outdir.has_value() || (outdir && (!outdir->is_absolute() || outdir->empty())) ||
      std::ranges::any_of(positive, [](auto value) { return value == 0; }) || calibration_block_attempts != 10 ||
      chunk_bytes < Chunk::required_bytes(1) || chunk_bytes > Chunk::maximum_bytes || shutdown_wait_ms == 0 ||
      shutdown_wait_ms > 5000 || collection_interval_ms == 0 ||
      collection_interval_ms > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      metadata_reserve_bytes >= disk_bytes || disk_bytes - metadata_reserve_bytes < chunk_bytes)
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool timeline policy violates resource bounds");
  if (enable && logical_device_minimum() > device_bytes)
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool device budget is below C0 logical minimum");
}

void Options::require_same_policy(const Options &peer) const {
  validate();
  peer.validate();
  if (*this != peer)
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool timeline participant policy mismatch");
}

} // namespace xpool::devkit::timeline
