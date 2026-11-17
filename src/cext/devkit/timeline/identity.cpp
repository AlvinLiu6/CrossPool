#include <xpool/devkit/timeline/identity.hpp>

#include <algorithm>
#include <cerrno>
#include <limits>
#include <stdexcept>
#include <system_error>

#include <sys/random.h>

namespace xpool::devkit::timeline {
namespace {

[[noreturn]] void invalid_identity(std::string_view reason, std::uint64_t producer = 0) {
  throw CollectionError(CollectionErrorCode::MetadataUnavailable, producer, std::string{reason});
}

IdentityUuid fresh_process_uuid() {
  auto result = IdentityUuid{};
  auto filled = std::size_t{0};
  while (filled < result.size()) {
    const auto count = ::getrandom(result.data() + filled, result.size() - filled, 0);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      throw CollectionError(CollectionErrorCode::Allocation, 0, "xpool process UUID entropy unavailable", std::nullopt,
                            count < 0 ? std::system_category().message(errno) : "empty random read");
    filled += static_cast<std::size_t>(count);
  }
  result[6] = (result[6] & std::byte{0x0f}) | std::byte{0x40};
  result[8] = (result[8] & std::byte{0x3f}) | std::byte{0x80};
  return result;
}

} // namespace

IdentityUuid parse_identity_uuid(std::string_view text) {
  const auto invalid = [] { throw std::invalid_argument("xpool identity requires canonical lowercase UUID text"); };
  if (text.size() != 36)
    invalid();
  auto result = IdentityUuid{};
  auto index = std::size_t{0};
  constexpr auto digits = std::string_view{"0123456789abcdef"};
  for (auto output = std::size_t{0}; output < result.size(); ++output) {
    if (index == 8 || index == 13 || index == 18 || index == 23)
      if (text[index++] != '-')
        invalid();
    const auto upper = digits.find(text[index++]);
    const auto lower = digits.find(text[index++]);
    if (upper == std::string_view::npos || lower == std::string_view::npos)
      invalid();
    result[output] = static_cast<std::byte>(upper * 16 + lower);
  }
  return result;
}

std::array<char, 36> format_identity_uuid(const IdentityUuid &bytes) noexcept {
  constexpr auto digits = std::string_view{"0123456789abcdef"};
  auto result = std::array<char, 36>{};
  auto output = std::size_t{0};
  for (auto index = std::size_t{0}; index < bytes.size(); ++index) {
    if (index == 4 || index == 6 || index == 8 || index == 10)
      result[output++] = '-';
    const auto value = std::to_integer<unsigned>(bytes[index]);
    result[output++] = digits[value >> 4];
    result[output++] = digits[value & 15];
  }
  return result;
}

std::shared_ptr<IdentityDirectory> IdentityDirectory::create(std::shared_ptr<MemoryBudget> host, IdentityUuid run,
                                                             std::size_t processes, std::size_t devices,
                                                             std::size_t generations) {
  constexpr auto maximum = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (!host || processes == 0 || devices == 0 || generations == 0 || processes > maximum / sizeof(RegisteredProcess) ||
      devices > maximum / sizeof(RegisteredDevice) || generations > maximum / sizeof(RegisteredGeneration))
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool identity directory geometry is invalid");
  try {
    return make_budgeted_shared<IdentityDirectory>(host, MemoryKind::Metadata, 0, ConstructionKey{}, host, run,
                                                   processes, devices, generations);
  } catch (const CollectionError &) {
    throw;
  } catch (const std::exception &error) {
    throw CollectionError(CollectionErrorCode::Allocation, 0, "xpool identity directory allocation failed",
                          std::nullopt, error.what());
  }
}

IdentityDirectory::IdentityDirectory(ConstructionKey, std::shared_ptr<MemoryBudget> host, IdentityUuid run,
                                     std::size_t processes, std::size_t devices, std::size_t generations)
    : host_(std::move(host)), run_(run), resource_(*host_, MemoryKind::Metadata), processes_(&resource_),
      host_clocks_(&resource_), devices_(&resource_), generations_(&resource_), process_capacity_(processes),
      device_capacity_(devices), generation_capacity_(generations) {
  processes_.reserve(processes);
  host_clocks_.reserve(processes);
  devices_.reserve(devices);
  generations_.reserve(generations);
}

void IdentityDirectory::require_admission() const {
  if (stopped_)
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool identity directory admission is closed");
}

std::uint64_t IdentityDirectory::next(AttemptSequence &sequence) {
  const auto id = sequence.try_next();
  if (id == 0 || id == UINT64_MAX) {
    stopped_ = true;
    exhausted_ = true;
  }
  if (id == 0)
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool preparation identity namespace exhausted");
  return id;
}

RegisteredProcess IdentityDirectory::assign_process(IdentityUuid host_creation, std::uint64_t pid,
                                                    std::uint64_t start_key, xpool::RuntimeRole role) {
  const auto lock = std::lock_guard{mutex_};
  if (pid == 0 || start_key == 0 || !xpool::is_valid(role) || role == xpool::RuntimeRole::Daemon)
    invalid_identity("xpool process assignment requires actual pid, creation key and role");
  for (const auto &entry : processes_)
    if (entry.process.host_creation_id == host_creation && entry.process.pid == pid && entry.start_key == start_key) {
      if (entry.process.runtime_role != role)
        invalid_identity("xpool actual process creation cannot change runtime role", entry.host.producer);
      return entry;
    }
  require_admission();
  if (processes_.size() == process_capacity_)
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool process identity capacity exhausted");
  const auto creation = fresh_process_uuid();
  if (std::ranges::any_of(processes_, [&](const auto &entry) { return entry.process.creation_id == creation; }))
    invalid_identity("xpool fresh process UUID collides with a retained creation identity");
  const auto process = ProcessIdentity{creation, host_creation, pid, role};
  auto clock = std::uint64_t{0};
  for (const auto &entry : host_clocks_)
    if (entry.host_creation_id == process.host_creation_id)
      clock = entry.id;
  const auto new_clock = clock == 0;
  if (new_clock)
    clock = next(clocks_);
  const auto result = RegisteredProcess{process, BatchSource{run_, next(producers_), clock}, start_key};
  if (new_clock)
    host_clocks_.push_back({clock, process.host_creation_id});
  processes_.push_back(result);
  return result;
}

BatchSource IdentityDirectory::register_device(IdentityUuid process, std::uint32_t logical_device,
                                               std::string_view physical_device_uuid, std::uint64_t raw_epoch) {
  const auto lock = std::lock_guard{mutex_};
  const auto owner =
      std::ranges::find(processes_, process, [](const auto &entry) { return entry.process.creation_id; });
  if (owner == processes_.end() || owner->process.runtime_role == xpool::RuntimeRole::Daemon ||
      physical_device_uuid.empty() || physical_device_uuid.find('\0') != std::string_view::npos || raw_epoch == 0)
    invalid_identity("xpool Device registration requires an actual Device process, UUID and epoch");
  for (const auto &entry : devices_)
    if (entry.process_creation_id == process && entry.logical_device == logical_device) {
      if (entry.raw_epoch != raw_epoch || entry.physical_device_uuid != physical_device_uuid)
        invalid_identity("xpool Device Producer clock/mapping facts cannot change", entry.source.producer);
      return entry.source;
    }
  require_admission();
  if (devices_.size() == device_capacity_)
    throw CollectionError(CollectionErrorCode::ResourceBudget, owner->host.producer,
                          "xpool Device identity capacity exhausted");
  auto entry =
      RegisteredDevice{process, {}, logical_device, raw_epoch, std::pmr::string{physical_device_uuid, &resource_}};
  const auto source = BatchSource{run_, next(producers_), next(clocks_)};
  entry.source = source;
  devices_.push_back(std::move(entry));
  return source;
}

std::uint64_t IdentityDirectory::register_generation(std::string_view fabric_generation) {
  const auto lock = std::lock_guard{mutex_};
  if (fabric_generation.empty() || fabric_generation.find('\0') != std::string_view::npos)
    invalid_identity("xpool observation Generation requires its exact production identity");
  for (const auto &entry : generations_)
    if (entry.fabric_generation == fabric_generation)
      return entry.id;
  require_admission();
  if (generations_.size() == generation_capacity_)
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool Generation identity capacity exhausted");
  auto entry = RegisteredGeneration{0, std::pmr::string{fabric_generation, &resource_}};
  entry.id = next(generation_ids_);
  const auto result = entry.id;
  generations_.push_back(std::move(entry));
  return result;
}

IdentityDefinitions IdentityDirectory::definitions() const {
  const auto lock = std::lock_guard{mutex_};
  return {shared_from_this(), processes_, host_clocks_, devices_, generations_};
}

void IdentityDirectory::stop() {
  const auto lock = std::lock_guard{mutex_};
  stopped_ = true;
}

bool IdentityDirectory::identity_exhausted() const {
  const auto lock = std::lock_guard{mutex_};
  return exhausted_;
}

} // namespace xpool::devkit::timeline
