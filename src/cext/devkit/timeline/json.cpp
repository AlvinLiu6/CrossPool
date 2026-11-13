#include <xpool/devkit/timeline/json.hpp>

#include <algorithm>
#include <charconv>
#include <limits>
#include <stdexcept>

namespace xpool::devkit::timeline::detail {

void JsonOutput::flush() {
  if (used_ > std::numeric_limits<std::size_t>::max() - total_)
    throw std::length_error("xpool snapshot JSON size overflows");
  if (used_ != 0 && sink_.write)
    sink_.write(sink_.context, {storage_, used_});
  total_ += used_;
  used_ = 0;
}

void JsonOutput::text(std::string_view value) {
  while (!value.empty()) {
    const auto count = std::min(value.size(), sizeof(storage_) - used_);
    std::copy_n(value.data(), count, storage_ + used_);
    used_ += count;
    value.remove_prefix(count);
    if (used_ == sizeof(storage_))
      flush();
  }
}

void JsonOutput::number(std::uint64_t value) {
  char storage[20];
  const auto result = std::to_chars(storage, storage + sizeof(storage), value);
  text({storage, static_cast<std::size_t>(result.ptr - storage)});
}

void JsonOutput::nullable(std::optional<std::uint64_t> value) {
  if (value)
    number(*value);
  else
    text("null");
}

void JsonOutput::quoted(std::string_view value) {
  constexpr auto hex = std::string_view{"0123456789abcdef"};
  text("\"");
  for (const auto character : value) {
    const auto byte = static_cast<unsigned char>(character);
    if (character == '"' || character == '\\') {
      const char escaped[] = {'\\', character};
      text({escaped, sizeof(escaped)});
    } else if (byte < 32) {
      const char escaped[] = {'\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 15]};
      text({escaped, sizeof(escaped)});
    } else
      text({&character, 1});
  }
  text("\"");
}

void JsonOutput::key(std::string_view value) {
  quoted(value);
  text(":");
}

std::size_t JsonOutput::finish() {
  flush();
  return total_;
}

} // namespace xpool::devkit::timeline::detail
