#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include <xpool/devkit/timeline/hostchannel.hpp>
#include <xpool/devkit/timeline/routingevent.hpp>
#include <xpool/devkit/timeline/writer.hpp>

namespace timeline = xpool::devkit::timeline;

namespace {
std::string read(const std::filesystem::path &path) {
  auto stream = std::ifstream{path, std::ios::binary};
  return {std::istreambuf_iterator<char>{stream}, {}};
}
} // namespace

TEST(TimelineWriterTest, RoutesMixedBatchesWithSharedIdentitiesAndExactPrefixAccounting) {
  for (const auto fail_second : {false, true}) {
    auto directory = std::array<char, 64>{};
    const auto pattern = std::string{"/tmp/xpool-writer-XXXXXX"};
    std::copy(pattern.begin(), pattern.end(), directory.begin());
    ASSERT_NE(::mkdtemp(directory.data()), nullptr);
    const auto root = std::filesystem::path{directory.data()};
    const auto graph = root / "graph";
    const auto routing = root / "routing";
    const auto host = timeline::MemoryBudget::create(4 << 20);
    auto disk = timeline::DiskBudget{1 << 20, 4096};
    const auto source = timeline::BatchSource{.producer = 5, .clock_domain_id = 4};
    const auto catalog = timeline::MetadataCatalog::create(host, source, 4, 1024);
    const auto document = std::string{"{}\n"};
    const auto metadata = catalog->append(1, std::as_bytes(std::span{document}));
    auto info = timeline::BatchInfo{.metadata_revision = 1};
    std::copy_n(metadata.sha256.begin(), info.metadata_sha256_prefix.size(), info.metadata_sha256_prefix.begin());
    const auto channel = timeline::HostChannel::create(host, source, info, 4, 2, 4);
    auto writer = timeline::RawWriter{source, graph, *host, disk};
    writer.register_directory(timeline::EvidenceClass::Graph, graph);
    writer.publish_pending_metadata(*catalog);
    // Joining after an earlier publication must replicate the exact metadata.
    writer.register_directory(timeline::EvidenceClass::Routing, routing);
    writer.register_directory(timeline::EvidenceClass::Routing, routing);
    EXPECT_THROW(writer.register_directory(timeline::EvidenceClass::Routing, root / "changed"),
                 timeline::CollectionError);
    if (fail_second) {
      std::filesystem::create_directories(routing);
      auto block = std::ofstream{routing / "chunks"};
      block << "blocked";
    }
    auto reservation = channel->collector().batches().try_reserve();
    ASSERT_EQ(reservation.status, timeline::BatchReserveStatus::Reserved);
    const auto graph_event = timeline::Event{.clock_domain_id = 4,
                                             .kind = 500,
                                             .presence = 1U << 3,
                                             .graph_owner = 5,
                                             .graph_template = 1,
                                             .role = 1,
                                             .data0 = 1,
                                             .data1 = 1};
    const auto execution =
        xpool::fabric::LaneExecution{.key = {.instance_index = 0, .invocation_sequence = 3},
                                     .executor_lease_sequence = 7,
                                     .layer_ordinal = 2,
                                     .payload_rows = 2,
                                     .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete};
    reservation.events[0] = graph_event;
    reservation.events[1] =
        timeline::make_routing_event(source, 1, 1, execution, 0, 19, 2, timeline::RoutingAttachmentState::Reserved, 0);
    reservation.events[2] = graph_event;
    for (auto index = std::size_t{0}; index < 3; ++index)
      reservation.events[index].event_sequence = index + 1;
    ASSERT_EQ(channel->collector().batches().publish(reservation.token, 3, info),
              timeline::BatchPublishStatus::Published);
    if (fail_second)
      EXPECT_THROW(writer.write_one(channel->collector(), catalog.get()), timeline::CollectionError);
    else
      EXPECT_TRUE(writer.write_one(channel->collector(), catalog.get()));
    EXPECT_EQ(read(graph / "metadata.5.1.json"), document);
    EXPECT_EQ(read(routing / "metadata.5.1.json"), document);
    EXPECT_TRUE(std::filesystem::exists(graph / "chunks/5/1.xpool"));
    EXPECT_EQ(std::filesystem::exists(routing / "chunks/5/2.xpool"), !fail_second);
    EXPECT_EQ(std::filesystem::exists(graph / "chunks/5/3.xpool"), !fail_second);
    const auto report = timeline::quality(channel->collector());
    EXPECT_EQ(report.counters.exported, fail_second ? 1 : 3);
    EXPECT_EQ(report.counters.post_commit_lost, fail_second ? 2 : 0);
    if (!fail_second) {
      writer.publish_quality(report);
      EXPECT_EQ(read(graph / "quality.5.1.json"), read(routing / "quality.5.1.json"));
      const auto payload = std::string{"tensor"};
      writer.publish_attachment("routing/5.safetensors", std::as_bytes(std::span{payload}));
      EXPECT_EQ(read(routing / "routing/5.safetensors"), payload);
      EXPECT_FALSE(std::filesystem::exists(graph / "routing/5.safetensors"));
    }
    auto bytes = std::uint64_t{0};
    for (const auto &entry : std::filesystem::recursive_directory_iterator{root})
      if (entry.is_regular_file())
        bytes += entry.file_size();
    EXPECT_EQ(disk.report().used, bytes - (fail_second ? 7 : 0));
    std::filesystem::remove_all(root);
  }
}
