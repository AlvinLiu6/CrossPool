#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>
#include <unistd.h>

#include <xpool/debug/options.hpp>
#include <xpool/devkit/timeline/attachments.hpp>
#include <xpool/devkit/timeline/preparation.hpp>

namespace timeline = xpool::devkit::timeline;

TEST(TimelineAttachmentTest, PublishesStructureAndRegisteredTemplateOnSharedWriter) {
  auto directory = std::array<char, 64>{};
  const auto pattern = std::string{"/tmp/xpool-b1-attachments-XXXXXX"};
  std::copy(pattern.begin(), pattern.end(), directory.begin());
  ASSERT_NE(::mkdtemp(directory.data()), nullptr);
  const auto outdir = std::filesystem::path{directory.data()};
  auto observers = xpool::debug::Options{};
  observers.graph_observer.enable = true;
  auto &preparation = timeline::ProcessPreparation::singleton();
  const auto policy = timeline::make_collection_policy({}, observers);
  preparation.configure(policy);
  const auto budgets = preparation.budgets();
  ASSERT_TRUE(budgets.has_value());
  const auto run = timeline::parse_identity_uuid("11111111-1111-4111-8111-111111111111");
  const auto host_source = timeline::BatchSource{run, 1, 1};
  const auto device_source = timeline::BatchSource{run, 2, 2};
  const auto process = timeline::RegisteredProcess{
      .process = {.creation_id = timeline::parse_identity_uuid("22222222-2222-4222-8222-222222222222"),
                  .host_creation_id = timeline::parse_identity_uuid("33333333-3333-4333-8333-333333333333"),
                  .pid = static_cast<std::uint64_t>(::getpid()),
                  .runtime_role = xpool::RuntimeRole::FfnAgent},
      .host = host_source};
  const auto capabilities = timeline::MetadataCapabilities{.graph = timeline::MetadataCapability::Enabled};
  const auto host = timeline::MetadataRegistry::create(budgets->host, host_source, policy, capabilities);
  const auto device = timeline::MetadataRegistry::create(budgets->host, device_source, policy, capabilities);
  for (const auto &registry : {host, device})
    registry->add_device(process, device_source, 0, "GPU-44444444-4444-4444-8444-444444444444", 1);
  timeline::prepare_attachments(host, device, 0, 1, outdir.native(), "", {});
  auto primary = timeline::make_graph_structure();
  primary.node_counts[cudaGraphNodeTypeKernel] = 2;
  primary.binding_site_count = 3;
  timeline::submit_graph_structure(std::move(primary), false, 0);
  auto lane = timeline::make_graph_structure();
  lane.node_counts[cudaGraphNodeTypeGraph] = 1;
  lane.compute_branch_count = 1;
  lane.delivery_branch_count = 2;
  timeline::submit_graph_structure(std::move(lane), true, 0);
  ASSERT_TRUE(timeline::finish_attachments(true));
  const auto root = outdir / "timeline.11111111-1111-4111-8111-111111111111";
  EXPECT_TRUE(std::filesystem::exists(root / "graph/1.1.json"));
  EXPECT_TRUE(std::filesystem::exists(root / "graph/1.2.json"));
  EXPECT_TRUE(std::filesystem::exists(root / "chunks/1/1.xpool"));
  auto found = false;
  for (const auto &entry : std::filesystem::directory_iterator{root}) {
    if (!entry.path().filename().string().starts_with("metadata."))
      continue;
    auto stream = std::ifstream{entry.path()};
    const auto text = std::string{std::istreambuf_iterator<char>{stream}, {}};
    found = found || (text.find("graph_structure") != text.npos && text.find("compute_templates") != text.npos);
  }
  EXPECT_TRUE(found);
  std::filesystem::remove_all(outdir);
}
