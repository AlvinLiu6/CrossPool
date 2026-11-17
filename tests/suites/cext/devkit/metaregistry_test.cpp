#include <string>

#include <gtest/gtest.h>

#include <xpool/devkit/timeline/metaregistry.hpp>

namespace timeline = xpool::devkit::timeline;

TEST(TimelineMetadataTest, EndpointSerializationExcludesInternalArenaSuffixAndRetainsPairingChecks) {
  const auto budget = timeline::MemoryBudget::create(32 << 20);
  const auto run = timeline::parse_identity_uuid("11111111-1111-4111-8111-111111111111");
  const auto host = timeline::parse_identity_uuid("55555555-5555-4555-8555-555555555555");
  const auto source = timeline::BatchSource{run, 3, 2};
  const auto metadata =
      timeline::MetadataRegistry::create(budget, source, timeline::make_collection_policy({}, {}), {});
  const auto process = [&](std::string_view creation, std::uint64_t pid, xpool::RuntimeRole role,
                           std::uint64_t producer) {
    return timeline::RegisteredProcess{.process = {.creation_id = timeline::parse_identity_uuid(creation),
                                                   .host_creation_id = host,
                                                   .pid = pid,
                                                   .runtime_role = role},
                                       .host = {run, producer, 1}};
  };
  metadata->add_device(process("22222222-2222-4222-8222-222222222222", 101, xpool::RuntimeRole::Instance, 1), source, 0,
                       "GPU-fixture-attention", 1);
  metadata->add_device(process("33333333-3333-4333-8333-333333333333", 102, xpool::RuntimeRole::AtnAgent, 9),
                       {run, 2, 3}, 0, "GPU-fixture-attention", 1);
  metadata->add_device(process("44444444-4444-4444-8444-444444444444", 103, xpool::RuntimeRole::FfnAgent, 4),
                       {run, 5, 4}, 0, "GPU-fixture-ffn", 1);
  metadata->add_generation(1, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 1, 1, true);
  metadata->add_model(1, 0, "test/test-model", 1, 1);
  metadata->add_pe_member(1, 0, 9, 2);
  metadata->add_pe_member(1, 1, 4, 5);
  metadata->add_endpoint(2, 1, 1, 0, 0, "test/test-model", "0123456789abcdef", 3);
  const auto bytes = metadata->document(1);
  const auto text = std::string{reinterpret_cast<const char *>(bytes.data()), bytes.size()};
  EXPECT_NE(text.find("\"contract_version\":7,"), text.npos);
  EXPECT_EQ(text.find("arena_handle_suffix"), text.npos);
  EXPECT_NE(
      text.find(
          R"("endpoints":[{"owner_producer":2,"endpoint_sequence":1,"generation":1,"instance_index":0,"instance_rank":0,"model_id":"test/test-model","sites":[{"producer":3,"site":"instance"},{"producer":2,"site":"atnagent"}]}])"),
      text.npos);
  EXPECT_NO_THROW(metadata->add_endpoint(2, 1, 1, 0, 0, "test/test-model", "0123456789abcdef", 3));
  EXPECT_THROW(metadata->add_endpoint(2, 1, 1, 0, 0, "test/test-model", "fedcba9876543210", 3),
               timeline::CollectionError);
}
