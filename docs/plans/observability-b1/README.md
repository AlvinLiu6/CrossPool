# Bounded Event Collection And Background Export

## Goal

Collect immutable Transport and Fabric events during execution with bounded
process storage and an independent Writer. Preserve production behavior and
retirement proofs. Use the C0 v7 contract and reference fixtures from
[C0 v7](https://github.com/saitewasreset/CrossPool/blob/feat/observability_c0/docs/plans/unified-timeline-observability/shared-contract.md), including its Device allocation and Transfer
failure rules.

## Accepted Changes

- Use v7 artifacts and IPC only, XPOOLC07, 256-byte Events, zero reserved
  bytes 136–143 and forbidden presence bit 8.
- Endpoint JSON excludes internal Arena handle suffixes. Retain suffixes only
  for internal pairing and equal-retry validation.
- Preserve each Transport boundary's actual result, including successful
  ResultObserved and OutputCopied followed by Closed(Shutdown). Offline
  consumers expose a typed read-only boundary-to-result mapping, not one
  aggregate result_code. Different results at the same boundary still conflict;
  different stable request facts across boundaries still conflict.
- Remove legacy Transport/Fabric Record reconstruction, getters, snapshot
  exports, evidence reports, local trace IDs and snapshot counters.
- Keep Graph structures and Routing tensors as attachments; remove their
  legacy getter/export contracts. Preserve PrefillLogitObserver.
- Charge every actual active allocation once, including alignment, IPC, current
  Hook state and transfer storage. Reject incomplete, overflowing or excessive
  requirements before activation.
- Preserve first stop reason and bounded diagnostic, track uncertain Transfer
  completion independently, and retain all affected owners and charges until
  actual retirement. Preserve raw invalid Transport values and actual local
  terminal facts without synthesizing missing evidence.

## Attachment Interfaces And Ownership

Runtime preparation passes typed Host/Device metadata registries and actual
Routing MemoryDemand into the shared process service. Device and Host
copy/serialization requirements derive from the placed Fabric projection before
collection starts; the same reservations transfer to the allocating owners.
Dense and non-Router participants declare Routing not_applicable.

Graph construction retains actual node inventories and binding/branch counts.
Register fresh Host-owned templates and Lane-to-compute references at their
owning construction boundaries. Immutable structure jobs publish on the shared
Writer and register their relative paths and SHA-256 in metadata. SGLang Host
calls emit v7 Events with distinct capture/template and execution identities;
Hook failure diagnostics are bounded and reported outside execution Hooks.

Routing publication allocates a Device-owned attachment identity independently
of Event/tensor capacity and emits kind 550 at the actual Hook. Retain original
Invocation, Lane/Lease, layer, clock and time. After actual drain, copy payloads
into admitted Host storage, then hand off once. The Writer invokes the installed
Safetensors library, publishes tensor files and final attachment outcomes, and
registers artifact metadata. Event artifact_id stays zero until publication;
attachment identity resolves the closing index. No getter or snapshot mailbox
is exposed. Uncertain copies retain the original targets and all charges.

## Collection Directories

Independent mode respects each enabled Observer's configured outdir. Each contains
`timeline.<run_uuid>/`, with `chunks/<producer>/<sequence>.xpool`, immutable
metadata and quality files, and that class's attachments. Equal directories
coalesce publications. All roots share one process disk ledger, one Collector
thread and one Writer thread. Chunk sequences remain unique per Source across
roots; copied/exported totals count each Event once. Metadata and cumulative
Source quality are replicated byte-for-byte into each Source's registered roots,
with every physical copy charged to the shared disk quota.

For offline tools, configure class outdirs as siblings beneath a common capture
root and use an explicit A3 input manifest listing their relative paths. Set its
single attachment_root to the Routing Run directory; tensor/index artifact paths
are relative to that directory. Final attachment index entries are supplied in
the manifest's attachment_indices; A3 does not discover files or import index
files automatically. Graph structures remain relative to the Graph Run directory.
Arbitrary independent outdirs remain valid collection configuration, but require
an explicit common input root or offline input organization for A3.

## Validation

Native tests cover independent v7 vectors, reserved fields, anomaly evidence,
budget admission and Transfer uncertainty. Python tests cover configuration and
runtime wiring. Use xtest, Ruff, ty, clang-format, Doxygen and Native Host
ASAN/UBSAN. Real multi-GPU qualification remains distinct from CPU evidence.

## Boundaries

Full B2 association, Dynamic Context and calibration providers are outside B1.
Unified mode must reject preparation until those providers are connected.
Independent observation remains useful with explicit unavailable capabilities.
Keep bounded collection/Writer ownership and actual CUDA/IPC retirement proofs;
never block production Hooks on collection capacity or file I/O.
