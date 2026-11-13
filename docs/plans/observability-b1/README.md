# Bounded Event Collection, Background Export And Legacy Snapshots

## Goal

Replace mutable Transport/Fabric observation histories with bounded immutable
events, collect them during execution, publish versioned raw artifacts on an
independent Writer, and reconstruct the existing legacy projections in Native
Host storage. Preserve production control flow and original snapshot read
preconditions. Host Graph lifecycle logging must also leave the execution path.

This plan owns the target delta and the accepted resource and failure rules
below. Remaining shared-contract questions gate only their dependent portions.

The scoped delta includes a narrow B2 prerequisite for shared Transport
Operation identity. It does not include full B2 association or dynamic Graph
Context. Its storage and ordering follow the fixed C0 contract below.

## Baseline

The source baseline is `ec01247e09b143c0eef239868cb8bc58d72c9a5f`.
[Devkit](../../designs/devkit.md), [Transport](../../designs/transport.md),
[Fabric](../../designs/fabric.md), [FFN execution](../../designs/ffn-execution.md)
and [Control Plane](../../designs/control-plane.md) own current behavior.

The C0 issue names `21554ea78533590ff2f122ce05f91b485d6da156`.
The selected baseline is nine commits later. Source comparison leaves Native
Devkit, `trace.cuh`, Python Devkit and SGLang Graph Observer unchanged in this
range; configuration, generation retirement and installed tooling use the
selected baseline's owners.

The fixed external specification is the
[C0 v4 shared contract](https://github.com/saitewasreset/CrossPool/blob/e7a61aaa741d43ffb7ea6caefd51e690ccfd5cec/docs/plans/unified-timeline-observability/shared-contract.md),
with its sibling fixtures and calibration specification. It owns wire values,
identities, presence, metadata, quality, Native collection declarations and
legacy evidence reports. The locally authorized [v5 anomaly candidate](transport-anomaly-v5.md)
extends this baseline; new B1 artifacts and IPC use v5 and reject mixed versions.
This is not evidence of A/B2 adoption or joint C0 release. Do not infer wire layout from native
object padding or regenerate expected fixtures using the implementation under
test.

Current `utils::trace::BufferView<Record>` reserves an append-only mutable
record with relaxed ordering. Reservation is not publication. Transport/Fabric
Host reads follow quiescence or synchronization. SGLang Graph logging writes
and flushes each event synchronously. Production Fabric drain includes PE-wide
outbound NVSHMEM quiet after local producer completion; it does not establish
remote detach or complete-world retirement.

## Proposed Changes

### Event storage and ownership

Declare the 256-byte Event and explicit C0 statuses, errors and quality values
under `xpool::devkit::timeline`. Guard its size and wire offsets; encode and
decode fixed-width fields explicitly in little-endian order. Run/Producer are
supplied by the chunk. Presence distinguishes an absent field from a valid zero.

The format layer uses `Event::encode/decode` for exact field bytes and
`Event::validate` for intrinsic vocabulary/presence/family facts. `ChunkMetadata`
borrows exact metadata bytes and registered source identity from the metadata
owner; it is not a JSON parser or a schema-validation result produced by the
byte codec. `Chunk::encode` writes into caller storage. `Chunk::decode` verifies
header layout/CRC, declared length, supplied metadata identity/hash, payload
CRC, intrinsic Events and order before writing caller-owned Event output.
Returned spans borrow that output; failures leave it unchanged. Metadata JSON,
Generation membership and cross-reference semantics remain the metadata owner's
responsibility. Integrate those checks at the prescribed C0 parse stage before
claiming complete strict-reader acceptance. Low-level caller geometry/alias
misuse raises `std::invalid_argument`; resource exhaustion and malformed bytes
have structured collection/parse errors. Public collection owners translate
low-level programming failures at their own typed boundary.

Use a fixed Host pool and a Device pool per process/Device Producer. A slot has
an Event payload, a generation and atomic ownership. The ownership path is
Free, Reserved, Committed, Copying, then Free at the next generation, with
private exclusive transition states for writing and recycling. A write token
contains slot, generation and attempted-event sequence. Verify ownership before
accessing payload; Reserved slots cannot be reclaimed. Commit publishes the
entire immutable Event with release ordering; acquire claims only committed
generations. Abandon retires an uncommitted reservation without publishing.

Reservation examines a bounded number of candidates with no wait for a free
slot or another producer. Full drops new evidence. Allocate the positive
attempted-event sequence independently of slot acquisition, including Full;
Stopped allocates none. Existing token owners can retire after stop. Identity
exhaustion stops affected observation without wrapping or aborting production;
quality counter saturation makes totals inexact. Pending lower sequences do
not prevent consumption of later commits.

Use Host atomics for Host pools and Device-scope CUDA atomics for Device pools.
Host code never treats a live Device ownership word as a consistent snapshot.
Keep original operation counters separate from Event attempts and snapshot
capacity exclusions. Hook adapters retain only current per-Endpoint/Instance/Lane
facts needed to repeat already known information; they keep no second Device
operation history.

### Process owner, collection and handoff

One process collection owner holds resource ledgers, fixed source/domain
tables, one Collector thread and one independent Writer thread. Observed runtime
roles are Instance, AtnAgent and FfnAgent. The Daemon distributes registered
identities and brokers descriptors; it owns no observation source or worker.
Coordinator uses the first FfnAgent PE and that process's source/local Fabric
trace sequence. Generation membership is authoritative preparation metadata.

The Collector runs during serving, submitting scans at the configured cadence
and progressing outstanding work without unbounded task admission. Use one
precreated nonblocking collection Stream and at most one in-flight transfer
frame per Device. An acquire kernel claims committed slots, a pack kernel
copies their immutable payload into fixed staging, and `cudaMemcpyAsync`
transfers the completed frame into pinned Host storage. Query a completion
event before Host inspection. Preserve the collecting thread's Capture mode
when polling during concurrent production Graph Capture. No production Stream
synchronization establishes ordinary collection progress.

After D2H completes, update distinct copied counts and bounded legacy views.
Copy into an available fixed Host delivery batch, sort by Event sequence and
publish for the Writer. If the delivery pool is full, count post-commit loss
and continue retirement. The DMA frame is independent of Writer-held storage.
Reclaim Device slots only after complete D2H and retire the transfer frame only
after its reclamation work completes. Reclamation never waits for disk.
Round-robin source scanning uses finite per-tick work.

`poll`/`release` retain the C0 Host-batch ownership boundary: a returned span
stays valid through its consume token's release. The process Writer owns the
normal consume path; additional clients do not compete for those batches.
Writer publication or counted discard releases the Host batch. CUDA failure
with uncertain retirement retains all affected owners and tokens.

Host events follow the same immutable publication and copy ownership protocol.
The existing SGLang Graph JSONL fields enter a bounded concrete compatibility
queue consumed by the same Collector/Writer. Preserve sequential caller order,
including pending earlier entries; do not turn this log into purported v4
Graph execution evidence. Its bounded delivery diagnostics remain separate
from the frozen C0 quality schema. Dynamic Graph identities belong to B2.

### Scoped B2 Operation identity prerequisite

Provide the shared identity needed by production Transport Events before
replacing their old observation histories. B1 preparation supplies the actual
Run, registered Host/Device Producers, Generation and immutable participant
policy. Endpoint allocation and shared Operation sequencing belong to this
prerequisite. It depends on the event/status declarations, resource admission
and prepared identity/metadata boundary; public legacy reader decisions are
independent of it.

Add concrete observation value/storage declarations near the Transport owner
(`src/cext-include/xpool/transport/observation.hpp` and its implementation).
Reuse Arena/Attachment lifetime and the process collection budget rather than
adding an independent general registry. The public IPC prefix is exactly the
C0 eight-byte aligned u64 Operation word, with no Graph relay. Retain only
bounded current identity/counter state per Endpoint; count its actual bytes
and allocation rounding, independently of Event capacity.

At the existing `Arena::create` preparation path, allocate one independent
observation region for each actual rank-local Arena pairing and assign a fresh
Endpoint sequence from the AtnAgent Device Producer. Export the exact C0 v4
six-field `observation_region` descriptor using canonical CUDA IPC handle
encoding. Carry it alongside the existing Arena handle through the existing
publication/acquisition broker record, wire validation and Native bindings;
introduce no new RPC. Metadata registers the actual Endpoint and optional
lineage before Events reference it. Opaque IPC handles remain control-plane
values, never raw metadata.

Derive one immutable `transport_identity_required` policy for both peers.
Unified Timeline, independent Transport observation or another explicitly
identity-dependent consumer requires the region. Graph-only/Routing-only does
not request it merely by being enabled. Required-but-missing, policy/version,
owner, Generation or allocation mismatch rejects preparation; disabled identity
allocates no region. `Arena::from_handle` and the original Instance attach path
open both required mappings before execution/Capture, retaining exact owners
on partial preparation or uncertain rollback. Stable borrowed pointers may
enter observation Context without changing production RequestMetadata or the
production Arena's allocation layout.

After `try_begin_staging()` succeeds and before the opening observation Hook,
the Instance generates a fresh positive Operation sequence and writes it before
`publish_request()`. Event reservation, loss or legacy projection enablement
does not govern sequencing. Failed Claims allocate none. Use the existing
mailbox publication release/acquire visibility. At the AtnAgent's actual
`observe()==Published` boundary, copy the word into current local observation
state before request validation, execution or terminal result publication.
Include the Published path used during Resident drain. Each related Event
contains that copied identity before `publish_result()` permits reuse; the
Collector never rereads mutable IPC state to complete a historical Event.
Later successful Claims get fresh sequences, including ordinary repeated
Graph Replay of the same captured kernel addresses.

Sequence exhaustion stops only the affected observation domain and is
reported explicitly; production outcomes remain unchanged. Zero, repeated or
decreasing observed identities are unavailable/conflicting evidence, never
substituted with an earlier identity. Endpoint or process recreation produces
an isolated domain. This prerequisite neither adds service-time Generation
rollover support nor emits TransportFabricAssociated/Graph association Events.

Instance mapping close requires actual completion of its submitted users.
AtnAgent allocation release additionally requires actual peer detach and
retirement of local/collection references. A broker lease or failed publication
response alone is no CUDA detach proof. Keep owners on uncertain publication,
failed cleanup, in-flight work and deadline expiry; only proven unpublished,
unused preparation permits local rollback. Integrate with the existing
production/collection close boundary, not a duplicate retirement state machine.

### Configuration and resource admission

Install immutable `debug.timeline` policy through `xpool.config` before Native
initialization and Capture, following existing precedence/environment rules.
Use C0 defaults: 8 MiB per process/Device collection memory, 32 MiB total Host
collection memory, 100 ms cadence, at most 1 MiB per complete chunk, 1 GiB
process disk quota with an 8 MiB metadata reserve, and at most 5000 ms
additional observation shutdown wait. Keep frozen C0 capacity fields and
reject invalid minima, arithmetic overflow and participant mismatch.

Charge every physical allocation once. Include slots/control, active IPC,
Context and sampler allocations, alignment, transfer scratch and current Hook
facts on Device. Charge pinned frames, delivery and Writer-held batches,
encoding scratch, metadata/control, explicit worker stacks/guards, legacy
slabs, canonical provenance and peak Native/binding read storage on Host.
Routing payload pools and structural Graph storage keep their independent
memory capacities; their files share the process disk ledger.

The 32 MiB Host ceiling bounds internal collection storage. Include all
simultaneously live collection, reconstruction, read/result construction and
automatic-export serialization allocations. File publication does not release
a snapshot or serialization object still held internally. Return its memory
charge only after actual release; the file has a separate disk charge.

Internal immutable reads and reports, including reads queued for export,
remain charged through their actual last release. The collection budget is
separate from total process RSS. Original snapshot getters retain their current
interfaces and behavior; the collection components expose budgeted internal
read images without an additional detached-result delivery API.

Start with 256-Event transfers and four fixed 256-Event process delivery
batches, publishing small chunks directly. These are internal geometry
choices, subject to complete admission rather than new public configuration.
Do not add a second aggregation queue or general backend/provider framework.
Endpoint/Generation preparation admits new resources before publishing them;
old live readers/work prevent reopening their charges. Reject insufficient
budgets instead of silently shrinking legacy retention or increasing memory.
Compute Device admission from the sum of distinct active allocation requests,
including actual alignment, before observation starts. Reuse
`DeviceSourceState::layout`, `TransferLayout::for_capacity`, the optional
IPC/Context/sampler owners' layouts and the shared `MemoryBudget`. The source
layout includes its states and slot controls; the transfer layout includes
packing payload, ReadTokens and transfer control. Add necessary storage not
already contained in those allocations exactly once. Never add a component
again on top of an inclusive layout or hide scratch within alignment. Reject
overflow, incomplete requirements or totals exceeding the configured limit;
the default remains 8 MiB. The fixed C0 logical example is a lower-bound check,
not complete physical admission. Carry this expanded allocation rule into the
shared C0 text without changing public capacities or wire values.

Use a single `MemoryBudget` per accounting scope, with mutually exclusive
allocation categories and a consistent used/peak report. External memory and
explicit thread stacks retain move-only reservations through actual retirement.
Pageable Host allocations own their reservation and release storage before
returning the charge; moving an allocation transfers both. Reuse the current
checked integer utilities. Add PMR/shared-container adapters only when an actual
owner requires them; preserve the public snapshot vector types.

`HostRequirements` requires each actual allocation/read-peak category to be
declared, including explicit zero for inactive owners. Missing component sizes
reject admission. `LegacyMemoryDemand` combines C0 slabs with the actual Native
Record/vector declarations and caller-declared finite provenance/report storage.
Keep coexisting domains and retained runtime reads charged; this sizing input
does not introduce a public reader restriction or waive delivered-value costs.
The known C0 logical Device minimum is exposed separately from complete physical
admission. Schema-valid policy alone cannot certify installed providers.
During incremental delivery, enabled unified policy fails at runtime preparation
until all four evidence classes, Context, calibration and the Collector/Writer
are actually connected. Policy resolution remains available for preparation and
review, and disabled timeline policy retains the existing runtime path.

Unified outdir owns artifacts and quota; enabled legacy outdirs must match.
Independent observers share one process ledger across directories. Legacy flags
install their own bounded projections, not a duplicate Device collector.
Disabled observation installs no collection allocations, workers or files.
Unavailable required providers cannot be advertised as enabled unified mode.

### Files and failure behavior

Writer owns all filesystem operations. Publish immutable metadata before any
referencing artifact; hash exact UTF-8 bytes, including their final newline.
Encode nonempty `chunks/<producer>/<sequence>.xpool` with the v4 128-byte header,
CRC-32/ISO-HDLC, metadata hash prefix and at most 4095 Events. Order each chunk
by Event sequence; allow late older sequences in later chunks. Do not infer
causality from file order or raw timestamps across clock domains.

Reserve disk before creating/extending unique same-directory temporary files.
Close before atomic rename. Retain failed temporary files and account their
actual bytes; preserve the complete published prefix. Quota exhaustion stops
collection, never deletes earlier evidence, and preserves close-summary space.
Metadata failure stops the affected source, marks totals inexact and prevents
undecodable chunks. Writer errors count accepted but unpublished events as
post-commit loss. The Collector can finish/count discards while export fails.
Report typed errors and frozen quality fields rather than inspecting messages
or inventing stop-reason values. Atomic publication does not promise power-loss
durability.

### Legacy compatibility and bounded readers

Each immutable internal read owns budgeted Record and evidence storage from
one cutoff. A retained report keeps its own storage charged after the Record
image retires. Reads, including pending automatic exports, remain charged while
held; each additional read must pass shared-budget admission. Registration
checks the complete internal read demand, and process admission includes that
peak with all active owners. Original public getters and JSON projections
retain their interfaces; no extra public Record/report copy API is introduced.


Retain earliest N local operation IDs, default 8192 per Transport Endpoint or
Fabric Producer/Generation. Allocate IDs at the original record-begin boundaries
independently of Event capacity. For AtnAgent, prepare the trace after the actual
validated Submission is filled, before both the independent association Hook
and the ordinary SubmissionPrepared Hook. Both listeners read the same trace;
neither allocates another operation ID. Repeated later facts can recover an eligible
record whose opening Event was lost. They never recover a missing timestamp.
IDs above capacity affect operation capacity loss; eligible IDs with no
evidence remain unobserved, with unsupported identities null.

The Fabric recording owner provides a stable borrowed Device view with the
registered BufferView, Generation, PE, and current trace. The view uses the
existing Source reservation API and raw Device globaltimer clock. One shared
trace sequence serves all roles and Instances in a Producer/Generation; bounded
current-operation tables retain facts for subsequent events, not history. A
serialized Instance owner prepares/updates its entry. Synchronous listeners
copy facts; the Collector never reads these mutable entries. Empty views are
disabled. Full/stopped Event admission does not allocate a second operation or
freeze the original record-begin count. Trace exhaustion closes the affected
Source without wrapping or changing production execution.

Production names only its owning Fabric Point. An explicit preparation method
on that Point is composed with the built-in Devkit adapter before listener
dispatch; it creates no third raw event. The association Point/listener remains
independent. Its pre-reservation missing/conflicting Operation, wrong Source or
invalid-fact failures enter a fixed diagnostic bitset, not fabricated Source
attempts, Context loss, new C0 counters or stop reasons. Successful reservations
retain the existing commit/abandon ownership rules.

Install the view only after metadata and the complete aligned allocation pass
the common budgets. State/current tables, binding/control and transfer storage
are included, with coexistence peaks. Keep views and their owners through all
kernel/Graph users and asynchronous collection; stop is not retirement. Reuse
the existing service and retained-resource path, with no association-owned pool,
thread or shutdown deadline. The public reader's zero timestamp is valid and
each listener reads time at its own boundary.

Cold participant preparation uses the actual ArenaProjection before Transport
preparation and `JOIN_READY`. It admits the Device source/frame, role-specific
current tables, binding, enabled Routing storage and the configured IPC/Resident
demand together. Host preparation retains the metadata catalog, reconstruction,
formatter, bounded exporter and shared service, then probes overlapping internal
reads and original getter copies. `FabricJoinPost` allocates the admitted Device
storage, installs its borrowed constant view and starts the existing two workers
before execution kernels or Graph capture. Later Endpoint registration appends
a catalog revision before those identities can reach a Hook.

DeviceChannel owns the state for AtnAgent, Coordinator and FfnAgent. Every completed
transfer carries absolute original-operation counters and diagnostic bits with
its Events. Collector merges these cumulative samples once; it does not count
200/450 as separate original operations. A quiescent getter requests a fresh
empty scan and uses that transfer's original count for its reconstructed image.
Original Native vector snapshots keep their own reservations through copies,
moves and destruction. Automatic export retains a Native internal read and
queues the original JSON filename and evidence pair to the sole Writer.

The first automatic export immediately before finalization and the final worker join share one
observation deadline. The daemon's existing absolute cleanup deadline travels
with heartbeats; process preparation maps CLOCK_MONOTONIC to a conservative
steady-clock bound and only tightens it. Without a daemon cleanup trigger it
does not invent a finite whole-production deadline. Finalization removes the
Device binding only after real local drain and destruction of FFN Graphs, then
waits only for source/frame retirement. Production NVSHMEM teardown does not
wait for Writer I/O. The observation owner retains Host data and charges until
the later worker join, using the same original deadline; a slow Writer is
polled after production cleanup without replaying that cleanup. Bounded export
results remain available after owner release. Uncertain CUDA completion retains
the owner and production storage; observer handoff remains retryable before
collective destruction starts. Never-activated preparation can be cancelled
without CUDA, while any attempted activation requires actual retirement.

Use the C0 256-byte Transport and 384-byte Fabric reconstruction slabs. Add
finite canonical source references and known-fact/boundary masks. Store common
metadata paths once per domain and derive report field names from fixed role
tables. Serialize reports incrementally instead of retaining unbounded
per-operation strings or JSON trees. Equal repeated facts retain the canonical
minimum source; conflicting identities/facts/Events fail reconstruction.

Preserve existing getter names, Record accessors, legacy JSON shapes,
nullability, filenames, model identity and exact duration/quantile rules.
Missing results retain the legacy sentinel plus unknown-result evidence.
Terminal classification remains independent of evidence completeness. Raw zero
and absent time differ in recorded masks; reversed intervals retain evidence
and suppress only their invalid durations. Resolve conditional applicability
from actual production facts and metadata, never observer enablement.

A read retains its original synchronization precondition, requests a fresh
Collector cutoff while producers stay quiescent, drains/merges through that
cutoff and obtains exact independent operation counts. Materialize one
immutable snapshot/report pair. The original legacy getter projects its
snapshot; an evidence-aware API returns both from the same read. Preserve the
existing C++ vector containers, getter signatures, Record accessors, Python use
and legacy JSON. A new read preserves earlier delivered snapshot values.

Account runtime-managed read windows, result construction, binding
materialization and automatic-export serialization at their simultaneous peak
and lifetime. Size for the original calling behavior: retaining snapshot A
does not make a later read return the old cutoff or introduce a one-reader
rule. Existing readers cannot be overwritten. Successful file publication does
not release internal snapshot storage; its charge follows the actual read and
report owners through their last release.
The internal implementation uses a source-bound `LegacyRegistry` and one fixed
slot/provenance table per `LegacyView`. Protocol-boundary fingerprints detect
changed repeats without keeping an event history. The Collector's single read
mailbox requests a new drain and freezes exact original counts with Records and
report; a timed-out waiter cancels delivery while the Collector retains ownership.
Every live internal read uses the common ledger, rather than a one-reader lease.
Preparation checks the actual internal Record, report and control allocations.
The existing JSON formatter owns its separately budgeted envelope metadata.
The Writer accepts immutable reads through fixed, prepared `SnapshotExport`
mailboxes. All source and snapshot metadata are validated before merging, even
when raw batch delivery is full. These components do not install production Hook
identities or replace process-global getters; those adapters retain the original
quiescence and metadata preconditions.

Writer hashes actual legacy JSON bytes, publishes
the `.evidence.json` report first, then atomically replaces the legacy JSON.
Only a matching hash makes the pair usable. No report fields enter legacy JSON;
Native-only reports have a null snapshot hash. Sidecar and temporary replacement
bytes enter quota.

### Shutdown

Use production quiescence, client-before-exporter detach, Transport-before-Fabric
drain and NVSHMEM completion. Request the final fresh collection cutoff and
hand an immutable snapshot to Writer before retiring related Device state.
An earlier transfer cannot acknowledge a later requested cutoff. Device
retirement requires completed producer/Graph users, copies/reclamation and
relevant peer detach; Host readers and Writer may outlive those Device owners.

All observation close operations share one absolute deadline bounded by the
remaining production deadline and configured additional wait. Deadline expiry
returns unfinished status, retains live threads/CUDA/IPC resources and prevents
creating a replacement owner. A blocked Writer call is not forcibly cancelled.
Only reliable final checkpoints claim complete tails or exact pending counts;
crash tails remain unknown. Signal handlers perform no CUDA or file rescue.
Distinguish never-submitted startup rollback from
completed or uncertain submitted work. Existing production preactivation drain
behavior needs separate scoped resolution if it blocks failure qualification.

## Implementation And Validation

Implement coherent layers in this order, splitting Host/Device or file/lifecycle
changes into separate reviewable increments:

1. Event, statuses, errors, explicit codec and fixed vectors.
2. Configuration and complete allocation/read-peak admission.
3. Host Source/slots, then Device ownership and concurrency.
4. Host delivery, then asynchronous transfer and reclamation.
5. Metadata/chunk/quota publication, then process workers and bounded close.
6. Existing Host Graph JSONL background delivery.
7. Native legacy merging, then evidence-aware readers and paired publication.
8. Registered identity preparation, then the scoped B2 prerequisite in three
   increments (storage/descriptor, actual mailbox sequencing, isolation and
   retirement), followed by production Event/getter integration. Preparation
   and the identity prerequisite can precede public legacy reader work;
   getter integration waits for its interface decision.
9. Actual normal/partial-startup/deadline retirement, then frozen-source
   acceptance and independent A1 comparison.

Use [test conventions](../../../tests/README.md) and
[Qualification](../../designs/qualification.md). Required evidence includes
all bytes of the valid raw vector, structured rejection of malformed vectors,
exact snapshot/report fixture agreement, out-of-order commits, multiple
CTAs/Streams, abandon/stale tokens/exhaustion, in-flight DMA, slow Writer,
bounded Host exhaustion, Capture interaction, short writes, quota, metadata
failure, pair interruption, shutdown timeout and unknown crash tails. Attempt
Host ASAN/UBSAN; add targeted race/memory checks where they can test the actual
ownership protocol.

Resource and failure acceptance also covers overlapping reconstruction/read/
serialization peaks, snapshots retained after successful publication, last
internal-reference release, and a caller retaining result A while reading B.
Check complete Device layout sums, optional-category activation, overflow and
over-budget rejection before observation starts. Verify that Transfer reports
`io_error` while ordinary I/O loss can retain proven exact totals; an uncertain
transfer must instead retain resources and suppress exact/complete conclusions,
including when a different stop reason was already recorded. Repeated errors
must not grow diagnostic storage. None of these checks infer physical memory
release solely from an ownership-accounting transfer.

Validate Generation-scoped membership and Coordinator placement against actual
prepared process owners. Prove original operation counts survive total Event
loss, readers retain charged storage and disabled modes start no workers.
For the scoped identity prerequisite, separately prove descriptor/policy
validation and allocation accounting, two-process CUDA IPC identity agreement
at the actual Claim/Published boundaries, Event-full independence, failed Claim
and pre-publication close behavior, repeated execution/Replay, exhaustion,
Endpoint recreation and process isolation. Cover normal and terminal Published
paths, partial attachment, uncertain publication responses, failed cleanup,
peer detach with in-flight users and deadline ownership retention. Host value
tests alone cannot validate cross-process CUDA visibility or safe release.
Graph retirement evidence must distinguish still-owned Graph/clone/GraphExec
handles from unfinished execution after every handle has been destroyed. An
actual Published request held by the peer can establish that pending execution
without an arbitrary sleep. Check retention again after the peer permits late
completion. User-object create, retain and release failures need both before-call
and after-success cases: an error return cannot prove that no reference changed.
The IPC test executable alone wraps these CUDA symbols at its Host link boundary;
the production core and wheel expose no injection API. Verify injection hits and
real-call outcomes, preserve ordinary CUDA errors, and reject retries without
additional reference operations. These synthetic returns exercise owner failure
handling; they do not establish behavior under a poisoned CUDA context. Expected
retention cases require separate leak-accounting expectations from normal cleanup.
Apply the same before-call/after-success checks to IPC open/close, allocation,
copy, export and free. Select observation and production mappings separately,
including failures after an earlier extent has already been released. Use the
actual Instance attachment and AtnAgent pending/published Arena owners. Compare
both ledgers at the failed API boundary and after rejected retries; a returned
null pointer or a partially closed mapping is not retirement proof.
Compare enabled/disabled and constrained-capacity runs using actual benchmark
metrics, resource categories and losses. No invented fixed accuracy or
performance threshold substitutes for measured evidence. Full runtime
acceptance needs one frozen source/build identity, canonical build/stubs,
strict suite and the qualifications invalidated by this change. Host or
component tests alone do not establish serving, CUDA IPC or performance.

## Out Of Scope

B2 work beyond the scoped Operation identity prerequisite includes actual
Fabric association and dynamic Graph Context. B3 owns Routing
publication/attachment identity integration.
B4 owns actual asynchronous paired sampling. A1 owns offline reconstruction;
A2 owns clock fitting; A3 owns Perfetto/query output. Full four-class unified
enablement and C1 Dense/MoE acceptance require these providers. This plan does
not implement live KV monitoring, cross-host support, production request
metadata changes, per-node profiling or new serving platforms.

## Local Source And Review Boundaries

`timeline/hostbuffer`, `source`, `buffer`, `device` and `devicesource` own the
bounded reservation protocol. `hostchannel` owns Host scratch separately from
source and delivery storage; Writer-held or exhausted delivery pools cannot
prevent completed source copies and counted loss. `DeviceChannel` owns the
private nonblocking transfer Stream, pack/token frame, pinned frame and delivery
slab until actual CUDA retirement. Transfer errors retain an uncertain owner.
Their public quality stop reason is the existing `io_error`, while existing
Native typed Transfer diagnosis is preserved. Completion uncertainty is
independent of that coarse classification: retain every affected resource and
charge, keep unknown pending work null, `tail=unknown` and `exact_totals=false`.
A prior stop reason must not mask a later uncertain transfer. Keep bounded
specific diagnostics through the existing error/logging path; introduce neither
a public error category nor a quality JSON field. The Collector retains a
one-way uncertainty flag independently of its first stop reason, including
failures observed after an earlier final-looking quality image. An uncertain
Device channel rejects subsequent CUDA submission and cleanup retries.

`MetadataRegistry` serializes typed, bounded preparation facts into an immutable
`MetadataCatalog`. Its retained semantic callback validates copied Events against
the exact selected revision. Catalog bytes alone are insufficient for process
service registration. This is a Native prepared-facts boundary; it is not an
untrusted JSON reader or the independent A1 implementation. Production identity
assignment and actual provider capability installation remain separate steps.

`IdentityDirectory` owns the launcher's bounded Run namespace. It assigns a
random process-creation UUID once for an exact Host-creation/PID/OS-creation-key
registration, and interns Host/Device Producers, clocks and production Generation
references. Identical registration retries return the existing assignment;
reused PIDs with different creation keys obtain fresh identities. The launcher
verifies liveness and actual Device mappings before registration. Metadata
imports the assigned values through its existing typed declarations. The
directory has no alternate UUID-import registration path and never registers
the Daemon as a source. Its tables, strings and shared controls use an explicit
control-plane budget, separate from observed-process collection ledgers.
Published prefixes retain immutable definitions and their budget owner. Capacity
failure rejects new preparation; it never evicts or recycles prior definitions.
Daemon/Actor distribution and process collection installation remain the
production integration step.

`CollectionService` allocates fixed source/exporter tables and two explicit worker
stacks before starting one Collector and one Writer. Complete Source entries may
join later: channel, typed catalog, Writer storage, optional reconstruction and
exporters all pass admission before a release-published count exposes the entry.
Workers acquire that count and read only the immutable published prefix; vectors
never resize during collection. Late Device entries must already have completed
activation. Stop and registration publication share a lock, so shutdown cannot
miss a new Source. Cold legacy attachment remains available before start; late
Sources supply their complete reconstruction bundle with registration. All
registered Host channels, Device Host resources, metadata and the optional
`GraphLogStream` must share its exact Host ledger. Every Writer shares its disk
ledger. The Writer publishes metadata before referencing chunks; temporary
extents remain charged on failure. Persistent paths and publication scratch use
budgeted storage. Preexisting raw-run extents must be registered by the cold run
owner before export. Active quality checkpoints do not seal source totals.

Observers that share a Device Producer borrow the registered channel, catalog
and reconstruction registry from that service. Transport on an AtnAgent reuses
its Fabric Source; Fabric preparation admits reconstruction handles for every
configured Endpoint alongside its own view. Borrowing does not allocate another
Event pool, transfer frame or worker pair, or prove any producer has retired.

Endpoint recreation reuses its existing snapshot destination slot through
`replace_snapshot_export()`. The old mailbox must be closed and have no queued
or Writer-held read. The replacement retains its publisher and charged file
extents; new formatter admission includes the still-retained old owner. The
Writer borrows each mailbox under a short handle mutex and releases that mutex
before I/O. A previously borrowed closed mailbox has no work after replacement.
Global stop forbids replacement. This lifecycle connection requires ABI 97;
production Transport installation and its accepted exceptional-event encoding
remain prerequisites for enabling the replacement pipeline.

Retain the service's first worker error as an immutable code/Producer record
with 512 fixed bytes for operation and underlying-cause text, included in the
budgeted service allocation. Truncate overlong text explicitly. Publish only
after the record is complete, allowing concurrent reads without retaining an
exception, backtrace or error history. `rethrow_failure()` exposes that record
through the existing `CollectionError` type; constructing the requested exception
does not add persistent diagnostic storage. CUDA failure messages identify the
failed launch, copy, completion query or resource-retirement operation.

Shutdown closes admission and accepts producer retirement only from the actual
runtime owner. Its absolute wait includes lifecycle-lock acquisition and uses
nonblocking thread joins. Timeout or uncertain CUDA cleanup retains workers,
channels and their charges, and prevents a second active process service. A
partial worker launch retains any started thread; it does not establish safe
production retirement. Production startup/retirement wiring remains a later
integration step.

`CollectionService::retire_sources()` supplies the separate bounded collection
handoff on that same owner. It accepts the same actual producer-retirement proof,
then waits for final Source scans, reconstruction and Device storage retirement.
Its deadline includes lifecycle-lock acquisition and is never renewed by a
wakeup. It does not join workers or wait for Writer I/O; queued and Writer-held
Host data, metadata and snapshots remain charged, and the process claim stays
held until full shutdown joins both lanes. A prepared service with registered
resources reports incomplete retirement until its workers have run; retirement
does not implicitly start them. An empty cold service has no such obligation.
This completion is the Collector side of the Endpoint recreation requirement;
actual Graph completion and IPC peer detach remain independent prerequisites.
Production calls are installed with the Event providers, after their accepted
shared-event boundary and full memory admission are available.

The independent SGLang Host Graph adapter uses `GraphLogSession` to prepare and
start the common process service before later raw provider installation. The
`ProcessPreparation` weak reference reuses that service without extending its
lifetime; Graph and Fabric owners retain it on their common ledgers. Two Source
slots cover the registered Host and Device Producers, while bounded exporter
slots cover the configured Endpoints and Fabric output. An unjoined owner cannot
be replaced. Graph setup initializes no CUDA context. Each
Hook enqueues into a fixed FIFO with a 127-byte backend-label limit; oversize,
full and competing-head outcomes drop complete new records and have separate
compatibility counters. Original JSONL keys, filenames and per-caller lifecycle
order are retained. Files become visible asynchronously. An existing destination
is preserved and reported as a Writer failure, rather than truncated. A separate
`.quality.json` summary uses protected disk allowance and complete-file atomic
publication. The session reports drops/errors independently of worker joining.
Graph-only sessions share immutable process Host/disk limits and the same
ledgers, including any closed session still retained by its caller. Repeated
Python installation reuses its original active session. `close()` and process
`atexit` close admission while Python holds the GIL, then
release it for the bounded wait. This proof covers only Graph Host enqueue
calls. Graph has its own collected marker, so its file can finish while raw
Sources continue producing. A mixed session's close returns false until the
actual raw owner has supplied the global retirement proof and both threads have
joined. With no raw entries, Graph close freezes registration before its proof
can cover the whole service. There is no second Graph service.

Automatic Agent exports begin at FINALIZING after the cross-process Drain.
The Fabric export is followed immediately by the Native finalize hook, which
also exports the borrowed Transport view before retiring their shared Source.
Quiesce alone starts no observation close wait. All observers and repeated
closes share `ProcessPreparation`'s first absolute
observation deadline, capped by the production cleanup bound; later calls can
only tighten it. Instance heartbeats deliver that bound before reporting a
terminal Fabric phase, including when only the early Graph owner exists. An
expired bound still allows nonblocking completion checks, never a fresh wait.
The shared registration and retirement layout requires ABI 96.

The collection retirement boundary and observed Transport lifetime require ABI 92. Canonical build/stub generation,
Host/CUDA tests and serving qualification must run remotely before acceptance.
Generated stubs are not edited by hand during local-only implementation. Unified
startup continues to reject missing production providers. Existing Transport and
Fabric getters/Record vectors and the FFN structural Graph snapshot retain their
current production paths until legacy reconstruction and Hook integration.

## Cold process preparation

The Transport storage increment supplies the exact six-field C0 descriptor,
an eight-byte `ObservationWord`, and a concrete `ObservationPool`. Preparation
supplies the existing Host and Device ledgers and retains one pool for the
entire registered Device Producer lifetime. Its fixed mapping slots and shared
controls use Host admission. Each owned CUDA extent charges the existing
256-byte allocation unit once; an IPC import charges its local Host control,
not a second Device allocation for the same exported storage. Complete startup
admission must include all simultaneously owned extents with other providers.

The pool admits Host/Device storage before CUDA and consumes a fresh Endpoint
immediately before attempting allocation. Admission failures unwind without
consuming an identity. Once CUDA begins, any uncertain result closes new
admission and retains the bounded slot, owner and charges; dropping a caller
reference does not release them. Explicit retirement needs local kernel,
Graph and collection completion, plus actual peer detach for owned extents.
It preserves charges until CUDA release succeeds. A retained Host descriptor
owner remains charged after its Device extent retires. These cold calls do not
read a live operation word or install mailbox sequencing.

`TransportArenaHandle` carries the optional immutable descriptor through the
existing wire record. Numeric fields reject coercion, overflow and zero
identities; preparation checks the actual owner, Generation and immutable
policy. Cold preparation owns one IPC pool per actual Device Producer and one
shared Device ledger. It admits the configured simultaneous owned IPC extent,
including each 256-byte allocation unit, before Arena creation. Importers charge
Host mapping controls and allocate no duplicate Device extent. This is admission
for the installed IPC provider; complete Event/transfer/Context/sampler and
legacy-provider production admission remains part of the common collection
installation, and unified enablement continues to reject missing providers.

The AtnAgent retains a stable pending Arena before allocation, initialization,
handle export or Hook execution can fail. The Instance likewise installs its
persistent attachment before opening either mapping. Observed Arena fallback
destruction never frees CUDA storage. A failed cold operation closes IPC admission
and retains the affected owner and charges; only successful explicit retirement
returns Device capacity. The Arena's Host control remains charged until its
object is destroyed, including after a successful CUDA release. The fixed pending
slot and its Host ledger are retained by the AtnAgent runtime.

The existing publication/acquisition RPC carries both handles. The broker rejects
missing/extra descriptors, wrong prepared owners/Generation, incompatible extents,
and shared Endpoints or observation handles across Arenas. Each Actor registers
the paired Device Producers and metadata-only Arena suffix before publication or
attachment. Native bindings enforce the frozen shared-identity policy and unique
prepared pool. These cold calls do not read the mutable word or start collectors.

Observed Instance capture retains a CUDA user object in the capturing graph
before submitting a request kernel. One Host-budgeted callback target and shared
control block belong to each Endpoint; the same object serves all its captures.
CUDA carries references into graph clones, executable graphs and unfinished
executions. Ordinary requests allocate no per-request application control.
The target retains itself across uncertain initialization and static teardown.
The callback only publishes an atomic completion fact; it performs no CUDA calls,
blocking waits or resource cleanup. CUDA-internal bookkeeping follows the existing
driver-internals accounting boundary.

The caller destroys every Graph and GraphExec that captured the attachment before
observed detach. Detach holds the existing lifecycle mutex, releases the creator's
reference exactly once and waits up to one second for the callback. Only then
does it synchronize actual Device work and close both mappings before sending
deregistration. A live Graph may still replay after an earlier synchronization,
so synchronization alone cannot authorize release. A callback timeout or CUDA
failure quarantines the attachment and preserves its mappings, callback target
and charges until process retirement; a late callback cannot recover it.
The callback target's Host charge is returned only after successful cleanup and
actual target destruction. After engine Host cleanup and local Device
synchronization, the SGLang adapter calls each captured phase backend's
`cleanup()` before releasing KV or detaching Transport. Cleanup failure retains
the runtime attachment. The native reference guard remains the final proof;
backend cleanup does not replace the full production shutdown deadline. The disabled
observation path retains its existing lifecycle preconditions.

AtnAgent release requires confirmed lease
quiescence plus its completed Resident drain, or a creation rollback proven to
precede every publication attempt. Later collection/Graph consumers must join
this same local completion boundary when installed. A missing proof, uncertain
CUDA operation or incomplete attachment retains ownership and rejects reuse.

An Instance keeps its prepared pool, Endpoint owner and last attached sequence
after successful detach. Exact retries while attached are idempotent; after
detach, the same or an older Endpoint cannot recreate a private counter at zero.
A later attachment requires an increasing Endpoint from that same prepared
owner. Pool/policy or owner replacement within this prepared process is rejected
before CUDA; changing the prepared process domain still requires the existing
production restart boundary. Attachment control outside the separately charged
Arena is admitted once in the same Host ledger and stays charged while retained.
The Python runtime rejects retired Endpoint retries before marking a native
operation uncertain. AtnAgent registration reconciliation preserves an observed
Arena's original Instance PID and immutable process/device preparation facts;
even PID reuse with a changed process start key requires a new Arena and Endpoint.
Instance CUDA pointer queries, capture-reference operations, request launch and health reads also quarantine
the attachment on failure; later submit, attach, reads and detach cannot bypass
that retained uncertainty. Request geometry checks precede these CUDA operations
and leave the valid attachment usable when the caller supplies corrected input.

Private Operation counters are separately allocated and initialized for both
peers before any request use. The public IPC prefix stays eight bytes. Each
private allocation uses the existing 256-byte allocation alignment; only the
exporter additionally charges the 256-byte IPC extent. Startup admits the full
configured pool demand, and uncertain preparation or release retains both
allocations and charges. A successful claimant's counter produces every
positive u64 once, then publishes zero with a retained identity-exhausted status.
The receiver copies increasing positive words and stops its own identity domain
on zero, repetition or regression, without substituting a previous identity.
These mechanisms return values for immutable events and do not control
production outcomes or depend on Event reservations.

The Instance invokes that counter only after its actual mailbox Claim succeeds.
The normal and terminal AtnAgent Published paths copy the shared word before
validation or result publication. Every protocol Hook receives a value containing
the Endpoint, Generation and Operation facts; later transitions use that copy
after mailbox reuse becomes possible. A borrowed `ObservationView` carries the
private pointers and immutable domain facts as a request launch argument or a
separate Resident table. Neither `ArenaView`, the production allocation layout,
nor `RequestMetadata` gains an observation field. Disabled observation supplies
an empty view and allocates no observation table.

An observed Resident uses the common Host allocator for its object, shared
control block, ordered source arrays and temporary sorting array. It admits the
complete aligned Device demand for the Arena array, drain state and observation
table before CUDA allocation. Preparation includes the maximum configured
Resident demand alongside the Region pool, counting each actual allocation once.
The current pointer types need 1024 bytes for an eight-Arena Resident, in addition
to 4096 bytes for its eight exported Regions; the formula uses actual `sizeof`
and checked alignment rather than these example totals.

Resident construction prepares Host storage only. The runtime installs its
stable owner before CUDA initialization; a self-reference retains that owner,
the copy inputs, streams and charges until explicit proven release. An uncertain
initialization, query, health read, drain command or release closes new pool
admission and requires process retirement. Only complete successful release
sets the released fact and returns the Device charge. A null partially initialized
pointer cannot authorize Arena destruction. Successfully released Host storage
remains charged until it is actually destroyed. This supplies identity facts
to Hooks; production Event writer/getter integration remains a separate delta.

The control-plane retirement prerequisite keeps publication attempts distinct
from acknowledged epochs. A lost response still requires lease quiescence before
arena release. Native attach or detach exceptions leave a sticky uncertainty
state in the Instance runtime: it retains its registration and client, rejects
CUDA retries and never sends a successful departure on that path. This is a
conservative process-retirement boundary until Native ownership can provide more
specific completion evidence.

An observed lease retains its exact Instance process identity and ABI with the
handle. Registration replacement, heartbeat expiry and process disappearance
cannot remove it or replace it with another mapping. Only the original owner's
existing deregistration, sent after successful Native detach, clears that
obligation. Quiescence reports the retained owner even if its registration is
gone. Plain Transport leases keep their existing abandoned-owner policy. This
increment establishes control-plane gating shared by the Arena owner. It does
not infer actual CUDA success from a timeout or a missing process, nor complete
raw provider admission or mailbox identity sequencing.

The existing registration request carries a bounded observation extension with
Linux boot identity, exact OS process-creation key, immutable configuration
fingerprint and the actual process-local logical/physical Device mapping.
The Daemon checks those facts against the live process and its configured
physical placement before assigning identities. Its bounded Native directory
uses a separate 1 MiB control-plane ledger; the Daemon creates no event Producer.
The existing FabricPlan distributes the Run, registered Sources and clocks,
Generation reference and complete authoritative Agent PE membership.

Bootstrap freezes policy before Native initialization. Enabled observed roles
admit the common Host/disk ledgers before constructing observation requests.
Agent preparation imports typed metadata before join; Instance preparation does
so at its executable barrier before attachment and Capture. Equal retries reuse
metadata. A changed live Plan requires a later explicit retirement path and
currently rejects. Requested raw providers remain unavailable until their Hooks
are connected; the unified startup guard remains in force.

`ProcessPreparation` and Graph compatibility sessions share one Host ledger.
Besides individually charged Native metadata and control allocations, admission
reserves 3 MiB for bounded Python cold-preparation storage: up to 128 participant
and PE descriptors, simultaneous wire decode/validation and retry copies, and
binding/control temporaries. This conservative capacity remains charged until
process exit; it is not an exact occupancy or RSS claim. A maximal-envelope test
checks the supported Python representation and coexisting copies against that
capacity. Existing production model/Plan facts are borrowed, not duplicated into
a Python metadata tree. The overall production owner must still admit Device
pools, IPC, all providers and service resources together before collection can
be declared ready.

## Open Questions

- **Acceptance boundaries:** The minimal Operation identity prerequisite is
  in scope; full association, Graph Context and the remaining providers are
  separate work. Its completion does not establish all of B2 or full unified
  readiness. B1 still needs actual collection, compatibility and failure-path
  evidence. Full unified startup must reject missing required providers rather
  than reinterpret v4 capabilities; agree the staged acceptance boundary
  without weakening this requirement.
- **Shared-text reconciliation:** Incorporate complete Device allocations and
  the existing `io_error` mapping with independent completion uncertainty into
  C0. These implementation directions are resolved; do not reopen the 8 MiB
  default, omit pack/tokens/state or add an error enum. The pinned historical
  C0 reference remains distinguishable from the clarified target.
- **Independent evidence:** Obtain a frozen A1 v4 implementation/input identity
  for cross-checks; local codec agreement is not A1 acceptance. Resolve actual
  production startup-retirement prerequisites before claiming failure-path
  qualification.


### Transport production connection and v5 handoff

Transport now keeps a fixed current-request table on DeviceChannel and writes
ordinary/anomalous events through the same reserve/commit Source. At AtnAgent
the table shares Fabric's channel, Collector, Writer and ledgers. Instance
prepares a channel from its registered metadata before attachment. Cold table
bytes, bindings, reconstruction, getter copies and simultaneous export reads
are admitted under the existing budgets; no Device history fallback is used.

Endpoint creation installs stable registered identity; original protocol Hooks
copy the existing OperationFact. Original quiescence permits count read and a
fresh Collector drain, then Native reconstruction creates the old Record view
and same-cutoff evidence. Native Endpoint close queues automatic publication.
The Writer serializes and publishes both files. Endpoint replacement reuses a
closed export slot; it never retires a still-shared Source. All-source retirement
still requires actual producer/Graph completion. Uncertain counter-copy targets
remain in the retained owner, and cannot authorize table reuse.

Serving qualification checks both Transport sites and Fabric artifacts against
raw chunks, metadata hashes and final source quality. Combined capacities must
fit the configured Host/Device budgets; excessive coexistence rejects explicitly.
The v5 candidate is a source handoff, pending its own remote build and execution
evidence. Earlier v4/ABI97 evidence does not validate this changed version.
