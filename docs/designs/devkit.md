# Devkit

Devkit observes real production execution through typed Hook Points. Transport,
Fabric, and FFN computation retain production control flow and resource lifetime.
Resolved Debug policy is immutable after native initialization.

## Collection ownership

Transport and Fabric Hooks publish immutable version-7 Events into one Device
Source per observed process/Device. Graph Host calls use the registered Host
Source. Runtime preparation distributes one Run UUID and registered Producer,
Generation, clock and Endpoint identities before capture. Observation state stays
outside production RequestMetadata and Transport/Fabric arenas. Successful
Mailbox Claim allocates Operation identity independently of Event admission;
Routing publication similarly allocates attachment identity independently of
Event or tensor capacity.

Each observed process shares one background Collector thread, one Writer thread,
and bounded Host, Device and disk ledgers. The Collector copies only committed
immutable slots, checks completion and reclaims slots after their copy retires.
Bounded Host batches transfer ownership to the Writer; file I/O never blocks the
Collector. Full capacity records explicit loss without changing production
execution. Disabled observation installs no workers or Hook storage.

Preparation admits actual layouts, alignment, Hook control, transfer buffers,
IPC regions and complete Routing copy/serialization coexistence. The allocating
owners retain those reservations through actual retirement. Unknown sizing,
overflow, inconsistent registration and insufficient budgets reject preparation.
Graph-only and Routing-only observation do not require Transport IPC identity.

A file error stops export and retains reliable Event prefixes. Transfer failure
preserves the first stop reason and bounded diagnostic while independently
tracking uncertain CUDA completion. Uncertain slots, tokens, frames, mappings
and memory charges remain owned until actual retirement proves release safe.
Timeout is not retirement; uncertain tails and totals remain explicitly unknown.
Runtime Quiesce/Detach/Drain boundaries supply producer-retirement proofs before
Source closure and worker joining.

## Hook contract

Host and Device call sites use the neutral typed surface:

```cpp
Point::hooks(Point::Context{...});
```

A production call site names only its owning Point. Host and Device adapters
form an explicit compile-visible catalog. Hook Points provide observation; the
production path owns control flow and execution selection.

Observe calls are additive, so the surrounding production code remains readable
and complete without them. Collective Device Points are reached by every
participating CTA thread; adapters that need only one observation select thread
zero themselves, while an Observer that copies a collective payload may use the
complete CTA.

## Protocol and Graph evidence

Protocol Events name actual publication or observation boundaries. Capacity and
delivery facts come from the FfnAgent that selected them; Coordinator evidence
does not infer those values. Transport anomalies retain original invalid request
values and actual local terminal observations. Readers cannot infer an absent
boundary or duration from repeated scalar facts or Source-wide loss counts.

Transport result codes describe their actual boundary. Successful result
observation and output copy can precede a Closed boundary with Shutdown; retain
each result instead of replacing it with an earlier value. Result presence is
set only after the actual local result boundary. Endpoint metadata follows the
contract schema; Arena handle suffixes remain internal pairing checks.

Graph construction registers fresh Host-owned templates and installed structure
inventories. Lane templates reference their actual registered compute templates.
SGLang capture and execution calls emit versioned Events with separate template
and execution identities. Hook diagnostics are bounded and reported outside
execution Hooks. Native structure attachments retain CUDA node kinds as integers,
binding-site counts and compute/delivery branch counts.

Routing captures CPU int32 IDs and float32 weights with live_rows by
effective_topk shape after actual producer drain. A budget-owned handoff keeps
tensors and their original Invocation, Lane/Lease, layer, PE, clock and raw time
alive through Writer serialization. The installed Safetensors library serializes
the payload; tensor-key positions locate data and never serve as attachment
identities. Dense and non-Router participants declare Routing not_applicable.

## Serialized observations

Raw publication uses XPOOLC07, 256-byte Events and immutable metadata/quality
revisions. Reserved bytes 136–143 are zero and presence bit 8 is forbidden.
Artifacts and IPC descriptors accept version 7 only; no older-version fallback
is installed. The Native ABI and calibration algorithm versions are independent.
The Writer publishes required metadata before referencing chunks and closes
same-directory temporary files before atomic no-replace publication. Failed
extents remain charged to the disk quota. Legacy getters, reconstructed Records,
JSON/JSONL exports and snapshot reports are removed. Prefill-logit observation
has its separate qualification contract.

Independent mode selects classes with Observer flags and respects each class's
outdir. Each directory contains `timeline.<run_uuid>/`, including
`chunks/<producer>/<chunk_sequence>.xpool`, `metadata.<producer>.<revision>.json`
and `quality.<producer>.<revision>.json`. Graph structures live under `graph/`;
Routing tensors under `routing/`, with `attachments.<producer>.json` supplying
final outcomes. Artifact metadata registers relative paths and SHA-256 hashes.
Only published attachment outcomes have usable paths/hashes.

Sources spanning multiple class directories keep one chunk sequence space and
count each Event once. Exact metadata and cumulative Source quality are copied
to each registered directory; every physical copy uses the shared disk ledger.
Equal directories coalesce publications. Configure sibling class directories
under a common capture root for convenient offline consumption. A1 takes explicit
file inventories; A3 requires input-relative paths and one Routing attachment
root. It does not discover files or import closing indices automatically: their
entries belong in the explicit input manifest. Graph paths resolve relative to
the Graph Run directory. Conversion outputs use a separate directory and quota.

Unified mode remains rejected until Dynamic Context, calibration and complete
four-class association providers are connected. Independent evidence remains
valid with explicit disabled, not_applicable and unavailable capabilities.
[Tooling](tooling.md#shared-serving-lifecycle) owns test-input validation and
semantic qualification assertions.

The SGLang Prefill-logit Observer retains one FP32 next-token distribution per
controlled qualification request on TP rank zero. Each EXTEND replaces that
request's snapshot, so completion exposes the final Prefill chunk. Files identify
the process and URI-encoded request ID; metadata retains the original request ID.
Serving-graph and Elastic Context State qualification share this observation.
