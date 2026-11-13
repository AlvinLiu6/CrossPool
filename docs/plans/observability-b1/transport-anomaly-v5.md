# Transport anomaly encoding: local C0 v5 candidate

This candidate implements the agreed requirement to preserve invalid request
fields and actual termination evidence. It is authorized for local B1 integration;
it is not a claim that A or B2 has adopted or jointly released version 5.
The original v4 fixture header remains unchanged. The separately named
`tests/suites/cext/devkit/timeline/fixturesv5.hpp` contains independent candidate
bytes, including ordinary v4-derived inputs and anomalous Transport events.

All newly produced metadata, raw chunks, quality checkpoints, snapshot evidence
and observation IPC descriptors use version **5**. Chunk magic is `XPOOLC05`.
The chunk header remains 128 bytes and each Event remains 256 bytes, with the
original little-endian field offsets. Native ABI is independently versioned.
A version 4 participant or parser must reject this candidate rather than skip
unknown kinds or interpret version 5 with version 4 semantics. Legacy snapshot
JSON retains its existing shape; its paired evidence report is version 5.

## Additional wire vocabulary

Kind **150**, `TransportAnomaly`, represents one actual Transport Hook whose
received request contains an invalid enum. Presence bit **18** denotes its raw
request payload and is forbidden for other kinds. Required presence is
`0x50107`: Generation, Endpoint, Operation, local trace, Instance rank and raw
request. Only optional result bit 15 may be added. Ordinary request/invocation
bits 9/10 and their fields remain absent/zero.

| Offset | Field | Meaning |
|---|---|---|
| 216 | data0, u64 | Original actual boundary number, 100–110 |
| 224 | data1, u64 | Forward mode in low u32; output requirement in high u32 |
| 232 | data2, u64 | DP layout in low u32; invalid mask in high u32 |
| 240 | data3, u64 | Received layer ordinal |
| 248 | data4, u64 | Received physical payload rows |

The invalid mask must be nonzero and exactly equal to the invalid fields:
bit 0 for forward mode outside `{1,2,4}`, bit 1 for output requirement outside
`{1,2}`, bit 2 for DP layout outside `{0,1,2}`. Higher bits are zero. Legal
enums with invalid rows/topology still use ordinary kinds 100–110.

Role 1 (Instance) permits boundaries 100–102 and 107–110. Role 2 (AtnAgent)
permits 103–106. Result is absent before 106, present at 106–110, and stores
the actual locally known code 0–3. Invalid enums do not imply ProtocolMismatch:
an actual Shutdown result remains 1. Instance Closed preserves a previously
observed effective result; without one, it records the actual close result.
Raw time zero is a recorded timestamp, not an absent boundary.

The Source supplies Event sequence exactly once at reservation. The existing
successful Claim supplies Operation identity. Local trace counts original
request begins independently of Event admission or loss. An anomaly is one
Event on the same bounded Source, with the same loss/quality semantics.

## Reconstruction and report

Deduplicate by observer domain, local trace and **original boundary**, not kind
150 alone. Different boundaries are distinct observations. Conflicting repeated
request/result facts, conflicting Event identities and ordinary/anomalous facts
for the same operation fail reconstruction. Each field citation retains the
lowest qualifying Event identity. Missing events never invent missing facts.

An invalid enum cannot be represented by the old typed Record, so that Record
is omitted. Its evidence entry remains observed, with `invalid_request`; it is
not an unobserved operation and is not `legacy_fact_missing` merely because its
enum is invalid. Existing missing-boundary, applicability, zero-timestamp and
conflict rules still apply. Terminal evidence alone cannot establish completeness.

Every report entry adds:

- `invalid_fields`: sorted field names from `forward_mode`, `output_requirement`,
  `dp_row_layout`; empty for ordinary/Fabric/unobserved entries.
- `raw_request`: null, or an object with integer `layer_ordinal`, `payload_rows`,
  `forward_mode`, `output_requirement`, `dp_row_layout`. All are original values.
- `terminal_boundaries`: an array in boundary-number order, containing only
  observed `result_published`, `result_observed`, `result_acknowledged`, `closed`.
  Each object contains `boundary`, `raw_time_ns`, `result_code`, including zeros.

These fields are constructed in the existing budgeted report slab. Report and
Record memory stays charged while retained; successful file publication does
not free another holder's memory. The paired file hash, atomic publication and
shared disk budget are unchanged. Transfer failures map to public `io_error`;
uncertain CUDA completion retains the Source, copy target and charges.

## Integration points

`contract.hpp`, `event.hpp` and `event.cpp` own exact wire definitions and
validation. `transportevent.hpp` maps production Hooks explicitly;
`transportrecording.cuh` owns only current facts and original local counts.
`transport_observer.cu` owns the production preparation/read/export lifecycle.
AtnAgent borrows Fabric's DeviceChannel and process service; Instance uses the
same components for its own Source. Neither uses the old Device history array.

`legacy.cpp`, `legacyrecord.cpp`, `evidence.cpp` and `evidencejson.cpp` implement
boundary deduplication, compatible Record projection and the report extension.
`fixturesv5.hpp` and `transportrecording_test.cpp` compare encoding with literal
independent bytes. The fixed v4 fixture is retained and explicitly rejected by
the v5 reader. Runtime/Device tests cover actual adapters and Source loss;
writing these tests does not establish that they have run.
