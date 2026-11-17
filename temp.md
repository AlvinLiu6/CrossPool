## Summary

**Merge dependency: merge this PR only after C0 (#103) and A1 (#104) have both merged, in that order. The required merge order is #103 → #104 → this PR.** This branch builds on C0's shared observability contract and reference fixtures and A1's offline artifact reconstruction.

Add deterministic single-Host clock calibration to `xpool.observability`. Convert contract-v4 paired clock samples into conditional Host/Device time mappings, conservative uncertainty intervals, duration and drift estimates, and structured JSON reports. Preserve raw evidence and provenance; insufficient evidence produces an explicit unavailable result instead of a guessed timestamp.

## Design

- Implement the [calibration contract](docs/plans/unified-timeline-observability/calibration.md) using A1's verified `Reconstruction` and explicit `SamplerDeclaration` inputs. Expose `build_calibration`, `calibrate`, `map_time`, `map_event`, `estimate_duration`, `estimate_drift`, and `build_calibration_report` through the Python API.
- Identify samplers by Run, Host Producer, Device clock domain, and raw Epoch. Keep processes on the same physical Device isolated. Retain `PairedClockSample` outcomes, attempt numbers, source locations, declared ranges, load phases, and optional measured diagnostics. Reject invalid or contradictory input through structured `CalibrationError` values.
- Use fixed ten-attempt blocks: eight attempts fit an affine feasible region, while positions 4 and 9 provide held-out validation. Compute constraints, vertices, estimates, and bounds with exact rational arithmetic; require an admissible positive slope and successful validation. Preserve incomplete blocks, sampling gaps, failed attempts, clock jumps, and other unavailable-model reasons.
- Restrict mappings to their inclusive fitted support. `CalibrationIndex` preserves gaps and Epoch boundaries and requires exactly one applicable model; overlapping models return `ambiguous_model`. Do not extrapolate, interpolate, remove outliers, or silently select another model. A decreasing Device clock invalidates the affected and subsequent blocks in that Epoch.
- Return exact estimates and conservative bounds alongside integer nanoseconds, using ties-to-even estimates and outward-rounded bounds. Propagate duration and drift intervals without clamping negative values or assuming statistical independence. Timing uncertainty remains separate from protocol causality.
- Generate deterministic `CalibrationReport` JSON with decimal-string rational components, samples, model provenance, validation results, window and sampling-interval distributions, uncertainty, drift, and optional measured overhead. Report continuous-time coverage and retained-event coverage separately, preserving unknown denominators and uncovered spans.

## Validation

- `uv run --no-sync xtest run --suite unit --suite integration tests/suites/unit/observability tests/suites/integration/observability tests/suites/unit/test_cext.py` — 349 unit tests and 37 CPU integration tests passed, with no skips or failures. Includes fixed calibration vectors, declaration and sample validation, exact arithmetic and rounding, support and Epoch isolation, duration/drift bounds, report coverage, wire enums, and a published-artifact-to-report round trip that verifies source files remain unchanged. Both stages ran with `devices=none`.
- `uv run --no-sync ruff format --check src/xpool/observability tests/suites/unit/observability tests/suites/integration/observability tests/support/observability.py` — passed; 31 files already formatted.
- `uv run --no-sync ruff check --no-fix src/xpool/observability tests/suites/unit/observability tests/suites/integration/observability tests/support/observability.py` — passed.
- `uv run --no-sync ty check` — passed across the repository.
- `git diff --check feat/observability_a1...HEAD` — passed.

These checks validate the offline implementation and its contract agreement. They do not establish real sampler accuracy, CUDA/MPS or multiprocess timing behavior, serving overhead, or two-device Dense/MoE qualification. No C++/CUDA source changes are introduced; ASAN/UBSAN were not run for this validation.

## Documentation

- Extend [Devkit](docs/designs/devkit.md#offline-clock-calibration) with the implemented numerical algorithm, declarations, model ownership, mapping and duration semantics, coverage definitions, reports, limitations, and Python examples.
- Update the [README](README.md#validation-and-development) and [design index](docs/designs/README.md) to expose the offline calibration capability.
- Add Paired Clock Sample to [domain terminology](CONTEXT.md), distinguishing observation samples from synthetic FFN resource-sizing samples.
- Preserve the inherited shared contract and reference fixtures at their existing paths; this PR does not relocate fixtures or rewrite C0's plan documents.

## Risks and follow-ups

- Contract-v4 inputs and A1 reconstruction are prerequisites. Merge #103 first, then #104, and only then this PR; this PR is not an independent replacement for either dependency.
- Calibration bounds are conditional on the declared segment-affine bracketing model. Held-out consistency and synthetic vectors do not provide independent hardware timing ground truth or an all-event accuracy guarantee.
- Coverage is limited to declared observation intervals and retained canonical events. It cannot recover lost events or infer unobserved runtime coverage; missing scoped counters and overhead measurements remain unknown.
- Runtime sampling, Native/Python Host-clock consistency measurements, calibration CLI integration, and Perfetto export remain separate work. Real idle/serving/stress, MPS, multiprocess, and multi-device calibration qualification requires dedicated execution evidence.
