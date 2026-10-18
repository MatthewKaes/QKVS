# Stage B execution decision

## Decision

**Stop before Gate 0.** The preregistered full-length batch-versus-incremental causal-equivalence
audit failed. The causal-oracle dataset evaluation did not run, and the eight-slot semantic
directory was not constructed or trained.

The unmodified raw result is preserved in
[`stage_b_full_length_causal_equivalence.json`](../artifacts/results/stage_b_full_length_causal_equivalence.json).
It was produced from source revision `43f510a5fedbb90f9895e72aa530f97930f54076` with a clean tracked
source tree. The artifact contains the complete frozen configuration, source-file hashes, dataset
checksum, input-token hash, and raw measurements. Those recorded hashes match the repository at
that revision.

## Result

The audit evaluated example 0 at 7,263 tokens on CUDA in BF16. It compared batched execution with
token-by-token cached incremental execution for both the rolling mask and the causally aligned
oracle mask.

| Mode | Hidden max error | Limit | Logit max error | Limit | Last-hidden cosine | Minimum | Top-1 match |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rolling | 15.75 | 0.002 | 0.53125 | 0.0001 | 0.999763131 | 0.999999 | Yes |
| Causal oracle | 15.75 | 0.002 | 0.390625 | 0.0001 | 0.999762058 | 0.999999 | Yes |

The gate required every numerical bound and the top-1 check to pass. Matching top-1 tokens do not
override the three failed numerical conditions. The largest hidden-state error exceeded its bound
by 7,875 times; the rolling and causal-oracle logit errors exceeded their bound by 5,312.5 and
3,906.25 times respectively.

The batched runs took 0.382 seconds for rolling and 0.295 seconds for causal oracle. Their
incremental counterparts took 62.843 and 62.159 seconds. These timings describe the audit paths;
they are not a preregistered throughput comparison.

## Interpretation boundary

Both mask modes have the same maximum hidden-state error and nearly identical mean hidden-state
errors. That pattern suggests the discrepancy lies in execution behavior shared by the batched and
incremental BF16 paths rather than in the oracle window alone. The current artifact does not
localize the error by token or layer, so it cannot distinguish ordinary long-sequence BF16 kernel
drift from a defect in the incremental reference.

The failure must not be relabeled as semantic equivalence based on matching top-1 output. The
thresholds were frozen before execution, and the protocol states that failure blocks subsequent
work. Any diagnostic amendment or revised numerical criterion now requires arbitration as a
substantive protocol change. Until then:

- do not run causal-oracle Gate 0;
- do not implement or train `S`;
- do not change directory capacity, retrieval layer, or execution topology; and
- do not reuse this failed result as positive Stage B evidence.

## Authorized numerical diagnostic

The arbiter subsequently authorized one bounded diagnostic investigation without reopening Stage
B or changing the failed artifact. The first implementation is frozen in
[`stage_b_numerical_diagnostic.json`](../configs/stage_b_numerical_diagnostic.json). It completed
both BF16 conditions at revision `99e48cb`, then exceeded the 11.53 GiB device while constructing
the FP32 trace. Its partial raw result and both OOM observations are preserved.

The replacement
[`stage_b_numerical_diagnostic_v2.json`](../configs/stage_b_numerical_diagnostic_v2.json) changes
only trace storage: batched layer states stream directly to CPU, the incremental trace moves to CPU
before subsequent replays, and the duplicate final trace stack is eliminated. It reruns all four
conditions under one new source revision rather than importing the two earlier measurements. Both
versions use the same example and input-token hash and record:

- the earliest rolling batch/incremental divergence at every decoder layer and token scale;
- BF16 default-SDPA, BF16 math-SDPA, FP32 math-SDPA, and FP32 eager conditions;
- uncached-prefix, cached-incremental, key/value-cache, mask-structure, and future-token
  counterfactual measurements; and
- final four-label losses and causal-oracle-minus-rolling utility under both execution paths.

This diagnostic has no pass condition into Gate 0. Its purpose is to distinguish precision/backend
effects from cache or masking defects and, only if the evidence justifies it, support a separately
reviewed replacement numerical-validation protocol. Gate 0, `S`, and directory changes remain
blocked.

V2 reports its current phase, condition count, elapsed time, and checkpoint completion directly.
Run one checkpointed condition per invocation. Repeating the same command advances to the next
frozen condition and atomically extends the same V2 raw artifact:

```bash
HF_HOME="$PWD/artifacts/cache/huggingface" HF_HUB_OFFLINE=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m qkvs.stage_b_numerical_diagnostic \
  --diagnostic-config configs/stage_b_numerical_diagnostic_v2.json \
  --device cuda \
  --max-conditions-this-run 1
```

## Numerical diagnostic result

V2 completed all four conditions at revision `a30d0a1`. The raw artifact is
[`stage_b_numerical_diagnostic_v2.json`](../artifacts/results/stage_b_numerical_diagnostic_v2.json),
SHA-256 `4544aea20122069c2d6e194ea33a2cf86bd345ae026078b0220a3c42b1223164`.

| Maximum observed discrepancy | BF16 | FP32 |
|---|---:|---:|
| Any rolling token/layer hidden value | 15.75 | 0.001911 |
| Final hidden value | 2.5 | 0.000282 |
| Full-vocabulary logit | 0.53125 | 0.0000858 |
| Four-label gold-loss delta | 0.1236 nat | 0.0000119 nat |
| Causal-oracle utility-path delta | 0.0760 nat | 0.00000244 nat |

Embeddings match exactly. In both BF16 conditions, divergence exceeds `0.01` at layer 0, token 0
and exceeds `1.0` by token 1 after subsequent layers. Forcing SDPA math does not remove it. FP32
math and eager both remain below `0.002` across the complete rolling token/layer trace and agree on
four-label loss within `8.6e-6` nat. This localizes the large discrepancy to low-precision,
shape-dependent execution rather than a particular SDPA backend.

Mask structure is exact in all four conditions: no future cell is visible, the causal oracle adds
exactly 128 cells at the final layer-1 retrieval row, and every other layer matches rolling. Four
same-shape future-token counterfactuals per condition leave all earlier states bitwise unchanged.
Every cache layer reaches exactly 7,263 entries. FP32 uncached-prefix versus cached-incremental
hidden errors remain below `0.00051`; BF16 uncached-prefix versus full-batch errors reach 2.5 even
without cache reuse. These checks provide no evidence of causal-mask or cache-length corruption.

The raw `layer_zero_batch_vs_cache` content comparison is invalid: its batched reference omitted
the layer-0 attention input normalization. It is excluded from the diagnosis, and the diagnostic
helper is corrected after the result-bearing revision with a CPU regression test. The preserved raw
artifact is not modified.

BF16 matching top-1 labels are not enough to restore the original equivalence gate. Its four-label
loss sensitivity is larger than the frozen 0.05-nat Gate 0 utility margin and changes the sign of
single-example causal-oracle utility between paths/backends. Stage B therefore remains blocked.

## Replacement numerical-validation proposal

The evidence justifies replacing one mixed numerical gate with separate causal, FP32 mechanical,
and BF16 decision-robustness checks. The complete review-only proposal is
[`stage_b_numerical_validation_replacement_proposal.json`](../configs/stage_b_numerical_validation_replacement_proposal.json).
It is **not authorized for execution**.

The proposal excludes calibration example 0 and preregisters examples 1, 2, 4, and 6, which are the
first post-calibration examples spanning all four answer labels and four distinct target positions.
It requires exact mask/future invariants, full-length FP32 batch/incremental validation under math
SDPA and eager attention, and explicit BF16 four-label sensitivity reporting. Because no honest
post-hoc BF16 tolerance can show that a 0.076-nat utility change is irrelevant to a 0.05-nat gate,
any later Gate 0 would have to pass independently under both batched and incremental BF16 paths.
The paths may not be averaged or selected after inspection.

Independent review is required before executing this replacement or Gate 0. `S`, directory, and
architectural work remain prohibited.

## Replacement numerical-validation authorization

On 2026-10-17, the independent arbiter approved
`stage_b_numerical_validation_replacement_v1` at repository revision
`848ed3e1fbc39c07bc0d81136defc2f1702e7d2f` for implementation and execution of the replacement
numerical validation only. The frozen validation examples, causal invariants, FP32 thresholds,
BF16 sensitivity measurements, and independent execution paths remain unchanged.

The approved validation must also compare corrected layer-0 batched and cached K/V projections on
the actual Qwen checkpoint at full context length under each execution configuration. It verifies
cache order, positions, shapes, and projection semantics and reports numerical differences without
introducing a post-hoc tolerance. The original failed audit and diagnostic artifacts remain
immutable. Any frozen validation failure stops execution without parameter changes or a rescue
rerun.

This authorization does not approve Gate 0, causal-oracle dataset evaluation, S training,
directory-capacity changes, or heat-field implementation. A completed replacement validation must
return its raw measurements, source revision, and observed outcome for a new independent decision.

The replacement runner executes all remaining validation units in one invocation and atomically
checkpoints the result before and after every condition. If the process is interrupted, rerunning
the same command records the interrupted attempt and resumes the same frozen unit from the last
complete checkpoint:

```bash
HF_HOME="$PWD/artifacts/cache/huggingface" HF_HUB_OFFLINE=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m qkvs.stage_b_numerical_validation \
  --proposal configs/stage_b_numerical_validation_replacement_proposal.json \
  --device cuda
```

Read-only progress is available without initializing CUDA:

```bash
.venv/bin/python -m qkvs.stage_b_numerical_validation --status
```

The default status is a concise progress summary for terminal use. Add `--json` when a machine-readable
status record is required.
Status inspection remains read-only when the current checkout differs from the measurement source;
it reports both revisions, the current tracked-tree state, and the artifact SHA-256 instead of
rejecting the preserved result.
