# Initial heat-field experiment preregistration draft

## Status

**Draft for independent review. Not authorized for execution.** The configuration at
`configs/heatfield_initial.json` is a concrete review target, not an established QKVS design or a
license to train a selector. Codex must not initiate its CUDA run.

## Question

Can a bounded, causal proposal policy use current hidden state and bounded per-layer recurrent state
to select historical K/V windows that improve next-token cross-entropy over rolling-only and fixed
offset controls at the same historical token-occurrence budget?

The claim is falsified if the learned policy does not improve paired downstream utility, violates
causality or the occurrence budget, relies on an O(N) inference search, or loses its utility when
teacher-only inputs are removed.

## Proposed protocol

- Frozen Qwen2.5-0.5B-Instruct reader at the pinned configuration revision.
- Long-context reading streams with article-disjoint train, validation, and evaluation partitions.
- Mandatory 256-token rolling attention in every condition.
- Eight proposals of width 32 and exactly 256 historical token occurrences per layer.
- Rolling-only, matched fixed-offset, and learned heat-field conditions through the same reader
  execution path.
- Learned gamma controls temporal persistence; learned beta controls proposal spread. Their
  parameters and ablations remain separate.
- Deployable S sees only the current causal hidden state and bounded recurrent state. Offline
  teachers may inspect dense history only to produce supervision and never enter reader logits.
- Primary metric: paired next-token cross-entropy. Secondary metrics: attention coverage, overlap,
  spatial distribution, retrieved occurrences, temporal stability, and responsiveness.
- Per-stream raw measurements and deterministic paired bootstrap intervals are mandatory.

## Controls and stops requiring review

Before authorization, the arbiter must freeze the dataset source and split, learned-policy
parameterization, optimizer, compute budget, checkpoint selection, utility threshold, coverage
target, ablations, and stop conditions. The current draft deliberately contains no success
threshold or training loop.

Execution stops on any future access, historical occurrence above budget, nonlocal scan by the
deployable selector, source/configuration mismatch on resume, nonfinite measurement, or failure of
the prerequisite causal-oracle headroom test selected by the arbiter.

Passing the CPU harness verification does not support the heat-field hypothesis. It establishes
only that the shared execution, access accounting, measurement, and artifact lifecycle behave as
specified on synthetic streams.
