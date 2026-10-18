# Agent guidance for QKVS

Read [README.md](README.md) before changing the harness. QKVS is a research instrument for causal,
bounded historical K/V allocation. It is not a collection of experiment-specific runners.

## Authority and scientific scope

- The human owner and independent arbiter authorize result-bearing experiments and changes to
  hypotheses, thresholds, datasets, or progression gates.
- Codex may implement and verify the harness with deterministic CPU-only synthetic tests. Codex
  must never initiate CUDA execution, including indirectly through tests or subprocesses.
- A draft preregistration or configuration is not authorization to execute it.
- Do not interpret loss reduction, selector accuracy, nonzero gradients, or a passing harness test
  as evidence for the QKVS research hypothesis.

## Architecture

- `qkvs.harness` owns configuration validation, lifecycle, checkpointing, resumption, progress,
  source identity, and artifact generation.
- `qkvs.model` owns the frozen causal reader, K/V cache, and actual attention accesses.
- `qkvs.selection` owns proposal policies and direct bounded proposal-to-window resolution.
- `qkvs.metrics` owns shared per-token, retrieval, paired-utility, and uncertainty measurements.
- `qkvs.__main__` is the only human-facing command dispatcher.

Adding an experiment should normally require a configuration, a compatible component behind these
interfaces, and targeted tests. Do not add a top-level runner, duplicate model loader, or separate
metric implementation.

## Invariants

- Every decision is single-pass autoregressive and uses only current or past state.
- Rolling attention is mandatory.
- Historical reads are counted as token occurrences; overlap and duplicates consume budget.
- Deployable selection performs no O(N) history scan. Dense teacher work is offline only.
- K/V values are retrieved from the reader's real cache and every actual index is instrumented.
- Proposal count, window width, and historical occurrence budget remain separate configuration
  values.
- Temporal gamma and spatial beta are distinct controls. Unsupported modes or combinations fail
  validation rather than being ignored.
- Recurrent selector state has fixed size independent of context length.
- Matched conditions share stream, reader, timing, scoring, and budget contracts.

## Workflow

1. `python -m qkvs validate --config <config>` checks schema and execution readiness without
   importing the model runtime or initializing CUDA.
2. CPU tests verify harness behavior, causality, access accounting, resumption, and source identity.
3. An independent arbiter freezes a result-bearing protocol and explicitly authorizes it.
4. The human owner invokes `python -m qkvs run ...` for any GPU execution.
5. `status` and `summarize` read artifacts without model execution.

Artifacts live under `artifacts/results/<run-id>/result.json` and record the source revision,
configuration hash, environment, raw token measurements, actual K/V accesses, and derived summary.
Resume only when source and configuration identity match. Write checkpoints atomically and never
rewrite a prior experiment to change its outcome.
