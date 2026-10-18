# Agent guidance for QKVS

This file tells coding/research agents how to work in this repository. The
canonical scientific context is in [README.md](README.md). Read it first, then
read the active stage's preregistration and configuration before proposing or
executing an experiment.

## Role and decision authority

- You are an **implementation and measurement agent**, not the final arbiter
  of scientific conclusions. The human owner and independent reviewer approve
  hypothesis changes and progression between research stages.
- Preserve the original hypothesis: **causal, bounded-cost selection of useful
  historical KV information**. Do not relabel exact-coordinate accuracy,
  gradient magnitude, or decreasing training loss as success on that hypothesis.
- When a preregistered stop condition fails, **stop** and return the raw
  evidence, smallest plausible diagnoses, and a bounded proposed next test.
  Do not secretly construct the next version, vary thresholds, increase model
  capacity, or redefine success to maintain momentum.
- Don't implement speculative features merely because they appeared in a
  conversation. A new research question requires an explicit protocol and
  authorization, not an incremental patch to a failed gate.

## Before writing code

1. Read **README.md**, then the active stage's `docs/*preregistration*.md`,
   configuration, relevant tested implementation, and previous raw results.
2. State which hypothesis is being tested, what inputs are available at
   *inference*, what measurement can falsify the claim, and the existing
   baselines and stop conditions.
3. Identify any causal leakage, answer shortcuts, mismatched attention budget,
   source-revision ambiguity, or insufficient downstream oracle capability.
4. If the protocol is missing, inconsistent, or was invalidated, **request
   review before executing**. No experiment should be rescued after outcomes
   are observed by silently changing its success criteria.

## Execution invariants

- **Strict single-pass autoregression**: decisions for token `t` may not
  depend on future token states; no retrospective changes to earlier token
  attention. Historical directory entries commit only after block closure.
- **Bounded lookup**: S cannot scan all historical K/V or an index growing
  with context length. Offline teacher selection is allowed for supervision
  but cannot leak full-history information into the deployable route.
- **Routing only**: directory keys and target labels never enter Q/K/V, reader
  residuals, or output logits. S exposes only selected historical addresses;
  attention retrieves values from actual cached K/V.
- **Matched controls**: same reader, causal timing, data, evaluation scorer,
  and budget for baseline versus learned policy. Audit batch/incremental
  equality under the *actual* checkpoint/backend, not only toy masks.
- **Train/evaluation separation**: reserve held-out data before training;
  use validation for checkpoint selection, not the evaluation set. Preserve
  per-example paired results, seeds, hardware/environment and Git SHA.
- **Stop before selector training** if the stage's causally realizable
  oracle cannot demonstrate useful retrieval. A teacher with earlier or
  full-history access is not a substitute.

## How to report work

Keep reports short and high-signal. Every gate report must give:

- **Protocol:** precommitted hypothesis, source revision, data/model versions,
  primary metric, baselines, and threshold.
- **Results:** completed sample count, raw per-example artifact, paired
  uncertainty interval, correctness/leakage checks, and observed pass/fail.
- **Diagnosis:** what the data establish, what they do not establish, and
  distinct plausible failure modes. Separate source-code correctness,
  representational capacity, identifiability, optimization, and throughput.
- **Decision:** proceed only when the predeclared gate passes; otherwise stop
  and request arbiter review. Do not conflate 'code implemented' with 'tested'
  or 'test passed' with 'research hypothesis confirmed'.

## Current stage and repository hygiene

Stage A V3 passed the narrow **annotated-block utility** gate; it did **not**
train a deployable selector. Stage B investigates an **eight-slot causal
semantic-address directory**. Follow
[the Stage B preregistration](docs/stage_b_preregistration.md); Gate 0
must establish *final-token, layer-1* causally aligned oracle utility before
S training. Verify the latest artifacts rather than assuming status from this
snapshot. The stage's exact thresholds and source hashes take precedence over
this guidance.

Prefer one canonical integration branch (`main` once established), short-lived
feature branches, and preserved experiment commit SHAs. Do not squash or rewrite
commits referenced by recorded results. Avoid proliferating markdown files or
restating mutable protocol values across documents. Keep operational commands
with the relevant experiment or main README; do not turn this file into a
second research report.
