# QKVS — Causal, utility-driven sparse memory retrieval

> **Research prototype, not a validated replacement for full attention.**
> This README is the durable **research charter**: it defines the question, invariants,
> evidence, and method. Exact experiment settings and stop conditions belong in the
> [current stage preregistration](docs/stage_b_preregistration.md), not in this charter.
> Coding agents should also read [AGENTS.md](AGENTS.md).

## What we are trying to learn

**Can a decoder-only autoregressive model use a bounded amount of historical
attention to preserve most of the *useful* information in arbitrarily long
contexts, without searching all historical KV entries at inference?**

QKVS adds an independently learned **S** (selection) projection alongside Q,
K, and V. Each layer keeps a dense rolling working-memory window of `L` tokens
and may additionally attend to at most `R` historical token occurrences through
`S`-selected windows. Layer `k + 1` can make a new selection conditioned on
information retrieved by layer `k`. Windows may overlap or repeat, and historical
KV entries are not rewritten.

The original intuition combined a substantial local **working memory** with a
small, adaptive, possibly multistage set of historical samples, inspired in part
by multiscale/Kawase style sampling. **S is meant to allocate attention to
*valuable evidence*, not reproduce arbitrary numeric pointers perfectly.**

For a fixed selection budget and a fixed-capacity index of `M` entries, the
intended *incremental decode* work per token is bounded by `O(L + R + M)` for
fixed model dimensions and layer count, independently of total history length
`N`. Storing the ordinary historical KV cache still takes `O(N)` space.
Neither this asymptotic property nor the existing Python implementation
establishes production wall-clock superiority over optimized dense attention.

## What successful retrieval means

We measure the effect of exposing a historical block to a reader, rather than
whether the reader reconstructs an annotated coordinate:

\[
U(b \mid A, x, y)
  = \ell(y \mid x,\,\mathrm{rolling}+A)
  - \ell(y \mid x,\,\mathrm{rolling}+A+\{b\}).
\]

`U > 0` means the block reduces answer loss. Utility is **conditional** on
other selected blocks: evidence can be redundant, complementary, or
interfering. A block containing an annotated answer need not be the only useful
block. Address-hit rate is a diagnostic; held-out **downstream utility** is the
scientific objective.

Report negative-log-likelihood/cross-entropy using **nats** (natural-log units)
and accuracy as separate metrics. For recovered utility, use a **ratio of
dataset-level mean loss improvements**, not an average of unstable per-example
ratios. Use paired uncertainty intervals and inspect individual cases.

The relevant interference in QKVS is **competition in softmax over selected
KV entries**. QKVS does *not* sum the whole history into a single outer-product
memory matrix. Orthogonality of every historical key is not a prerequisite for
successful retrieval, but reading irrelevant candidates can still dilute or
redirect useful attention.

## Non-negotiable inference constraints

1. **Single-pass causal autoregression.** At token `t`, each selection uses
   only state already available by that token and layer. No future-question
   information in historical writes, second question pass, retroactive updates
   to earlier query tokens, or retrospective granting of history access.
   Verify batched causal execution against incremental KV-cache execution.
2. **No context-length-dependent historical search at deployment.** S must not
   scan all historical K/V or growing collections of summaries to select a
   block. A bounded directory is an admissible *positive control*: a constant
   number `M` of semantic-key/address bindings can be searched per query.
   Offline teachers may scan history **for supervision only**.
3. **Routing and reading are separate.** S has its own projections, independent
   of attention Q/K/V. A directory may expose semantic keys and addresses to S;
   it must not pass answer vectors, gold labels, or directory contents into
   the reader's Q/K/V, hidden state, or logits. Only the **selected address**
   enables reading the actual historical KV window.
4. **Causal memory maintenance.** A directory entry is written from causally
   available representations when its block closes, and may be read only after
   commitment. A teacher's full-context hidden state is not deployable input.
5. **Identifiability is a prerequisite, not a parameter-tuning issue.** A
   randomly placed item cannot be located from a query representation that
   contains no location information. The model must preserve semantic-to-location
   information causally, or exploit genuinely predictable document structure.
6. **Matched comparisons.** Hold the reader, task, answer scoring, causal
   timing, selection budget, and source revision fixed when comparing learned,
   rolling, random, recency, annotated-oracle, and other reference policies.

The first research directory uses eight explicitly marked slots to isolate
semantic addressing. It is **not** a proof that eight slots are sufficient at
scale or that a directory is the minimal possible memory representation.
Admission, eviction, and compression are separate unanswered questions.

## Evidence and current boundaries

| Investigation | What the evidence supports | What it does **not** establish |
| --- | --- | --- |
| Direct-address attention mechanics | Causal block selection, bounded selected work, immutable historical KV, overlap and multiplicity, and incremental reference tests | A production-quality fused kernel or decode speedup at the intended large budget |
| Working-memory write/read | Controlled rolling-memory markers store information recoverable through actual cached K/V; selected later layers can read it | Autonomous decisions about what to store, maintain, or retrieve |
| Supervised address learning | A standalone hierarchical head copied eight supplied addresses through 64K in controlled tests; some in-distribution pointer cascades worked | That exact-address regression is useful semantic memory learning; robust extrapolation or integrated convergence |
| Integrated synthetic v1–v5 | Exposed missing gradients, inappropriate shortcut-bearing data, weak reader controls, hard-selection gradient barriers, and addressability failures | Evidence that larger models or more precise coordinates alone resolve selection |
| **Stage A V3 — retrieval utility** | **Preregistered gate passed** on a frozen pretrained reader: selecting an annotated distant block improved loss over rolling, random, and candidate-recency | A trained S, autonomous memory organization, production throughput, or general superiority to full attention |
| **Stage B — causal directory** | An eight-slot semantic-address positive control and a stricter final-token causal oracle have been preregistered and approved for gated evaluation | Successful causal-oracle parity or trained-directory utility at the last reviewed checkpoint |

### Stage A's significant positive result

Using frozen `Qwen/Qwen2.5-0.5B-Instruct`, 24 four-choice reading examples,
~7.3K-token contexts, a 256-token rolling window, and **one** selected
128-token old block beyond the 24-layer rolling receptive field:

| Reader condition | Accuracy | Mean loss (nats) |
| --- | ---: | ---: |
| Full attention | 79.2% | 0.511 |
| Rolling only | 83.3% | 0.688 |
| Rolling + fair random block | 83.3% | 0.525 |
| Rolling + candidate-recency block | 83.3% | 0.675 |
| **Rolling + annotated supporting block** | **91.7%** | **0.305** |
| Rolling + offline best-of-eight block | 91.7% | 0.245 |

The annotated block reduced mean loss versus rolling by **0.383 nat**,
with a preregistered paired 95% bootstrap interval of **[0.137, 0.725]**;
it also beat the matched random and candidate-recency baselines. The support
block reduced loss on 23/24 examples; the offline utility oracle selected it
on 21/24. This is **promising but small-sample, oracle-based evidence**.
The original Stage A oracle made history available to multiple query tokens;
that access schedule is *not* itself deployable using a final-query S decision.
The Stage B **causally timed oracle must pass independently**.

Read the [decision history](docs/retrieval_utility_decision.md),
[raw Stage A V3 observations](artifacts/results/retrieval_utility_stage_a_v3.json),
and [earlier research report](docs/research_report.md) for provenance,
negative results, limitations, and exact source revisions.

## Current experiment: Stage B, not another pointer-regression variant

The next falsifiable question is whether **causally maintained semantic-to-address
bindings** let an S-specific query read useful history. The positive control is
an append-only eight-entry directory:

\[
D_t = \{(k_i, a_i)\}_{i=1}^{8},
\]

where `k_i` is a frozen causally computed block representation transformed by
a small S-owned write projection, and `a_i` is the block address. The query
uses a separate S-owned projection for **eight-way semantic-entry matching**;
its selected address alone enables historical attention. The reader is frozen.

**Execution order is mandatory:**

1. Verify real-checkpoint, full-length incremental/batched causal equivalence.
2. **Gate 0:** show that an oracle selection available only at the *final prompt
   token's layer-1 attention* improves utility over rolling-only. If not,
   **stop before training S**. Compare the original Stage A oracle only as a
   nondeployable timing diagnostic.
3. If Gate 0 passes, train only the preregistered write/query projections on
   article-disjoint training examples. Evaluate held-out downstream utility,
   moved-block counterfactuals, shuffled bindings, retrieval removal, and
   matched random/recency/query-only baselines.
4. Treat nondeployable full-context query representations only as diagnostics.
   Do not rescue a failed causal result by introducing a second pass.

**The exact configuration, seeds, layers, metrics, and stop thresholds live in
[Stage B preregistration](docs/stage_b_preregistration.md) and
[its config](configs/stage_b_semantic_directory.json).** This README must not
silently amend them. At the creation of this charter the Stage B design was
approved but the oracle dataset and selector-training results were pending;
consult the latest raw artifacts for the actual status.

Later possibilities—learned admission and eviction, recurrent/compressed directory
state, several conditioned retrievals per layer, larger working windows, longer
contexts, and optimized Triton/CUDA kernels—require separate evidence and gates.
They are not preauthorized follow-ups to a failed Stage B control.

## Research method: falsify rather than embellish

- **One question at a time.** Define the deployment information available, null
  baseline, decisive observation, alternative explanations, and stop condition
  *before* adding code or doing expensive training.
- **Preregister and preserve.** Freeze splits, seeds, masks, models, metrics,
  confidence method, and pass/fail gates before examining outcomes. Record
  protocol amendments separately. Never erase a failed or invalid experiment.
- **Reader before router.** Demonstrate downstream headroom using a
  *causally realizable* oracle before optimizing S. A weak oracle reader is not
  evidence for a weak router.
- **Diagnose the right failure:** (a) information unavailable to the router,
  (b) unusable causal block representations, (c) selector optimization,
  (d) downstream reading, or (e) kernel performance. Do not conflate them.
- **Treat training telemetry as diagnostics.** Record per-loss gradient paths,
  norms *before* clipping, clipping ratios, parameter updates, entropy and
  uncertainty. Nonzero gradients or falling loss do not prove useful routing.
- **Check shortcuts and leakage.** Use moved-location counterfactuals, shuffled
  key/address bindings, removal tests, truly separated evaluation articles,
  and incremental causal equivalence. Labels or teacher features may not
  arrive at deployable selection time.
- **Avoid moving targets.** Distinguish proposed, implemented, and measured
  work. Record exact code revision and raw per-example results. Do not change
  multiple factors after a failure and reinterpret it as a clean test.
- **Stop is an acceptable result.** If an approved gate fails, report what it
  actually falsifies and request independent review. Do not automatically add
  parameter count, loss complexity, sampling geometry, or new kernel work.

### Source-of-truth hierarchy

1. **This README:** long-lived research question, constraints and standards.
2. **Current preregistration and config:** frozen specifics and stop criteria.
3. **Tested source code:** what actually executes, including deviations.
4. **Revision-bound raw results:** what was measured, including failures.
5. **Historical reports and chats:** rationale only; never override current
   evidence or retroactively alter a frozen protocol.

If these disagree, surface the discrepancy to the reviewer; do not silently
choose whichever interpretation allows work to proceed.

## Working in this repository


Install the pinned local environment and run the existing tests:

```bash
uv sync
uv run pytest
uv run python -m qkvs.environment
```

Research measurements run on an NVIDIA RTX 5070 (~12 GB VRAM). The host has
experienced sporadic NVIDIA/PCIe driver failures (including Xid 79); use
short, checkpointed CUDA workloads and preserve raw partial results. Consult
the relevant experiment module, preregistration, and README for its precise
invocation. Do not merge results from different source revisions into one
artifact.

Keep a **single canonical Git integration lineage** (prefer `main` once
created), use short-lived task branches, preserve result-bearing commit SHAs
and avoid unnecessary squashing or rebasing. Branch names are not the source
of experimental truth.

For the earlier reference implementation and tests, see
[the architecture specification](docs/architecture.md). To understand why
the research reset occurred, see [the historical report](docs/research_report.md).
