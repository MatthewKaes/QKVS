# QKVS — Causal heat-field retrieval for transformers

> **Independent research prototype — not a validated replacement for dense attention.**
> QKVS investigates whether a decoder-only autoregressive transformer can **learn where to spend a bounded historical attention budget**, guided by semantics and a compact causal memory of previous selection states, without scanning the entire historical KV cache at inference.
>
> This README is the enduring **research charter and conceptual specification**, not an experimental preregistration. See [AGENTS.md](AGENTS.md) for agent operating rules, the [Stage B preregistration](docs/stage_b_preregistration.md) for frozen experimental details, and the [execution decision](docs/stage_b_execution_decision.md) for the latest recorded results and stops.

## The idea in one picture

![Comparison of five historical attention strategies: full-context dense attention, partial or rolling local attention, isolated sparse attention, combined local and sparse attention, and QKVS heatmap-guided retrieval.](docs/images/attention_retrieval_methods_comparison.png)

**Figure 1 — Historical attention access patterns (conceptual).** The timeline runs from old tokens on the left to the current query on the right. Blue/green/orange highlighted positions indicate history accessible to attention, not a model's measured attention weights. The last panel introduces a **predicted heat field** whose peaks describe where historical information *might be useful*. QKVS converts a bounded number of those predictions into actual KV windows and always retains the local rolling window. **It does not build or scan the depicted full-width heatmap at inference.**

| Strategy | What the current token can read | Main limitation or distinction |
| --- | --- | --- |
| **Full context / dense** | Every preceding KV token | Per-query attention work grows with history; more access does not guarantee better predictions. |
| **Partial / rolling** | A fixed recent window of `L` tokens | Older evidence becomes unreachable through direct attention, though representations can sometimes relay it across layers. |
| **Sparse context** | Selected isolated historical tokens or windows | Historical coverage without a guaranteed local working-memory region. |
| **Partial + sparse** | Rolling `L` plus `R` selected historical token occurrences | A *connectivity pattern*; it does not specify how to learn the selection. |
| **QKVS** | Rolling `L` plus bounded windows proposed by `S` | Research proposal: learn the **semantics, spatial layout, and temporal evolution** of useful retrieval. |

The distinction is not “QKVS can read distant tokens while other sparse schemes cannot.” The distinction we want to establish is that **a small learned state can predict a valuable pattern of historical reads** without a context-length-dependent search. The early motivation drew on *multiscale/Kawase-style sampling*: several overlapping, differently positioned samples can collectively cover a meaningful field without inspecting every possible location.

## Target architecture: Q, K, V, and S

Ordinary attention still uses **Q/K/V**. QKVS adds **S**, a separately learned selection path. At autoregressive position `t` and transformer layer `ℓ`:

- **`h_t^ℓ`** is the causally available incoming hidden state.
- **`S_t^ℓ`** describes where to retrieve historical KV windows; it is **not** a replacement for Q/K/V.
- **`L`** is the mandatory local rolling token window.
- **`J`** is the maximum number of sparse historical proposals, with optional inactive proposals.
- **`R`** is the total historical **token-occurrence** budget actually read by those proposals. Repeated/overlapping windows count against the budget.
- **`N`** is total history length. Historical K/V storage remains `O(N)`; direct historical selection must not require an `O(N)` scan.

![Four-stage QKVS inference pipeline: causal hidden state and bounded prior selection state enter S; S generates heat-field peaks; peaks resolve directly to sparse windows alongside a rolling window; ordinary Q/K/V attention reads only those windows.](docs/images/qkvs_retrieval_pipeline_infographic.png)

**Figure 2 — Proposed causal inference path.** The model computes a compact selection policy from available state, converts its proposals **directly** into a bounded number of historical addresses, then reads selected KV entries with ordinary attention. A selected window at layer `ℓ` may influence the hidden representation seen by layer `ℓ+1`, allowing **cross-layer refinement**. A compact prior state may also inform selection at the **next token**, allowing **temporal propagation**. Neither requires an earlier token to be recomputed.

### A collective heat field, not `J` unrelated coordinates

A candidate form for `S` is `J` spatial proposals, each with a center, spread, importance, and optional activity gate:

```math
S_t^\ell = \{(\mu_j,\sigma_j,w_j,g_j)\}_{j=1}^{J}, \qquad
H_t^\ell(x) = \sum_{j=1}^{J}g_jw_j\,\kappa_{\beta}(x;\mu_j,\sigma_j).
```

`H(x)` is a **conceptual heat field over eligible historical positions**. Overlap between its component kernels can express more than a set of independent point predictions:

- A strong peak near the beginning when the query refers to an introduction.
- Several clusters when comparing different historical discussions or examples.
- Concentrated sampling around a topic transition or a recalled event.
- Broadly spaced proposals when evidence is dispersed across a long discussion.
- Locally evolving peaks as newly retrieved information changes the next layer's decision.

**`β` is a possible spatial sharpness/selectivity control:** higher sharpness may concentrate heat around peaks; lower sharpness may spread it. Exact parameterization (kernel width, temperature, or an explicit learned gate) has **not** been validated. The **local rolling window is mandatory and separate** even if a conceptual figure depicts heat near the present.

The implementation must **not** materialize `H(x)` for all `N` positions and then run a top-k scan in deployment. It must map a fixed number of proposal parameters directly to bounded addresses/windows; a dense plot is for explanation and **offline supervision only**.

A proposed **64-slot** allocation study offers substantially more room for clustered and distributed proposals than the earlier eight-slot controls. This is a *capacity hypothesis*, not an experimentally proven minimum. Increasing `J` without holding `R` fixed also changes attention cost: 64 windows of 32 tokens cost 2,048 historical reads, while 64 windows of four tokens cost 256. Compare `J`, window width, and `R` separately. Likewise, eight **directory entries** in Stage B do not imply that eight *heat proposals* are sufficient.

## Temporal propagation is part of the model: γ (gamma)

A principal design goal is for `S` to carry a **lingering retrieval heat signature** forward through autoregressive time. The next token should not have to rediscover the entire relevance pattern from scratch. This is **not merely an optional implementation trick**: learning how persistent or reactive historical selection should be is part of the QKVS research question.

![Causal propagation of a compact selection state over successive tokens, with plots illustrating fast-decaying heat for low gamma, intermediate persistence for medium gamma, and slow-decaying heat for high gamma.](docs/images/temporal_propagation_and_gamma_effects.png)

**Figure 3 — The gamma tradeoff (illustrative).** **Low `γ`** favors new semantic evidence and rapid changes in retrieval focus. **High `γ`** keeps prior heat peaks active, smoothing transitions but potentially retaining stale relevance. The illustration uses a simplified linear decay; the eventual S update may be learned, gated, and nonlinear.

A stable **candidate** recurrence at each layer is:

```math
\begin{aligned}
u_t^\ell &= F_\ell(h_t^\ell,\,S_t^{\ell-1}),\\
z_t^\ell &= (1-\gamma_t^\ell)\,u_t^\ell
              +\gamma_t^\ell z_{t-1}^\ell,\\
S_t^\ell &= G_\ell(z_t^\ell), \qquad 0\leq\gamma_t^\ell<1.
\end{aligned}
```

`z_t^ℓ` is a small **persistent latent heat state**; `S_t^ℓ` decodes that state into spatial retrieval proposals. The first-layer cross-layer input is omitted or initialized causally. `γ` could be fixed, learned per layer, or dynamically gated per token/proposal; those are **alternatives to test**, not existing implementation facts.

- **`γ` controls persistence in time** (reactivity vs. smoothness).
- **`β` controls concentration in historical space** (broad vs. sharp heat peaks).
- The current hidden representation and lower-layer results can **shift or refine** the field, even while temporal state persists.
- Keep only the latest bounded `z_{t-1}^ℓ` / S state per active layer. Do **not** add a historical S vector for every token to the KV cache merely to enable temporal propagation.

For a simplified constant-`γ` linear update, the influence of a past impulse decays as `γ^k` after `k` tokens. A two-tap update using only the current and previous states can still have a **long effective memory through recurrence**; “two inputs” does not mean “two tokens of memory.” If peaks are stored in relative or normalized coordinates, they must be re-anchored as `t` advances or they may drift away from their historical locations. Stable gating, coordinate consistency, and the ability to forget obsolete heat remain open tests.

This design offers **two causal directions of refinement**: *across tokens* via the previous S state, and *across layers* via newly read information. Neither may use future hidden states, reread the full question, or retroactively change earlier token attention.

## Training S: attention coverage **and** downstream utility

**Train the allocation as a whole set. Do not independently reward every proposal for landing on the same high-scoring token.** The underlying concern is that a pure task-reward objective may overfit narrow answer targets or distort model behavior, while an attention-mass objective alone may chase sinks, separators, or redundant evidence. The proposed training paradigm combines **broad teacher attention coverage** with **causally measured predictive utility**, and tests whether they agree.

![Three-stage proposed S training pipeline: a frozen teacher provides dense attention importance and block-level prediction utility; a budgeted set mask discounts overlap; a bounded S learns proposal kernels with differentiable occupancy and emits only bounded hard windows at inference.](docs/images/teacher_guided_heat_field_learning_infographic.png)

**Figure 4 — Proposed training flow, not a completed experiment.** Offline teacher computations may inspect all historical tokens and produce an `O(N)` importance heatmap or a binary `0/1` coverage mask. The deployable S receives **only causally available features** and emits `J` bounded proposals; it never predicts, stores, or scans an `O(N)` array at inference.

### Signal A: teacher attention coverage

A dense teacher can produce historical attention importance `a_i ≥ 0` for token `i`, aggregated across selected heads/layers and queries. This is a **dense proxy** for where a trained model tends to allocate attention. It supplies a spatial learning signal even when individual question labels are scarce.

### Signal B: causal downstream utility

For selected historical windows `B`, evaluate the reader's answer loss with and without those windows under **the same causal timing**:

```math
U(B\mid x,y) = \mathcal L(y\mid\text{rolling},x)
                - \mathcal L(y\mid\text{rolling}+B,x).
```

`U(B) > 0` means the selected set helps; `U(B) < 0` means the set hurts. Utility is **not additive**: complementary blocks may help only together, and a duplicate can change Q/K/V softmax multiplicity. The model must learn **set-level marginal value**, not merely rank individual addresses by a scalar label.

### Explicitly discount overlapping coverage

Let `c_i(B)` be the number of selected windows covering historical token `i`. An illustrative discounted attention-coverage reward is:

```math
C_\alpha(B) = \sum_i a_i\sum_{r=0}^{c_i(B)-1}\alpha^r,
\qquad 0\leq\alpha<1.
```

The first read receives full credit, a second overlap adds only `α` times that credit, a third adds `α²`, and so on. With `α=0` the teacher rewards **unique coverage only**. A very important peak may still justify multiple overlapping windows if the discounted marginal value exceeds reading something else. The **measured downstream utility** decides whether that overlap is actually beneficial.

A differentiable **training surrogate** can use the soft coverage `q_{ji}` of proposal `j` at token `i` and approximate unique occupancy as

```math
p_i = 1-\prod_{j=1}^{J}(1-q_{ji}).
```

This is a useful approximation, not a claim that an integer KV gather transmits gradients. A teacher can generate a sparse binary mask or continuous target from discounted budgeted coverage; training may use set-level distillation, offline utility preferences, or an explicitly justified gradient estimator. **Do not assume answer-loss gradients pass through hard addresses.**

An illustrative future objective mixes normalized coverage and utility while accounting for token/window budget:

```math
\max_{B:\,\mathrm{reads}(B)\le R}
\lambda_A\,\widetilde C_\alpha(B)
+\lambda_U\,\widetilde U(B)
-\lambda_{\mathrm{cost}}\,\mathrm{cost}(B).
```

These terms have different scales; **weights, normalization, curriculum, and estimator are unvalidated design variables**. Teacher coverage should help regularize narrow optimization, but it can introduce biases of its own. The primary experimental question is whether **capturing the teacher's attention mass also recovers useful prediction quality** on held-out tasks. If attention mass and true causal utility disagree, **utility wins**. Benchmark both signals separately before endorsing any mixture.

### Critical training / inference boundary

| Offline training may do | Causal inference must do |
| --- | --- |
| Dense Q/K attention, historical scans, counterfactual block ablations, budgeted teacher set construction | Produce a bounded set of addresses from current causal hidden state and bounded recurrent state |
| O(N) importance maps and binary region targets | **No O(N) heatmap output, softmax, or historical top-k search** |
| Differentiable coverage surrogates and expensive utility estimation | Read at most `R` historical token occurrences plus local `L` |
| Use teacher labels to grade or supervise the selector | Never expose teacher-only history, gold support positions, or future questions as S inputs |

## Non-negotiable invariants

1. **Strict single-pass autoregression.** Token `t` and layer `ℓ` use only information available before their selection/attention operation. No second question pass, future-dependent historical writes, or changes to earlier token states. Validate causal masking and incremental KV-cache execution separately from numerical floating-point equality.
2. **Bounded selection.** For fixed model dimensions and fixed `J`, `L`, and `R`, incremental historical selection and attention work must not grow with total context `N`. Intended per-token work is `O(J + L + R)` (plus fixed-size latent computation); historical KV storage is still `O(N)`.
3. **Independent selection pathway.** S's parameters are distinct from Q/K/V. Q/K/V read **real historical KV entries** after selection. Selector annotations, full-context teacher state, directory values, and answer labels must not bypass the selected attention path into decoder logits.
4. **Causal state evolution.** Cross-token state is updated from current/past representations only; cross-layer refinement follows the model's forward order. A bounded external directory is permitted as an experimental control, not assumed necessary or sufficient for QKVS.
5. **Identifiability.** No model can infer an arbitrary randomized semantic-to-position mapping from a causal query state that does not contain such information. A learned spatial prior can succeed on *predictable structure*; exact arbitrary bindings require causally preserved organization or an allowed bounded index. Test these separately.
6. **Comparable research controls.** Hold task, frozen reader, token budget, causal schedule, answer scoring, source revision, and training/evaluation split constant wherever comparisons require them. Numerical backend changes can affect measured utility and must be reported.

## Evidence: what we have actually established

| Study | Supported finding | Not yet established |
| --- | --- | --- |
| Causal selection mechanics | Reference block selection, overlapping windows, hard gathers, and layer-specific attention paths have been implemented and tested. | Production fused-kernel advantage at the intended budget. |
| KV memory write/read probes | Controlled marker experiments demonstrated information readable through actual cached KV entries. | Autonomous memory prioritization and robust learned selection. |
| Address and pointer experiments (v1–v5) | A standalone supervised hierarchical address decoder copied supplied coordinates; integrated experiments revealed shortcut-bearing tasks, zero gradients through hard selection, and reader failures. | Exact-address regression as a useful semantic retrieval objective. The v1–v5 pointer-recall progression is **stopped**. |
| **Stage A V3 — utility** | **Preregistered narrow gate passed:** a correctly selected distant block improved answer loss versus rolling, fair random, and candidate-recency. | Learned S, single-pass final-token oracle success, or superiority to dense attention in general. |
| **Stage B — strict causal positive control** | The eight-entry semantic directory and final-token layer-1 oracle were preregistered; short CPU incremental tests and more extensive numerical diagnostics exist. | Stage B causal-oracle dataset utility, a trained S, or validated BF16 batch/incremental decision equivalence. The first full-length BF16 numerical gate **failed**. |
| **Heat-field / gamma / dual-signal S** | **Core target design, documented here for future research.** | Not yet a trained, validated architecture or an amendment to the frozen Stage B experiment. |

### Stage A V3: a useful oracle result, with limits

Frozen **Qwen2.5-0.5B-Instruct**, **24** four-choice reading examples, approximately **7.3K** context tokens, `L=256`, and one annotated historical block of **128 tokens**, deliberately placed beyond the 24-layer rolling relay distance:

| Reader condition | Accuracy | Mean answer loss (nats) |
| --- | ---: | ---: |
| Full dense attention | 79.2% | 0.511 |
| Rolling only | 83.3% | 0.688 |
| Rolling + fair random historical block | 83.3% | 0.525 |
| Rolling + candidate-recency block | 83.3% | 0.675 |
| **Rolling + annotated supporting block** | **91.7%** | **0.305** |
| Rolling + best block scanned offline | 91.7% | 0.245 |

The annotated block improved mean loss over rolling by **0.383 nat** (preregistered paired 95% bootstrap interval **[0.137, 0.725]**) and decreased loss on **23/24** examples. The offline best-block oracle agreed with the annotated location on **21/24** examples. These observations **demonstrate useful historical retrieval when the correct region is supplied**; they **do not establish** that a deployment-compatible S can locate that region. Stage A's oracle exposed history to several earlier query-token computations, so the strictly causal final-token oracle must be tested independently.

See [raw Stage A V3 observations](artifacts/results/retrieval_utility_stage_a_v3.json) and [the preregistration and decision history](docs/retrieval_utility_decision.md).

### Current controlled Stage B and numerical validation

Stage B's deliberately narrow positive control is an append-only **eight-entry semantic-address directory** `D_t={(k_i,a_i)}`. It tests whether causal hidden states can write useful semantic-to-location bindings and a small S classifier can select the annotated entry. It is **not the defining QKVS heat-field architecture** and cannot by itself establish semantic spatial allocation at 64 proposals.

Stage B first restricts a causally aligned annotated oracle to the **final prompt token at layer 1**, before any S training. The original full-length BF16 CUDA batch/incremental numerical gate failed. A diagnostic subsequently found close FP32 agreement but larger BF16 execution-shape sensitivity affecting answer loss. **No causal-oracle dataset Gate 0 result or S training result has been established in the source snapshot reviewed for this charter** (main `0905424`). A replacement numerical-validation procedure was proposed; its execution and any later gate outcomes must be checked in the current [Stage B execution decision](docs/stage_b_execution_decision.md) and raw artifacts before further work.

Do not alter frozen Stage B prerequisites, numerical acceptance criteria, training split, or architecture through README edits. A future heat-field experiment requires a **separately approved preregistration** and a causally realizable oracle. Failed or invalid experiments remain part of the evidence.

## Research method: falsify rather than embellish

- **Predefine one falsifiable question at a time.** State what S can actually know at inference, the null hypothesis, meaningful baselines, metric, stopping rule, and observations that distinguish competing explanations.
- **Reader before selector.** Establish useful headroom under the *exact causal timing* of a deployable selection before spending GPU time on S optimization.
- **Freeze before observing.** Preregister sources, splits, masks, seeds, primary comparisons, confidence methods, thresholds, and compute budget; amend transparently before a new experiment. Preserve partial, failed, and invalid artifacts with immutable source SHAs.
- **Diagnose separately:** information/identifiability, block representation, S training and its gradients, downstream reading, softmax interference, numerical backend, and kernel throughput. Improving one does not establish the others.
- **Use controls that defeat shortcuts.** Include rolling, matched random and recency, query-only priors, oracle selection, historical swaps, overlap/coverage baselines, removal of selected blocks, and where relevant shuffled semantic bindings. Evaluation should be unseen at both example and document/article level.
- **Report downstream utility with uncertainty.** Use paired per-example answer losses in **nats** (natural-log units), accuracy, seed variation, and a transparent cost metric; address hit rate and attention coverage remain diagnostics. When reporting recovered utility, use the **ratio of aggregate mean loss improvements**, not the average of unstable per-example ratios.
- **Respect numerical reality.** Compare incremental vs. batched causal execution, FP32 vs. BF16 where appropriate, and backend-specific effects on answer-loss decisions. Matching top-1 outputs alone is not numerical equivalence. Do not relax thresholds to pass a failed audit.
- **Stop at failed gates.** No automatic new address head, larger model, different objective, extra directory capacity, or kernel project to rescue a negative result. Report what failed, what remains uncertain, and the smallest next falsification for review.

### Source-of-truth order for future agents

1. **This README:** goals, architectural intent, invariants, and research standards.
2. **Active preregistration + config:** exact conditions, stage-specific decision gates, and permission to execute.
3. **Tested code:** what actually runs; discrepancies with the protocol must be called out.
4. **Raw result artifacts tied to a Git revision:** what was measured, including failure/uncertainty.
5. **Older reports and conversations:** historical rationale, not authority to modify a frozen experiment.

If these disagree, **surface the conflict and stop**. Do not rewrite results or silently make a stage succeed.

## Repository and reproducibility

```bash
uv sync
uv run pytest
uv run python -m qkvs.environment
```

Research runs target an NVIDIA RTX 5070 (~12 GB VRAM). The Linux host has experienced sporadic NVIDIA/PCIe driver failures (including Xid 79), so runs should be bounded, checkpointed, and resumable. Preserve full configuration, environment, source identity, example-level measurements, and any driver failures. Performance claims need a matched comparison against optimized attention kernels; **O(J+L+R) selection work is an architectural goal, not a measured throughput guarantee**.

Keep one canonical `main` integration lineage, short-lived review branches, and immutable experiment commits. Do not rewrite commits cited by artifacts or multiply documentation copies. The **four diagrams in `docs/images/` illustrate the intended model, not completed Stage B results**.

Further details: [AGENTS.md](AGENTS.md) · [historical architecture specification](docs/architecture.md) · [research report](docs/research_report.md) · [Stage A decision](docs/retrieval_utility_decision.md) · [Stage B preregistration](docs/stage_b_preregistration.md) · [Stage B execution decision](docs/stage_b_execution_decision.md).
