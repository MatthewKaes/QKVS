# QKVS — Causal heat-field retrieval for transformers

> **Independent research prototype — not a validated replacement for dense attention.**
> QKVS investigates whether a decoder-only autoregressive transformer can **learn where to spend a bounded historical attention budget**, guided by semantics and a compact causal memory of previous selection states, without scanning the entire historical KV cache at inference.
>
> This README is the enduring **research charter and conceptual specification**, not an experimental preregistration. See [AGENTS.md](AGENTS.md) for agent operating rules.

## The idea in one picture

![Comparison of five historical attention strategies: full-context dense attention, partial or rolling local attention, isolated sparse attention, combined local and sparse attention, and QKVS heatmap-guided retrieval.](docs/images/attention_retrieval_methods_comparison.png)

**Figure 1 — Historical attention access patterns (conceptual).** The timeline runs from old tokens on the left to the current query on the right. Blue/green/orange highlighted positions indicate history accessible to attention, not a model's measured attention weights. The last panel introduces a **predicted heat field** whose peaks describe where historical information *might be useful*. QKVS converts a bounded number of those predictions into actual KV windows and always retains the local rolling window. **It does not build or scan the depicted full-width heatmap at inference.**

| Strategy                 | What the current token can read                            | Main limitation or distinction                                                                                            |
| ------------------------ | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Full context / dense** | Every preceding KV token                                   | Per-query attention work grows with history; more access does not guarantee better predictions.                           |
| **Partial / rolling**    | A fixed recent window of `L` tokens                        | Older evidence becomes unreachable through direct attention, though representations can sometimes relay it across layers. |
| **Sparse context**       | Selected isolated historical tokens or windows             | Historical coverage without a guaranteed local working-memory region.                                                     |
| **Partial + sparse**     | Rolling `L` plus `R` selected historical token occurrences | A *connectivity pattern*; it does not specify how to learn the selection.                                                 |
| **QKVS**                 | Rolling `L` plus bounded windows proposed by `S`           | Research proposal: learn the **semantics, spatial layout, and temporal evolution** of useful retrieval.                   |

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

A proposed **64-slot** allocation study offers substantially more room for clustered and distributed proposals than earlier eight-slot controls. This is a *capacity hypothesis*, not an experimentally proven minimum. Increasing `J` without holding `R` fixed also changes attention cost: 64 windows of 32 tokens cost 2,048 historical reads, while 64 windows of four tokens cost 256. Compare `J`, window width, and `R` separately. A positive-control directory size does not imply that the same number of *heat proposals* is sufficient.

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

| Offline training may do                                                                                  | Causal inference must do                                                                        |
| -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Dense Q/K attention, historical scans, counterfactual block ablations, budgeted teacher set construction | Produce a bounded set of addresses from current causal hidden state and bounded recurrent state |
| O(N) importance maps and binary region targets                                                           | **No O(N) heatmap output, softmax, or historical top-k search**                                 |
| Differentiable coverage surrogates and expensive utility estimation                                      | Read at most `R` historical token occurrences plus local `L`                                    |
| Use teacher labels to grade or supervise the selector                                                    | Never expose teacher-only history, gold support positions, or future questions as S inputs      |

## Non-negotiable invariants

1. **Strict single-pass autoregression.** Token `t` and layer `ℓ` use only information available before their selection/attention operation. No second question pass, future-dependent historical writes, or changes to earlier token states. Validate causal masking and incremental KV-cache execution separately from numerical floating-point equality.
2. **Bounded selection.** For fixed model dimensions and fixed `J`, `L`, and `R`, incremental historical selection and attention work must not grow with total context `N`. Intended per-token work is `O(J + L + R)` (plus fixed-size latent computation); historical KV storage is still `O(N)`.
3. **Independent selection pathway.** S's parameters are distinct from Q/K/V. Q/K/V read **real historical KV entries** after selection. Selector annotations, full-context teacher state, directory values, and answer labels must not bypass the selected attention path into decoder logits.
4. **Causal state evolution.** Cross-token state is updated from current/past representations only; cross-layer refinement follows the model's forward order. Recurrent selector state remains bounded independently of context length.
5. **Identifiability.** No model can infer an arbitrary randomized semantic-to-position mapping from a causal query state that does not contain such information. A learned spatial prior can succeed on *predictable structure*; exact arbitrary bindings require causally preserved organization or an allowed bounded index. Test these separately.
6. **Comparable research controls.** Hold task, frozen reader, token budget, causal schedule, answer scoring, source revision, and training/evaluation split constant wherever comparisons require them. Numerical backend changes can affect measured utility and must be reported.

## Research harness

QKVS uses one configuration-driven runtime. Selection policies emit bounded spatial proposals;
the reader resolves them directly to historical windows, gathers the corresponding entries from
its real K/V cache, and records every token occurrence actually read. Rolling-only and alternative
policies use the same causal execution and scoring path.

The checked-in executable configuration is a deterministic CPU verification using a small frozen
synthetic transformer. The same reader interface also supports frozen Hugging Face
`Qwen2ForCausalLM` full-attention checkpoints, including grouped-query attention and the
checkpoint's positional and normalization modules. A pretrained run additionally requires a
pinned model and tokenizer revision, an explicit execution dtype, a SHA-256-pinned UTF-8 source,
and fixed token boundaries. Each checked-in approved long-context configuration authorizes one
specific exploratory, human-initiated GPU run; authorization does not extend to parameter changes
or additional runs.

The Qwen2 adapter writes each layer's position-rotated K/V once, then gathers only the actual
historical occurrences and mandatory rolling window used by that layer's ordinary Q/K/V
attention. Its full-history mode is an explicitly unbounded, short-sequence diagnostic for
comparison with standard Hugging Face execution; it is not a deployable selection policy.
Architectures other than Qwen2, Qwen2 checkpoints with sliding-attention layers, and Qwen2
checkpoints with nondefault RoPE scaling are currently unsupported.

### Dense bootstrap and QKVS activation

Execution has three explicit phases, controlled by working-memory width `W` and activation
threshold `T` (`W=4096`, `T=6144` in the initial long-context configuration):

1. Positions `0..W-1` use ordinary full causal Q/K/V attention. S is neither invoked nor advanced.
2. Positions `W..T-1` remain full causal Q/K/V attention. This dense bridge creates history that
   becomes eligible after the working window starts sliding; eligibility is not a retrieved read.
3. At position `T`, S starts from a zero recurrent state. The newest `W` cache positions, including
   the current position's newly written K/V, are mandatory working memory. Only older positions are
   eligible for at most `R` S-directed historical occurrences. `J`, spatial resolution, and `R`
   remain independent.

The zero state participates in the same recurrence as every later state: the first update is
`z_T = gamma * 0 + (1 - gamma) * u_T`. It is not a special case that bypasses gamma.

Training and evaluation use this same boundary and absolute position accounting. The dense prefix
is independent of S, so Qwen2 can persist an immutable prefill snapshot containing its per-layer
rotated K/V, dense-score metadata, final boundary logits, and next absolute position. Snapshots are
keyed and validated by exact prefix tokens, frozen model identity and weight digest, RoPE/position
configuration, numerical settings, `W`, `T`, and train/evaluation ownership. Identical trajectories
within one owner can share a snapshot; restoration copies cache containers before appending, and no
post-transition K/V or S state is persisted or shared between trajectories.

Dense prefill uses a fixed-capacity Hugging Face cache in bounded query chunks, then stores one
contiguous K/V tensor per layer. Post-transition Qwen2 caches are preallocated and selected with
direct tensor gathers, avoiding per-token full-cache stacking and repeated cache concatenation.
Continuation losses and predictions transfer to the host in one batch rather than synchronizing
once per scalar token result. Training metrics and proposal parameters for all supervised targets
are packed into one host transfer per rollout, and each bounded index tensor is shared by its K/V
gathers. The dense soft-history training branch reads the contiguous causal cache directly instead
of rebuilding and selecting a full-cache index.
Access artifacts encode exact rolling ranges and selected windows with compact varints rather than
expanding thousands of contiguous token indices into JSON. Proposal parameters used for temporal
heat-field plots are sampled at a configured interval (64 tokens in the long study); access counts,
overlap, and exact executed windows remain recorded at every continuation token and layer.

The first learned-S implementation uses an independent per-layer MLP over the causal hidden state
and the bounded current lower-layer proposal state. It emits center, spread, importance, and gate
for each of `J` proposals. Recurrent state stores absolute token centers so fixed gamma persistence
does not drift as the eligible context grows. Effective hard window width is the configured spatial
resolution multiplied by learned spread and divided by fixed beta; every resulting occurrence,
including overlap, consumes `R`. Inactive gates produce no hard read.

Training keeps Q/K/V frozen. A dense offline teacher supplies historical attention coverage, using
the differentiable collective occupancy `1 - product(1 - q)` with alpha zero so redundant soft
overlap earns no additional coverage. The original `soft_occupancy` utility estimator remains
available. The alternative `hard_multiplicity_ste` estimator executes the ordinary resolver in its
forward pass and retains the explicit occurrence-ordered K/V gather, including duplicate BF16
reads. A parallel differentiable branch uses Gaussian proposal multiplicities whose continuous
center, spread, importance, and gate contributions are capped to the configured `R`. The complete
hard target-layer output is combined with the exactly zero forward correction
`soft_hidden - soft_hidden.detach()`, so downstream logits use the real hard result while gradients
follow the soft branch. Complete logits and loss are checked against a separate explicit hard
execution at every target with absolute tolerance `1e-5` for FP32 and `5e-3` for FP16/BF16.

This straight-through gradient is deliberately biased: it smooths integer window rounding, the
gate threshold, importance sorting, and whole-window budget skips. It therefore does not estimate
the true derivative of the discrete resolver and may be misleading near those boundaries. Hard
causal rollouts, absolute-mass teacher coverage, frozen Q/K/V, recurrence, and target-local
truncation are unchanged. Fixed gamma and beta remain the deliberately narrow first
implementation; learned or dynamic gamma/beta are unsupported rather than silently ignored.

The eligible historical slice retains its **absolute mass from the teacher's full causal
attention distribution**. It is not renormalized after working memory is removed. Coverage loss is
the absolute historical mass left uncovered, `sum_i a_i (1 - p_i)`: a query that assigns negligible
attention outside working memory therefore creates negligible coverage pressure. The existing
proposal gates remain free to deactivate historical reads when the downstream utility signal finds
them unhelpful; training does not impose full `R` utilization.

After dense activation, training advances each stream once through coherent causal rollout
segments. K/V and recurrent S continue across target positions and optimizer updates; selector
weights stay fixed within a segment and change only at its boundary. Target positions are
stratified across each segment and rotate through layers. Prior hard decisions and recurrent states
remain detached, so gradients apply only to selected target token/layers rather than providing full
temporal credit assignment. Atomic checkpoints include the exact post-transition training K/V, S
state, position, optimizer, and policy, and may resume only the same identified trajectory.

Training may set `predictive_continuation` to `qkvs` so a supervised target is completed with the
same per-layer S proposals and bounded hard K/V reads as deployed QKVS. The selected and rolling
counterfactual branches fork the current K/V and recurrent selector state; only the target layer's
historical read differs, while downstream layers independently execute their normal causal S
decisions with cross-layer refinement. Those non-target selector decisions stay detached, but the
frozen downstream transformer remains differentiable back to the target layer's STE output. The
reported utility is therefore the **target-layer marginal effect on the complete current-token
QKVS loss**, conditional on the already constructed causal trajectory. It is not a full-trajectory
intervention and provides no gradient through prior tokens or non-target selector choices.

`full_qkv` is an explicit reference continuation: it restores the same dense prefill at `T`, then
reads the complete causal cache on the same ground-truth stream. It is deliberately unbounded and
is never reported as an `R`-matched control. Its next-token loss is a reference measurement, not a
logit target or presumed quality ceiling.

```text
configuration
    -> shared harness lifecycle
        -> selection policy -> bounded proposal resolution
        -> frozen reader -> actual rolling + historical K/V gathers
        -> shared per-token and retrieval metrics
    -> atomic run artifact
```

The public command surface is:

```bash
qkvs --help
qkvs validate --config configs/harness_smoke.json
qkvs run --config configs/harness_smoke.json --device cpu
qkvs status --run artifacts/results/harness-smoke
qkvs summarize --run artifacts/results/harness-smoke
```

`uv` is the bootstrap package manager: it creates an isolated user-level environment and installs
the `qkvs` command with its Python dependencies. It is not part of experiment execution. QKVS does
not use a repository `.venv` or modify the operating system's Python packages. The pinned stack
supports Python 3.14, which `uv` uses to bootstrap the installed command:

```bash
./bootstrap.sh
```

Bootstrap installs the current checkout as an editable tool with its runtime dependencies. Run it
again only when the package metadata or dependencies change. It does not validate or execute an
experiment.

Execution is independent of `uv`. `run.sh` accepts a configuration, validates it, selects its sole
declared `execution.allowed_devices` entry, checks driver health for CUDA, and invokes the installed
CLI:

```bash
./run.sh configs/harness_smoke.json
./run.sh configs/heatfield_learning_study.json
```

Configurations intended for `run.sh` must authorize exactly one device. Use the public `qkvs run`
command directly when choosing among multiple authorized devices.

`uv` installs commands in its user tool bin directory. `run.sh` uses `qkvs` from `PATH` or its
default user-tool location. To use `qkvs` directly from any shell, run `uv tool update-shell` once
and start a new shell if `uv tool install` reports that its bin directory is not already on `PATH`.

The human-started interface for the long-context held-out learning study is:

```bash
./run.sh configs/heatfield_learning_study.json
```

The study configuration is `approved` for one human-initiated exploratory learning/feasibility
pilot. It is not a conclusive convergence or generalization test. It uses independent 16,384-token
training and held-out spans from a SHA-256-pinned Project Gutenberg source, with `W=4096`, `T=6144`,
`J=64`, spatial resolution 8, and `R=512`. A missing local source is fetched atomically from its
pinned HTTPS origin and rejected unless byte count and digest match. Conditions are rolling-only,
fixed offsets, identically initialized untrained S, trained S, and full QKV. Stop on nonfinite
losses, identity mismatches, or resource exhaustion; preserve resumable artifacts rather than
changing the frozen protocol. `heatfield_initial.json` remains an incomplete earlier design draft.
Codex must not run CUDA workloads. Configuration validation, status, and summarization do not
import the model runtime or initialize CUDA.

The completed pilot remains identified by source revision `4285d5f` and its unchanged approved
configuration. Post-hoc descriptive inspection found only five directly supervised targets per
layer. Across early/middle/late training thirds, mean teacher historical mass was 0.467/0.416/0.460,
while 93.1%/94.6%/94.2% of target proposals passed the gate and hard retrieval used
93.0%/94.9%/94.9% of `R`. Held-out trained-S utilization stayed near 96% in every third; active
centers remained concentrated near normalized position 0.5 with standard deviation about 0.082.
This combination is consistent with the normalized teacher objective rewarding historical
coverage even when absolute historical value varied; it is diagnostic evidence, not a causal or
generalization claim.

The fixed-offset control has a separate boundary artifact. At activation position 6144, the
eligible history ends at 2049. Of its 64 offsets, 48 are at least 6144 and clamp to token zero, so
48 proposals repeatedly read the same initial eight-token window. Those reads consume 384 of the
512 historical occurrences while covering only eight unique tokens (376 duplicate
occurrences). This helps explain the control's strong early and poor late position dependence and
must not be interpreted as 64 distinct regions.

`configs/heatfield_learning_followup.json` is `approved` for one human-initiated exploratory
comparison using the
same frozen model, data boundaries, W/T, J/R, and evaluation stream. It changes only the corrected
objective's source identity and increases `targets_per_rollout` from 3 to 24. Forty causal rollouts
therefore provide 960 target-local objectives, exactly 40 per transformer layer instead of five,
distributed across the full post-transition trajectory. Target-local dense teacher and utility
work is expected to increase approximately eightfold; rollout construction, evaluation work, and
checkpoint state size are otherwise unchanged. Reusing the held-out stream preserves pilot
comparability but cannot establish generalization or separately attribute any change to coverage
calibration versus supervision density. Stop on nonfinite values, identity mismatch, or resource
exhaustion; resume only unchanged artifacts and never alter the frozen protocol to finish a failed
run.

`configs/heatfield_hard_aligned_study.json` identifies the approved but failed first hard-aligned
attempt. At source revision `f54c84b`, its BF16 parity guard stopped before training with loss error
`0.0390625`; its partial artifacts and configuration remain immutable and must not be resumed under
changed source.

`configs/heatfield_hard_aligned_corrected_study.json` is approved for one human-initiated
exploratory GPU run. It preserves the failed attempt's model and data identities, seed, W/T/J/R, 40
streaming updates, 960 target-layer objectives, and all five evaluation conditions. Its distinct
run ID prevents the failed partial run from being resumed. The reused held-out stream makes this a
controlled development comparison, not an independent generalization test. Compatible predictive
baselines may be reused by identity, while performance measurements remain fresh. Stop on
nonfinite values, forward discrepancy, identity mismatch, or resource exhaustion. The command is:

```bash
./run.sh configs/heatfield_hard_aligned_corrected_study.json
```

`configs/heatfield_full_qkvs_learning_study.json` is a separate `draft_not_authorized` study. It
preserves the corrected pilot's frozen model, data, W/T/J/R, gamma/beta, 40 streaming updates, 960
target-layer objectives, and evaluation controls. Its only training change is
`predictive_continuation: qkvs`, and its distinct run ID prevents reuse of the completed selector
checkpoint. It must not be executed without independent authorization.

### H1a: unconstrained snapshot compressibility

H1a is a separate first-principles measurement using the shared reader and harness. It has no
rolling window, activation threshold, proposal count, historical budget, gamma, beta, learned
selector, optimizer, or persistent sparse trajectory. For each configured token position `t`, the
reader executes dense full-QKV over `[0,t)` once and persists an identity-checked immutable prefix.
The full reference and every sparse candidate independently execute token `t` through every layer;
each layer recomputes current-token K/V from that branch's hidden state and reads its own sampled
positions from `[0,t]`. The branch is then discarded.

`uniform-aligned-blocks-v1` randomly shifts width-eight block boundaries and samples blocks
uniformly without replacement. Layers receive independently derived seeds, selections do not
overlap within a layer, and every layer receives at least one position. Neither recency nor the
current token is mandatory. A 100% candidate deterministically selects every causal position and
must match the full-QKV logits and loss within the declared numerical tolerance.

Each completed position/fraction trial set is written atomically outside the manifest. Candidate
records contain KL divergence `KL(p_full || p_sparse)`, probability-distribution overlap
`sum_v min(p_full[v], p_sparse[v])`, next-token loss delta, top-one agreement, realized occurrence
fraction, read/block counts, per-layer reconstruction seeds, and compact location histograms. The
overlap is computed from the same float32 distributions as KL, and summaries report descriptive
success fractions at `0.9999`, `0.9995`, `0.995`, and `0.95`; those thresholds are not run gates.
Full block ranges are retained for the best candidate. Configured
`trial_start` and count ranges permit later run identities to add nonoverlapping observations
without replacing prior trial-set artifacts. A typical record is:

```json
{
  "position": 4095,
  "causal_history_length": 4096,
  "requested_sampled_fraction": 0.1,
  "realized_sampled_fraction": 0.1001,
  "kl_divergence_nats": 0.012,
  "probability_distribution_overlap": 0.997,
  "next_token_ce_delta_nats": -0.004,
  "top_one_agreement": true,
  "candidate_seed": 123,
  "trial_id": 17,
  "execution_identity_sha256": "..."
}
```

Frontiers report the smallest fraction found by the configured random search at each KL tolerance.
Failure to observe a qualifying sample is recorded as `not_found`, never as theoretical
impossibility. `configs/h1a_snapshot_compressibility.json` contains the seven positions, ten
sampling percentages, 200 trials, and four descriptive KL thresholds. It is `approved` for one
human-initiated exploratory GPU study. The command is:

```bash
./run.sh configs/h1a_snapshot_compressibility.json
```

Completed snapshot studies can be exported without loading a model:

```bash
qkvs export-snapshot \
  --run artifacts/results/h1a-unconstrained-snapshot-compressibility \
  --output docs/data/h1a-unconstrained-snapshot-compressibility
```

Use `--compress-trials` when the complete candidate CSV would exceed the hosting service's normal
single-file limit. This writes deterministic `H1a_trials.csv.gz` while retaining both compressed
and uncompressed hashes and sizes in the manifest; reference and feasibility tables remain plain
CSV.

The completed context-expansion run is published under
`docs/data/h1a-context-density-expansion/`. Its 264,060-row candidate table is stored as
`H1a_trials.csv.gz`; the dense-reference and feasibility tables remain directly readable CSVs.
The run is pinned to source revision `8b9b389`, and its manifest retains hashes for all 720 original
trial sets and 60 dense-prefix snapshots.

`H1a_KL_0_01_feasibility.csv` is a compact derived cache containing only the 720 measured
position/coverage cells for KL≤0.01. It retains attempts, successful candidates, and hit
probability; the complete feasibility and candidate files remain unchanged and authoritative.

Two follow-up configurations are each approved for one human-initiated exploratory GPU run. Each
reports only the KL≤0.01 analytical threshold.
`configs/h1a_kl_0_01_existing_cell_extension.json` adds trial IDs 400–499 to every
non-dense cell of the completed 60-position grid (66,060 passes including dense controls) and
identity-checks/reuses that run's immutable dense snapshots. `configs/h1a_kl_0_01_position_extension.json`
adds 50 new seeded positions across the 32K range, with 500 candidates at each of the same eleven
non-dense coverage percentages plus a dense control (275,050 passes). Authorization does not extend
to altered parameters or additional runs.

The native `uniform-aligned-blocks-v2` sampler is reserved for fresh studies. It uses NumPy's
pinned PCG64 generator and samples the smaller of the selected or omitted aligned-block sets, then
reconstructs the selected set in causal order. Its seed-to-candidate mapping intentionally differs
from v1 while preserving uniform sampling without replacement. New v2 snapshot execution is
aggregate-only: every completed position/coverage cell appends its KL feasibility row directly to
the requested plain CSV, while `result.json` retains compact identity, resume, and discovered-
frontier metadata. Each CSV row carries its run, source revision, configuration hash, and sampler
identity. The harness does not write per-cell JSON or retain raw candidate trials.

The fresh 500-trial v2 configuration is `draft_not_authorized` with a new run identity.
`configs/h1a_kl_0_01_fresh_v2.json` covers all 110 positions and twelve coverage percentages in
one study. It reuses the 60 compatible immutable dense prefixes and computes the remaining 50.
After independent authorization, one command writes the initially absent CSV:

```bash
./run.sh configs/h1a_kl_0_01_fresh_v2.json \
  docs/data/h1a-context-density-expansion/H1a_KL_0_01_feasibility_v2.csv
```

The same directory includes deterministic SVG and PNG density heatmaps for the `0.001`-nat and
`0.01`-nat KL thresholds. They use actual token coordinates and bilinear interpolation between
measured hit rates; plotted dots identify the observations underlying the smooth field. Regenerate
them from the feasibility CSV with:

```bash
python docs/data/h1a-context-density-expansion/visualize.py
```

PNG rasterization uses `rsvg-convert`; interpolation and SVG construction require no additional
Python packages.

The exporter verifies every trial-set hash, then writes atomic candidate, dense-reference, and
feasibility CSVs plus a provenance manifest. Seeds use exact unsigned 64-bit hexadecimal strings;
candidate rows are ordered by stream, position, requested fraction, and trial ID. The completed
H1a artifacts record 20 sampled-location bins, despite an interim handoff describing 16, so the
export preserves all 20 rather than merging or inventing bins. Dense K/V snapshots remain separate.

`configs/h1a_context_expansion.json` is approved for one human-initiated exploratory GPU study. It
contains 48 seeded positions stratified from position 512 toward the 32K model boundary plus a
separate 12-position replication cohort in the 15.5K–16K region. The 60 frozen positions exclude
all seven original H1a positions; twelve sampling percentages and 400 non-dense trials imply
264,060 candidate passes. `configs/h1a_independent_document_replication.json` remains a separate
unauthorized draft over a SHA-256-pinned Project Gutenberg edition of *Moby-Dick*. Authorization
does not extend to that independent-document study.

## Configuration and artifacts

Configurations keep proposal count, window width, and historical token-occurrence budget separate.
They declare the rolling baseline, matched alternative policies, gamma and beta controls, scoring,
data identity, nonoverlapping train/evaluation token boundaries, device authorization, and run ID.
Unsupported combinations fail validation.

Each run writes `artifacts/results/<run-id>/result.json`. The artifact contains source and config
hashes, raw per-token losses, compact exact K/V access ranges, overlap and coverage measurements,
paired utility, deterministic bootstrap uncertainty where at least two independent streams exist,
progress, and completion state. A one-stream comparison retains its raw paired loss but is labeled
descriptive and has no bootstrap confidence interval. Common-prefix and post-transition quality are
summarized separately, including early/middle/late continuation loss. Wall time covers snapshot
restoration plus continuation; accelerator peak memory is recorded when available. These are
implementation measurements, not theoretical speedup claims. Writes are atomic; rerunning the same
command resumes without duplicating completed condition/stream pairs.

Across run IDs, the harness may reuse completed predictive measurements for rolling-only,
fixed-offset, full-QKV, and identically initialized untrained-S conditions. Compatibility is keyed
by the exact evaluation tokens and support annotations, frozen model and dense-prefill identity,
W/T, condition-specific policy parameters, numerical software environment, and an explicit
predictive-execution version. Untrained S additionally requires the initialized state-dict digest.
The new result stores only a hash-verified reference to the immutable originating artifact, then
evaluates its newly trained S checkpoint normally. It never reuses post-transition K/V or recurrent
S state. Referenced measurements retain their originating revision and hashes.

Predictive and performance reuse are deliberately separate. Reused token scores and K/V accesses
carry no latency or peak-memory measurement in the new run; those values remain attributable only
to their original implementation and hardware execution. A missing, changed, incomplete, or
incompatible source artifact causes normal evaluation or rejects an already-recorded reference.
The completed absolute-mass follow-up seeds rolling, fixed-offset, and full-QKV reuse. Its untrained
S ran before the harness guaranteed selector placement on the requested device, so that condition
must be refreshed once before it can seed later reuse.

Progress covers dense prefills, durable training-checkpoint ranges, and evaluation conditions. In an
interactive terminal, a short ASCII spinner and elapsed timer update once per second on one physical
row; the 24-cell `█▓▒` bar uses third-cell shading, and the line is clipped to the detected terminal
width and erased before every redraw. Redirected output receives one line per work unit instead of
heartbeat writes. During selector training, the active checkpoint's bar advances fractionally after
each optimizer step and reports completed steps, supervised targets, and measured seconds per step.
The durable unit count still advances only after the corresponding checkpoint and result artifact
are written atomically. `qkvs status` reports the same durable phase breakdown for an interrupted or
concurrent run.

`configs/heatfield_learning_study.json` identifies the completed pilot;
`configs/heatfield_learning_followup.json` identifies its one approved absolute-mass follow-up;
`configs/heatfield_hard_aligned_study.json` identifies the preserved failed hard-aligned attempt;
`configs/heatfield_hard_aligned_corrected_study.json` identifies the completed correction;
`configs/heatfield_full_qkvs_learning_study.json` identifies the unauthorized full-QKVS training
draft; `configs/h1a_snapshot_compressibility.json` identifies the completed approved H1a snapshot
study; the context-expansion H1a study is approved for one human-initiated run; and the
independent-document H1a configuration remains unauthorized.
Passing CPU smoke tests, completing either run, or observing a favorable region does not establish
that learned S converges, generalizes, or improves hard next-token prediction.
