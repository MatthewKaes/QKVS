# QKVS retrieval-utility decision

## Decision status

The v1-v5 pointer-recall progression is stopped. Those experiments remain useful engineering and
negative evidence, but they do not establish that exact address reconstruction is the right
objective for learned sparse attention. No v6 router is planned.

The only active investigation is the Stage A gate below. It asks whether one selected old block can
improve a frozen pretrained model's prediction at all. Training `S` is not justified until that
opportunity is demonstrated. V1 is invalidated, V2 is complete but nondecisive, and V3 has
conditional approval with its final gate amendment frozen before execution. Every protocol and raw
result remains separate.

## What the previous experiments established

- Direct block decoding, causal bounded selection, overlapping windows, immutable KV caches, and
  fixed retrieval work are implemented and mechanically tested.
- Controlled oracle experiments show that information can be written into an intermediate token's
  cached K/V and read later.
- A standalone hierarchical decoder can copy explicitly supplied addresses through 64K. This is a
  positive result for address representation under direct supervision, not learned memory utility.
- Hard selection gives answer loss no gradient to address outputs. The local two-neighbor relaxation
  restores only a local gradient and cannot teach a distant jump by itself.
- Integrated V4 routing remained weak and failed to extrapolate. In V5, even exact supplied-pointer
  windows left answer accuracy at chance, despite all 512 audited windows containing their intended
  records. That gate failed downstream of address decoding.
- The chunked implementation substantially improved prototype training throughput, but the eager
  sparse decode path did not beat optimized dense attention.

These findings support the implementation mechanisms and several failure diagnoses. They do not
show that a deployable `S` can discover useful semantic history or outperform fixed sparse
attention.

## Assumptions discarded

1. Exact reconstruction of arbitrary numeric pointers is not a surrogate for useful retrieval.
2. Falling coordinate loss, nonzero gradients, and query-sensitive routes are not evidence that a
   retrieved block improves a prediction.
3. An oracle address is not enough: the frozen or trained reader must be able to use the exposed
   information.
4. Teacher access to historical locations cannot be treated as deployable router information.
5. Random placement is not merely an optimization challenge. If the router input carries no
   information about the sampled location, direct address prediction is unidentifiable.
6. More windows, address digits, temperature schedules, or kernel work are not justified before a
   rolling-versus-oracle utility gap exists.

## Mechanism reassessment

QKVS does not sum historical key/value outer products into one memory matrix. Capacity and
interference results for linearized attention or fast-weight memories therefore do not transfer
directly: those mechanisms explicitly accumulate additive outer-product updates
([Schlag et al.](https://arxiv.org/abs/2102.11174),
[Katharopoulos et al.](https://arxiv.org/abs/2006.16236)). QKVS retains individual K/V entries.

Its relevant interference occurs after selection. Every additional occurrence joins the same
softmax competition, so a block is useful only when its evidence improves the answer more than its
irrelevant tokens dilute or redirect attention. Duplicate windows also change softmax multiplicity.
This is why block presence, attention mass, and downstream utility must be measured separately.

Existing content-routed methods relax a different constraint. Routing Transformer clusters token
representations; Landmark Attention scores block landmarks; SeerAttention pools both Q and K to
predict block sparsity; and Memorizing Transformers performs approximate KNN lookup
([Routing Transformer](https://arxiv.org/abs/2003.05997),
[Landmark Attention](https://arxiv.org/abs/2305.16300),
[SeerAttention](https://arxiv.org/abs/2410.13276),
[Memorizing Transformers](https://arxiv.org/abs/2203.08913)). These results show that selected
history can help, but their selectors inspect historical content or summaries. They do not answer
whether a constant-work direct-address head can infer a useful location from its current state.

Simple needle retrieval is also insufficient evidence for the intended claim. RULER reports that
models which pass vanilla needle tests can degrade sharply on multi-needle, multi-hop, and
aggregation tasks ([Hsieh et al.](https://arxiv.org/abs/2404.06654)). Stage A therefore begins with
natural reading-comprehension dependencies and competing evidence rather than a new pointer code.

## Retrieval utility

For target output `y`, rolling memory `M`, already selected historical blocks `A`, and candidate
block `b`, define conditional marginal utility as

\[
U(b \mid A, x, y) =
\ell(y \mid x, M \cup A) - \ell(y \mid x, M \cup A \cup \{b\}).
\]

Positive utility means adding the block lowers held-out answer loss. The definition directly
measures the property the retrieval mechanism is intended to preserve. It does not require a block
to match a unique oracle coordinate.

Utility is conditional on the selected set. Two individually useless blocks may be jointly useful;
two individually useful blocks may be redundant; and a useful block may become harmful beside a
competitor. Consequently, per-block utilities are not assumed additive. Stage A uses a one-block
budget specifically to avoid that interaction in the first falsification. A later two-block test
must measure joint and conditional effects rather than sum singleton scores.

The offline utility oracle is label-dependent and scans candidates. It is an upper-bound teacher,
not an inference algorithm. Dense-attention mass may later be compared with utility, but it is not
used as the target here.

## Identifiability boundary

The deployable router may use the current hidden state, the current position and address-space
scale, and state that earlier causal computation deliberately preserved. It may not read all old
keys, block summaries, gold supporting locations, or teacher hidden states computed with full
history.

This yields a necessary counterfactual for Stage B: hold the local query suffix fixed and move the
supporting evidence among historical blocks. If the deployment-compatible router representation is
unchanged, no deterministic direct-address head can predict both locations. More parameters or a
different coordinate loss cannot repair missing information. A viable system would need an online
location-bearing state, predictable document structure, or some bounded historical index. Which of
those belongs in QKVS is deliberately left open until Stage A passes.

## Stage A protocol lineage

### V1: invalid prompt visibility

The first mask allowed full attention to see the system instruction while sparse conditions lost it
after it left the rolling window. After five examples, full attention was 5/5, rolling and the
annotated block were each 2/5, and the label-dependent utility oracle was 5/5 but selected the
annotated block 0/5 times. These numbers do not measure historical-block utility because task
instructions were not held constant.

A one-variable CPU counterfactual retained the same inputs, rolling window, annotated block, model,
and output labels while making only the fixed 32-token prefix globally visible. Annotated-block
accuracy changed from 2/5 to 5/5 and mean loss from 1.706 to 0.00233. The raw pilot and audit are
preserved in
[retrieval_utility_stage_a_invalid_prompt_visibility.json](../artifacts/results/retrieval_utility_stage_a_invalid_prompt_visibility.json)
and
[retrieval_utility_stage_a_prompt_visibility_audit.json](../artifacts/results/retrieval_utility_stage_a_prompt_visibility_audit.json).

### V2: prompt-corrected but nondecisive

V2 made the fixed prompt globally visible and completed all 24 examples. Its raw results are
preserved in
[retrieval_utility_stage_a_v2.json](../artifacts/results/retrieval_utility_stage_a_v2.json).

| Condition | Accuracy | Mean loss | Mean utility vs rolling |
|---|---:|---:|---:|
| Full | 95.8% | 0.137 | 0.139 |
| Rolling | 87.5% | 0.276 | 0.000 |
| Random | 87.5% | 0.388 | -0.112 |
| Recency | 87.5% | 0.293 | -0.017 |
| Annotated block | 100.0% | 0.00778 | 0.268 |
| Utility oracle | 100.0% | 0.00778 | 0.268 |

The annotated block beat rolling, random, and recency under the V2 mask. The strict V2 gate still
failed because the paired full-minus-rolling bootstrap lower bound was 0.0379 nat, below the
predeclared 0.05 threshold. More importantly, V2 has three design defects that make it unsuitable
as the decisive Stage A result:

1. the random baseline was sampled only from non-target blocks, biasing it downward;
2. target position was `index mod 8` and answer label was `index mod 4`, so labels were determined
   by target position rather than independently randomized;
3. its 1.6K sequence fit inside the frozen model's multilayer rolling receptive field, so the
   rolling baseline could receive target information by relay without direct historical access.

V2 is therefore retained as an informative intermediate result, not support for proceeding to
Stage B.

## V3 preregistration amendment

V3 changes only data geometry, factor randomization, baseline sampling, mask validation, and resume
integrity. It does not implement `S`, change an `S` loss, or add a routing architecture. Full
execution is blocked pending review of this amendment.

### Hypothesis

With task-prompt visibility fixed and supporting evidence beyond the multilayer rolling receptive
field, a causally relevant natural-language block at a one-block historical budget reduces frozen-
model answer loss relative to rolling-only, random-block, and recency-block attention.

### Frozen model and natural task

The model remains `Qwen/Qwen2.5-0.5B-Instruct` at immutable revision
`7ae557604adf67be50417f59c2c2f167def9a775`. It has 24 layers and a 32,768-token native context
according to its [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). No parameter is
trained.

The same checksum-pinned SQuAD 1.1 development set supplies natural Wikipedia passages, questions,
and annotated answer spans ([Rajpurkar et al.](https://arxiv.org/abs/1606.05250)). Each of 24
examples contains 56 fixed 128-token document blocks and three competing answer options that also
occur in distractor blocks. No example is selected using model performance. Total sequence length
is 7,251-7,274 tokens.

### Independent factors and fair baselines

The candidate set is the first eight document blocks. Target positions are balanced at three uses
per block and shuffled with seed `20262018`. Answer labels are balanced at six uses per label and
shuffled separately with seed `20263018`. Dataset order uses seed `20261017`. This removes the V2
deterministic position-label mapping while preserving balanced marginals.

The random baseline samples uniformly from the same eight-block candidate set with seed `20264018`.
It is allowed to select the target; the frozen schedule does so on 2/24 examples. Recency always
selects candidate block seven, and the independently shuffled target schedule places the target
there on 3/24 examples. Random, recency, annotated, and utility-oracle conditions therefore use the
same one-block historical budget and candidate domain.

### Multilayer rolling exclusion

With 24 layers and a 256-token inclusive window, the maximum token-to-token relay distance is

\[
24(256-1)=6120\text{ tokens},
\]

or 6,121 tokens including the current token. Every candidate block ends at least 6,197 tokens
before the final prediction, leaving a minimum 77-token margin beyond that maximum dependency path.
Construction fails rather than emitting an example if this invariant is violated.

Every sparse condition gives all tokens the same fixed 32-token system/user-role prefix. It contains
no question, answer option, or document evidence. All other computation is rolling-local; only query
tokens gain access to the selected historical block at its original RoPE position.

### Compared conditions and final gate amendment

- `full`: ordinary full causal attention.
- `rolling`: the last 256 tokens plus the fixed task prefix.
- `random`: rolling plus the preregistered uniform candidate.
- `recency`: rolling plus candidate block seven.
- `annotated_oracle`: rolling plus the SQuAD support block.
- `utility_oracle`: the lowest-loss block found by an offline label-dependent scan of all eight
  candidates.

The primary loss remains four-label cross entropy in nats. Full attention is retained as a diagnostic
but is not a gate: selective retrieval can legitimately beat full attention by suppressing
interfering blocks. Stage A passes only if all primary checks hold on all 24 examples:

1. annotated-block accuracy is at least 50%, twice the four-choice chance rate and the absolute
   reader-capability sanity check;
2. the lower bound of the paired 95% bootstrap interval for annotated-block-minus-rolling utility is
   at least 0.05 nat;
3. annotated-block utility beats fair random selection, with a paired lower bound above zero;
4. annotated-block utility beats candidate-recency, with a paired lower bound above zero.

Full accuracy and the full-minus-rolling mean and interval remain in the raw result. They cannot
change the gate outcome in either direction.

If any check fails, stop before training `S`.

### Mask and resume integrity

The actual pinned 0.5B checkpoint was run on CPU under both its built-in causal mask and the explicit
dense 4D mask used by this experiment. All final hidden-state and logit differences were exactly
zero in FP32 over 128 tokens. The executable test is
[test_checkpoint_mask.py](../tests/test_checkpoint_mask.py), and the raw audit is
[retrieval_utility_actual_checkpoint_mask_audit.json](../artifacts/results/retrieval_utility_actual_checkpoint_mask_audit.json).

Before V3 examples are measured, the same actual-checkpoint comparison must pass a 128-token BF16
CUDA smoke test. Its preregistered requirements are finite outputs, identical top-1 vocabulary
prediction, last-hidden cosine at least 0.999, hidden maximum absolute error at most 0.125, and logit
maximum absolute error at most 0.5. This smoke tests CUDA mask invariance only; it cannot contribute
examples to the Stage A result.

A new result stores the Git revision and SHA-256 hashes of the experiment module, runtime module,
configuration, `pyproject.toml`, and `uv.lock`. Initial execution requires those tracked files to be
committed and clean. Every resume recomputes and exactly compares the revision, hashes,
configuration, dataset checksum, and device before loading additional measurements.

The amended configuration is `configs/retrieval_utility_stage_a_v3.json`. Its data preparation has
been executed, but no V3 model measurement has run. The final gate amendment is frozen before the
CUDA smoke and full evaluation. Further QKVS investment remains unjustified until the smoke and the
primary sparse-utility gate pass.
