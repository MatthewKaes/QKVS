# Stage B preregistration: causal semantic address directory

## Status

Stage A is frozen and remains the evidence that one correctly selected historical block has useful
downstream value. Stage B is a separate positive-control investigation on branch
`research/stage-b-causal-directory`.

This document preregisters the experiment before causal-oracle evaluation or S training. The
directory training path has not run. Learned admission, eviction, compression, recurrent memory
tokens, multiple retrieved windows, and kernel optimization are explicitly out of scope.

## Question and hypothesis

Stage B asks whether a small causal representation of memory organization makes useful historical
locations inferable without scanning the historical KV cache.

The positive-control representation is a fixed eight-entry semantic address directory:

\[
D_t = \{(k_i, a_i)\}_{i=1}^{8}.
\]

Each `k_i` is written when candidate block `i` closes. Each `a_i` is that block's historical start.
The directory contains no value vector, answer label, future-query feature, or full-context state.
It is visible only to S. The reader receives only the selected address through its historical
attention mask.

The Stage B hypothesis is that frozen causal representations contain enough semantic information
for small S-specific projections to match a final local query to the appropriate directory entry,
and that using the entry's stored address recovers downstream utility.

This is not a claim that eight slots are minimal. It is the smallest explicit positive control that
separates missing memory organization from coordinate prediction.

## Strict single-pass execution

The frozen Qwen2.5 reader has 24 transformer layers. Stage B uses zero-based layer 1 as its only
retrieval layer.

1. Every historical token is processed once with the 256-token rolling mask and fixed global task
   prefix.
2. At a candidate block's closing token, the directory writer may use only layer-1 input states
   already computed for that block. The entry is committed after the block closes.
3. At the final prompt token, layer 0 first performs ordinary rolling attention over the complete
   local question and answer choices.
4. S reads the incoming state of the final prompt token at layer 1 and the already committed
   directory. This happens before layer-1 attention.
5. The selected entry contributes only its stored address. Only the final prompt row of layer 1
   gains the corresponding historical block. Every earlier position and every other layer remains
   rolling-only.
6. Layers 2-23 transform the retrieved residual without another historical selection.

No question preprocessing, retrospective query-token update, second reader pass, future token, or
teacher state exists in the deployable path. A decision can never alter an earlier hidden state.

The batched reference accepts one mask per layer and has a token-by-token KV-cache implementation.
Unit tests verify causal mask placement, future-token invariance, and incremental equivalence for
rolling, causal-oracle, and Stage-A-oracle masks.

An actual-checkpoint CPU audit found the first provisional `1e-5` hidden-state bound too strict:
shape-dependent SDPA reductions produced maximum hidden error `0.001633` while logit error remained
below `2.48e-5` and top-1 predictions were identical. That failed audit is preserved. The declared
mechanical equivalence bounds are therefore:

- hidden maximum absolute error at most `0.002`;
- logit maximum absolute error at most `0.0001`;
- final-hidden cosine at least `0.999999`;
- identical top-1 token.

The actual checkpoint passes these bounds for all three mask modes. Before the oracle dataset runs,
one deterministic full-length example must also pass an incremental CUDA replay under the same
semantic conditions. Failure blocks all subsequent work.

## Historical isolation and data

The model, tokenizer, 32-token global prefix, 56-block history, 128-token block width, 256-token
rolling window, and one-block retrieval budget are unchanged from Stage A V3.

Exactly eight marked semantic documents occupy candidate positions 0-7. Their content is randomly
permuted among those positions. The remaining 48 blocks are natural distractors. Candidate blocks
remain at least 6,197 tokens before the prediction, beyond the 6,120-token maximum 24-layer rolling
relay distance. Semantic documents are not distributed among the later 48 blocks.

The future question selects one of the eight candidates, but the writer never sees the question.
Target position and answer label use independent balanced schedules. Wrong options are supported by
other candidate documents, as in Stage A.

The official SQuAD splits are checksum-pinned:

- training: `train-v1.1.json`, SHA-256
  `3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955`;
- evaluation: `dev-v1.1.json`, SHA-256
  `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`.

The 442 training and 48 evaluation article titles have zero overlap. Training articles are
deterministically split 80/20 by seed `20265001`; validation uses only held-out training articles.
The frozen sizes are 512 training, 128 validation, and 128 evaluation examples. No representation,
normalization statistic, checkpoint choice, or hyperparameter may use evaluation examples.

The exact 24 Stage A V3 question IDs are frozen in
[`stage_b_excluded_question_ids.json`](../configs/stage_b_excluded_question_ids.json), together with
the source artifact hash. They are removed from the SQuAD development pool before Stage B held-out
examples are sampled. Gate 0 deliberately reuses those questions only to measure causal-oracle
parity with Stage A; they cannot appear in selector training, validation, model selection, or the
Stage B held-out learned-selector evaluation.

## Gate 0: causally aligned oracle parity

This gate runs before S training on the same 24 examples used by Stage A V3:

- `rolling`: no historical block;
- `causal_oracle`: the annotated block is visible only to the final prompt token at layer 1;
- `stage_a_oracle`: the annotated block is visible to every query token at every layer, retained only
  as the original diagnostic.

The causal oracle passes only if:

1. answer accuracy is at least 50%; and
2. the paired 95% bootstrap lower bound for causal-oracle-minus-rolling utility is at least 0.05
   nat.

Recovered oracle utility is reported as the ratio of aggregate mean improvements:

\[
\frac{\bar L_{rolling} - \bar L_{causal\ oracle}}
     {\bar L_{rolling} - \bar L_{Stage\ A\ oracle}}.
\]

Per-example ratios are never averaged. If Gate 0 fails, stop before constructing or training S.

## Directory representation and training

The directory uses layer-1 input states, so write and query representations occupy the same frozen
representation level.

The primary block representation is the mean of the 128 causally computed block states available at
block close. The bounded control uses only the closing-token state. No new semantic encoder is
introduced.

For directory entry `i` and final query state `h_q`:

\[
k_i = \operatorname{normalize}(W_w r_i), \qquad
q = \operatorname{normalize}(W_s h_q), \qquad
z_i = q^T k_i / 0.07.
\]

`W_w` and `W_s` are independent 896-to-32 linear projections owned only by S. The loss is eight-way
cross entropy against the annotated supporting entry. The selected entry supplies its stored
address directly; no coordinate regression, hierarchical decoder, soft historical selection, or
answer-loss gradient through the hard selection is used.

The reader remains frozen. Training uses AdamW, batch 32, learning rate `1e-3`, weight decay `0.01`,
gradient clipping at 1.0, and at most 1,000 steps. Validation runs every 100 steps and chooses the
lowest validation cross entropy. Seeds are `20267001`, `20267002`, and `20267003`. Raw gradients,
clipping ratios, parameter-update magnitudes, selection entropy, and address distributions are
recorded despite the small selector.

Frozen causal features may be cached for optimization, but cache creation must use the single-pass
causal reference. Feature caching is not a second deployable reader pass.

## Required controls

| Control | Purpose |
|---|---|
| Query-only S with eight learned position keys | Tests whether query or position priors suffice |
| Correct directory plus learned S | Primary deployable condition |
| Shuffled key/address bindings | Tests dependence on semantic-address association |
| Random candidate and candidate-recency | Fixed-budget sparse baselines |
| Causally aligned annotated oracle | Establishes single-pass reader headroom |
| Stage A oracle | Quantifies utility lost to stricter timing |
| Full-context query representation | Explicitly nondeployable teacher diagnostic |
| Moved-block counterfactual | Tests whether S follows semantics to a new address |
| Retrieved-block removal | Tests whether answer information bypasses historical attention |
| Closing token versus causal mean pool | Isolates causal representation quality |

Moved-block pairs preserve all document and query content while deranging candidate positions. The
directory is rewritten causally with the new addresses. Evaluation records address selection on
both members and whether the selected address follows the moved supporting document.

For removal, S and the directory still execute, but the selected historical mask is suppressed. The
answer logits must match rolling-only within `1e-5`. Directory keys never enter Q/K/V, residual
states, logits, or the answer head.

## Metrics and final gate

All downstream comparisons use paired example-level answer losses with 2,000 deterministic
bootstrap samples. Address accuracy and entropy are diagnostic; answer loss is primary.

Recovered learned utility is the ratio of aggregate improvements:

\[
\frac{\bar L_{rolling} - \bar L_S}
     {\bar L_{rolling} - \bar L_{causal\ oracle}}.
\]

It is reported without a pass threshold. Stage B passes only if all of the following hold on the
article-disjoint evaluation split:

1. Gate 0 passed before S training.
2. The paired bootstrap lower bounds for learned-S improvement over rolling, random, and
   candidate-recency are each above zero.
3. Moved-block address accuracy is at least 50%, its bootstrap lower bound exceeds 12.5% chance, and
   the predicted address follows the move on at least 50% of pairs.
4. Retrieved-block removal matches rolling logits within `1e-5` and eliminates learned retrieval
   utility.
5. Full-length incremental execution passes the declared causal-equivalence bounds.

Results are reported independently for all three training seeds. The primary gate uses the pooled
causal representation. Closing-token and nondeployable-teacher results cannot rescue a failed
primary condition.

The primary three-seed aggregate is formed before any gate calculation. For every held-out example
and condition, answer loss, correctness, address correctness, and counterfactual indicators are
first averaged across the three fixed seeds. Paired condition differences are then formed from
those per-example seed means. The 2,000-sample paired bootstrap resamples examples, carrying each
example's complete three-seed tuple together. Seeds are not treated as independent examples, no
seed is selected after evaluation, and all individual-seed measurements remain in the raw result.
Recovered utility uses aggregate mean losses computed from these per-example seed means; it never
averages seed-level or example-level ratios.

If the causal oracle passes but learned S fails, diagnosis proceeds in order: closing versus pooled
causal representation quality, semantic selection loss/accuracy, then optimizer telemetry. Model
capacity, directory size, admission, and architecture must not change within this experiment.

## Related boundary

Recurrent Memory Transformer and Block-Recurrent Transformer demonstrate that fixed recurrent state
can carry information across segments
([Bulatov et al.](https://arxiv.org/abs/2207.06881),
[Hutchins et al.](https://arxiv.org/abs/2203.07852)). Landmark Attention represents blocks for
retrieval but searches a landmark collection that grows with history
([Mohtashami and Jaggi](https://arxiv.org/abs/2305.16300)). Stage B instead fixes the directory at
eight semantic-address bindings and tests only whether that explicit causal organization is
sufficient for S.
