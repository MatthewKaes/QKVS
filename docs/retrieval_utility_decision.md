# QKVS retrieval-utility decision

## Decision status

The v1-v5 pointer-recall progression is stopped. Those experiments remain useful engineering and
negative evidence, but they do not establish that exact address reconstruction is the right
objective for learned sparse attention. No v6 router is planned.

The only active experiment is the Stage A gate below. It asks whether one selected old block can
improve a frozen pretrained model's prediction at all. Training `S` is not justified until that
opportunity is demonstrated. The CUDA measurement is pending; the protocol, data construction,
model checkpoint, masks, thresholds, and raw-result schema are frozen before observing results.

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

## Stage A: smallest falsifying experiment

### Hypothesis

At a fixed one-block historical budget, a causally relevant natural-language block reduces frozen
model answer loss relative to rolling-only, random-block, and recency-block attention.

### Frozen model and data

The model is `Qwen/Qwen2.5-0.5B-Instruct` at immutable revision
`7ae557604adf67be50417f59c2c2f167def9a775`. It has 0.49B parameters and a 32,768-token native
context according to its [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and fits
comfortably on the 12 GB device. No model parameter is trained.

The experiment deterministically constructs 24 four-choice questions from the official SQuAD 1.1
development set. SQuAD questions were written against Wikipedia passages and their answers are
passage spans ([Rajpurkar et al.](https://arxiv.org/abs/1606.05250)). Each example contains twelve
128-token natural-document blocks. Three wrong answer options also occur in distractor blocks,
creating competing evidence. Correct labels and target block positions are balanced. The target is
always outside the 256-token rolling window, and no example is selected or rejected based on model
performance.

Every token is computed with rolling-local attention for sparse conditions. Only query tokens gain
access to the selected old block, at its original RoPE position. This retains the pretrained
model's local memory formation while controlling historical retrieval. Sequences contain
1,618-1,638 tokens.

### Compared conditions

- `full`: ordinary full causal attention.
- `rolling`: the last 256 tokens only.
- `random`: rolling plus one uniformly sampled eligible old block.
- `recency`: rolling plus the most recent fully historical block.
- `annotated_oracle`: rolling plus the SQuAD supporting block.
- `utility_oracle`: the lowest-loss one-block intervention after an offline scan of every eligible
  block.

The primary loss is four-label cross entropy in nats. Accuracy, per-example marginal utility,
bootstrap intervals, utility-oracle/support agreement, timings, and peak CUDA allocation are also
recorded. Results are written atomically after every example and the command resumes from the same
artifact.

### Predeclared gate

Stage A passes only if all checks hold on all 24 examples:

1. full-context accuracy is at least 50%;
2. the lower bound of the paired 95% bootstrap interval for full-minus-rolling utility is at least
   0.05 nat;
3. the same lower bound for annotated-oracle-minus-rolling utility is at least 0.05 nat;
4. annotated-oracle utility beats both random and recency, with paired lower bounds above zero.

If any check fails, stop before training `S`. A failure means this model/task/protocol does not
provide the retrieval opportunity needed for the next claim; it does not by itself disprove every
possible model or task.

## Execution and result

The CPU mask tests and an actual-checkpoint CPU integration forward pass both pass. The natural
dataset and model revisions are cached and checksum-pinned. The CUDA measurement must be run from
the user's terminal because invoking this host's GPU through Codex has repeatedly destabilized the
driver.

At this checkpoint the result is **pending**. Further QKVS investment is not yet justified. If the
gate passes, Stage B may compare a deployment-compatible minimal selector with random and positional
priors. If it fails, the correct outcome is to stop or change the frozen model/task foundation—not
to add another router architecture.
