# Learned direct-address sparse attention: initial research report

Date: 2026-10-17

## Executive result

The prototype establishes **engineering success**: direct learned addresses can select arbitrary
old, immutable KV entries with `O(L + R)` attention work per decoded token, fixed independently of
history length. The implementation is causal, preserves duplicate-selection semantics, has
independent `S` projections at every layer, and matches its batched reference during incremental
decoding.

It also establishes a narrow **learning success**. With explicit pointer supervision, a strict
three-hop model in which only the final query can retrieve history achieved 72.7% native answer
accuracy at its trained length (chance 3.125%). Its per-layer route accuracies were 95.3%, 91.4%,
72.7%, and 71.1%, and ablating preceding pointers changed later addresses. This model was trained
with oracle-selected attention routes on every example while `S` learned from an auxiliary loss;
it is a supervised-router baseline, not end-to-end routing discovery from answer loss. The same
network with oracle routes achieved 100%, so answer capacity was not the limiting factor.

The initial address scheme did **not** establish generalization success. Strict native three-hop
accuracy fell to 4.7% at 2x context and 3.9% at 4x, despite 100% oracle accuracy. Four-hop native
routing also failed. Absolute scalar regression learns useful in-distribution routes, but later-hop
errors compound and address precision does not extrapolate reliably.

Semantic retrieval divides into two cases. A semantic key with a stable layout learned a direct
address (100% at train length, 74.2% at 2x, 38.3% at 4x). When record placement was independently
randomized, native QKVS remained near the 6.25% task chance rate while dense content attention
reached 100%, 95.3%, and 65.6%. Direct addressing cannot infer a per-example random address from a
query that contains no placement information; an index-building mechanism or content search would
be required.

The eager PyTorch implementation demonstrates bounded work but no wall-clock speedup. Its
materialized gather and many small address-decoding kernels cost about 0.355 ms per one-layer decode
from 512 through 32K tokens, while optimized dense SDPA rose from 0.014 to only 0.038 ms. A fused
kernel is necessary before performance claims are credible.

## Implementation

Each pre-normalized layer input independently produces `Q`, `K`, `V`, and `S`. `S` emits one
normalized scalar per historical block window. For a query at position `t`:

- through `L + R` tokens, the candidate sequence is exactly the full causal prefix;
- afterward, it contains the last `L` occurrences and at most `R` historical occurrences;
- an address maps directly to a block with a multiply and floor, without scoring or scanning old
  keys;
- repeated blocks remain repeated occurrences in the softmax sequence;
- every layer predicts from the hidden state produced by preceding layers;
- cached K/V tensors are never mutated.

Q/K/V use BF16 under CUDA autocast. `S` remains FP32 because BF16 scalar precision becomes an
addressing limit as block count grows. Hard gathers receive no address gradient, so training uses a
supervised auxiliary route loss and optional teacher-routed warm-up. The improved loss measures
error in block coordinates rather than normalized-coordinate distance.

The full specification and unresolved alternatives are in
[architecture.md](architecture.md). Source is organized around
[attention.py](../qkvs/attention.py), [model.py](../qkvs/model.py), procedural
[data.py](../qkvs/data.py), and the checkpointable [training.py](../qkvs/training.py).

## Environment and reproducibility

- GPU: NVIDIA GeForce RTX 5070, 12,227 MiB reported VRAM, compute capability 12.0 (`sm_120`)
- Driver: 610.57.04
- PyTorch: 2.9.0+cu130; bundled CUDA runtime 13.0
- System CUDA compiler: 13.3.73
- Python: 3.13.15
- OS during recorded runs: Linux 7.2.3-arch1-3, glibc 2.44

Dependencies are pinned in [pyproject.toml](../pyproject.toml) and [uv.lock](../uv.lock). Each run
records its model, training configuration, seed, environment, raw checkpoints, periodic loss
history, and evaluation counts. GPU training is not asserted to be bit-for-bit deterministic.

One run triggered a known sporadic host failure: NVIDIA `Xid 79` (GPU fell off the PCIe bus), then
`Xid 154` requested reboot. Kernel logs did not report CUDA OOM; later reduced runs used only about
60 MiB peak PyTorch allocation above the process baseline. The incident and mitigations are
recorded in [xid79_2026-10-17.json](../artifacts/system/xid79_2026-10-17.json). It is treated as an
environment limitation, not an architecture result.

## Experiment A: correctness and scaling

The test suite covers:

- exact full-prefix selection below the `L + R` threshold and equality to dense attention math;
- direct block decoding, partial blocks, fixed selection width, and causal bounds;
- overlap and repeated occurrences;
- equivalence of repeats to a deduplicated `+log(m)` logit bias;
- auxiliary gradient delivery to `S` and independent `S` parameters per layer;
- batched versus incremental decode equivalence across the sparse threshold;
- a retrieval-mask ablation used to isolate query-level cascading;
- pointer-chain and semantic-dataset invariants.

The final local suite has 19 passing tests. The mathematical decode score count is `N` for full
attention and at most `L + R` for QKVS; KV storage remains `O(N)` for both.

The CUDA reference benchmark used one BF16 decode query, width 128, four heads, `L=256`, `R=256`,
and block width 16. KV bytes below are theoretical K+V storage for four layers. Times are medians of
50 eager iterations:

| Context | Full decode | QKVS combined | Address projection | Address decode incl. projection | Gather + attention | 4-layer KV |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.0140 ms | 0.3553 ms | 0.0148 ms | 0.2060 ms | 0.0815 ms | 1 MiB |
| 2,048 | 0.0144 ms | 0.3562 ms | 0.0149 ms | 0.2020 ms | 0.0815 ms | 4 MiB |
| 8,192 | 0.0202 ms | 0.3549 ms | 0.0150 ms | 0.2003 ms | 0.0815 ms | 16 MiB |
| 32,768 | 0.0384 ms | 0.3566 ms | 0.0149 ms | 0.2037 ms | 0.0806 ms | 64 MiB |

Bounded latency is flat across this range, supporting the complexity claim, but dense Flash SDPA is
still 9.3x faster at 32K. Address decoding is dominated by eager kernel-launch overhead rather than
the `S` projection itself.

Four-layer prefill further exposes the reference implementation's materialized gather:

| Context | Dense prefill | QKVS prefill | Dense temporary peak | QKVS temporary peak |
|---:|---:|---:|---:|---:|
| 128 | 1.652 ms | 3.393 ms | 1.9 MiB | 52.7 MiB |
| 512 | 1.655 ms | 7.817 ms | 3.4 MiB | 207.0 MiB |
| 2,048 | 1.647 ms | 28.746 ms | 9.4 MiB | 824.3 MiB |

These numbers are implementation evidence, not an architectural speed claim. The raw artifact is
[rtx5070_reference.json](../artifacts/benchmarks/rtx5070_reference.json).

## Experiment B: learned explicit addresses

The first one-hop model used four layers, width 96, `L=24`, `R=8`, block width 2, and trained at
length 96 for 300 steps. Native answer accuracy was:

| Length | Answer accuracy | Mean final-query route accuracy |
|---:|---:|---:|
| 96 | 100.0% | 67.3% |
| 192 | 97.7% | 24.5% |
| 384 | 92.2% | 11.3% |

The discrepancy between answer and final-query route accuracy revealed distributed relay paths:
other positions retrieved records and propagated their contents into later rolling representations.
This is a valid architecture behavior, but it prevents this experiment alone from proving a
successive-query cascade. The result is in
[one_hop_learned.json](../artifacts/results/one_hop_learned.json).

## Experiment C: multihop cascading

### Initial study and controls

At `L=24`, `R=8`, length 96, the initial three-hop learned, random, unconditioned, and rolling
models all stayed near the 3.125% answer chance rate after 300 steps. Dense attention reached 100%.
The learned model also stayed at chance under oracle routes, showing that this first failure mixed
routing and answer-capacity/training problems.

A 1-to-2-to-3-hop curriculum with one retrieved window reached 99.7% at length 64, 81.4% at 2x,
and 54.4% at 4x. Tracing showed that it often answered without the expected final-query route;
ablating intermediate pointers reduced accuracy, and removing record tags collapsed it to chance.
The network had learned distributed relays. A query-only retrieval ablation still allowed rolling
propagation because four layers with `L=16` have a broad effective receptive field.

### Strict cascade isolation

The decisive control uses `L=1`, `R=1`, one-token blocks, and disables historical retrieval for
every non-final-query position. Non-query states can attend only to themselves, so the answer must
follow the final query's layer-by-layer addresses.

| Model | Length | Native answer | Oracle answer | Native routes by layer |
|---|---:|---:|---:|---|
| 3-hop, normalized loss/curriculum | 64 | 35.5% | 100.0% | 68.6%, 45.5%, 23.7%, 15.7% |
| 3-hop, FP32/block loss/teacher | 64 | 72.7% | 100.0% | 95.3%, 91.4%, 72.7%, 71.1% |
| same | 128 | 4.7% | 100.0% | 3.9%, 1.6%, 0.8%, 0.8% |
| same | 256 | 3.9% | 100.0% | 2.3%, 0.0%, 0.8%, 1.6% |
| 4-hop, FP32/block loss/curriculum | 64 | 8.6% | 98.4% | 79.7%, 19.5%, 3.1%, 2.3% |

For the successful in-distribution three-hop model, ablating the preceding record pointer changed
the next two address vectors by mean absolute amounts 0.364 and 0.269. This supports genuine
conditioning on retrieved content. It does not generalize to a larger address range.

Raw strict results are in
[three_hop_block_loss_teacher_safe.json](../artifacts/results/three_hop_block_loss_teacher_safe.json)
and
[four_hop_block_loss_curriculum_safe.json](../artifacts/results/four_hop_block_loss_curriculum_safe.json).

## Experiment D: memory persistence

A trained one-hop model was evaluated with historical retrieval explicitly disabled at the final
query. “Frequent” externally supplies address-bearing refresh tokens between the record and final query;
“once” retrieves once and then leaves a long unused gap; “never until query” is a direct-retrieval
positive control; “no access” is the negative control.

| Length | Frequent refresh | One early retrieval | Direct at final query | No access |
|---:|---:|---:|---:|---:|
| 96 | 84.4% | 3.9% | 100.0% | 1.6% |
| 192 | 35.2% | 2.3% | 87.5% | 7.0% |
| 384 | 23.4% | 2.3% | 71.1% | 2.3% |

This measures persistence when the model is repeatedly given the correct memory address; it does
not show autonomous decisions to refresh a memory. Frequent retrieval can keep information
decodable through subsequent dense-window processing, but a single old retrieval did not create
permanent memory. This is consistent with immutable caches and transient hidden-state propagation. See
[one_hop_memory_persistence.json](../artifacts/results/one_hop_memory_persistence.json).

## Experiment E: overlap and multiplicity

Across 100 random references, explicit duplicate occurrences and unique entries with `+log(m)`
logit bias differed by at most `3.58e-7` in FP32.

The learned four-window one-hop router produced at least one duplicate in 90.6% of length-96
examples, 63.3% at length 192, and 33.6% at length 384. Forcing duplicate blocks to the nearest
unused blocks at the same selection budget changed answer accuracy by exactly zero in this sample:
100.0%, 99.2%, and 89.1%, respectively. Duplicates are semantically correct and frequently
emitted, but this model gives no evidence that they improve answers. The forced-unique pass is an
inference ablation, not a separately trained unique-router baseline. See
[learned_overlap_ablation.json](../artifacts/results/learned_overlap_ablation.json).

## Experiment F: semantic retrieval

Records contain categorical key/value pairs and the query contains only a categorical key—no
numeric pointer. Chance answer accuracy is 6.25%.

| Placement / model | Length 64 | Length 128 | Length 256 |
|---|---:|---:|---:|
| Fixed layout, native QKVS | 100.0% | 74.2% | 38.3% |
| Fixed layout, oracle QKVS | 100.0% | 100.0% | 100.0% |
| Random layout, native QKVS | 10.2% | 7.0% | 2.3% |
| Random layout, oracle QKVS | 100.0% | 100.0% | 100.0% |
| Random layout, dense attention | 100.0% | 95.3% | 65.6% |
| Random layout, rolling-only | 6.2% | 7.8% | 14.1% |

The fixed-layout result is learned semantic-to-address mapping, not arbitrary semantic lookup. The
random-layout result identifies a structural limit: a bounded direct router cannot recover an
independent random placement without information supplied by the sequence, a learned index, or a
content-dependent search. Artifacts are
[semantic_fixed_qkvs.json](../artifacts/results/semantic_fixed_qkvs.json),
[semantic_random_qkvs.json](../artifacts/results/semantic_random_qkvs.json), and
[semantic_random_dense.json](../artifacts/results/semantic_random_dense.json).

## Conclusions

### What is supported

- Fixed-budget direct-address causal attention is correct and incrementally executable.
- `S` adds no context-length-dependent search; cache storage remains linear.
- Independently conditioned routers can learn a supervised three-hop cascade in distribution.
- Retrieved information can be refreshed through normal hidden states without mutating old KV.
- Semantic keys can learn stable layout addresses.
- Duplicate occurrences have the expected log-multiplicity semantics.

### What is not supported

- Absolute normalized scalar routing did not extrapolate strict multihop retrieval beyond its
  training context length.
- Four-hop learned routing was not reliable.
- Random-placement semantic lookup is impossible for the direct router without an additional
  placement signal or indexing process.
- The eager PyTorch implementation is slower and more memory-hungry than optimized dense SDPA at
  all measured lengths.
- No practical language-model benefit has been demonstrated.

### Best next experiments

1. Replace scalar regression with hierarchical coarse/fine address digits whose precision is
   independent of total block count.
2. Train the router with quantization-aware classification or a soft local window, then harden it;
   retain block-scaled supervision.
3. Separate prefill index construction from decode-time direct lookup and account for its cost.
4. Fuse address decoding, KV gather, and selected attention in Triton before repeating performance
   comparisons.
5. Repeat strict multihop runs across several seeds and larger held-out sets before claiming robust
   learning success.
6. Evaluate language data only after routing generalization improves; current evidence does not
   justify that expense.

## Reviewer-directed second iteration

The following controls were added after the initial report. They preserve the reference commit's
results while testing address representation, causal conditioning, overlap geometry, and the
rolling-memory mechanism more directly.

### Per-hop oracle intervention

The strict three-hop checkpoint was evaluated with the first `k` query-layer routes replaced by
oracle addresses while all later routes remained native. Every `k` used the same generated
examples.

| Length | Native | First hop oracle | First 2 oracle | First 3 oracle | All 4 oracle |
|---:|---:|---:|---:|---:|---:|
| 64 | 67.6% | 69.5% | 73.4% | 75.8% | 100.0% |
| 128 | 3.1% | 3.5% | 3.9% | 6.6% | 100.0% |
| 256 | 4.7% | 4.3% | 4.7% | 8.2% | 100.0% |

Correcting the explicit first hop does not restore long-context answers. Successive learned routes
also fail on the unseen address grid; the problem is not isolated to the initial pointer encoder.
See [strict_router_interventions.json](../artifacts/results/strict_router_interventions.json).

### Paired pointer counterfactual

Pairs hold values, record positions, types, query, and initial retrieval A fixed. Only the pointer
stored at A changes from B to C. Layer 0 is oracle-routed to A; layer 1 is native.

At length 64, the next predicted address changed in 100% of pairs, the original A-to-B transition
was correct in 95.3%, the modified A-to-C transition in 93.8%, and both were correct in 89.5% of
pairs. At 128, both-correct fell to 0% despite predictions changing in 97.7%. This is stronger causal
evidence that in-distribution `S` follows retrieved pointer content, and equally strong evidence
that the transition does not extrapolate.

### Pointer representation and mixed lengths

Replacing Fourier features with an unlearned raw scalar in fixed hidden dimensions was worse: the
fixed-length model achieved only 14.1% answers at length 64, and mixed-length raw training reached
14.8%. RMS normalization and residual/token mixing make a bare scalar insufficient in this design.

Training the Fourier model cyclically at lengths 32, 48, 64, and 96 materially repaired first-hop
extrapolation:

| Evaluation length | Fixed-length first hop | Mixed-length first hop | Mixed third hop | Mixed answer |
|---:|---:|---:|---:|---:|
| 64 | 95.3% | 92.2% | 33.6% | 29.7% |
| 96 | not measured | 82.0% | 17.2% | 20.3% |
| 128 | 3.9% | 77.3% | 8.6% | 8.6% |
| 192 | not measured | 45.3% | 3.1% | 3.1% |
| 256 | 2.3% | 37.5% | 0.0% | 3.9% |

The initial first-hop collapse was largely single-grid overfitting. Later-hop representation and
error compounding remain unsolved. See
[three_hop_fourier_mixed.json](../artifacts/results/three_hop_fourier_mixed.json).

### True partial overlap

Window width and address stride are now independent. A width-4, stride-1 window can partially
overlap another width-4 window while retaining the same four-token budget as width-4, stride-4.

A boundary task points at a span start while the answer is the marked token immediately after it.
When the start is the last position of a stride-4 block, the block-aligned window cannot contain
the answer; stride 1 always can.

| Length | Block-aligned native / oracle | Partial-overlap native / oracle |
|---:|---:|---:|
| 64 | 77.6% / 77.6% | 100.0% / 100.0% |
| 128 | 10.9% / 69.8% | 13.0% / 97.9% |

The oracle gap validates additional expressivity independently of address prediction. Native
long-context results remain limited by router extrapolation. See
[span_block_aligned.json](../artifacts/results/span_block_aligned.json) and
[span_partial_overlap.json](../artifacts/results/span_partial_overlap.json).

### Rolling working-memory sink without refresh pointers

A source record is placed within eight tokens before a pointer-free memory marker. The marker must
learn through ordinary dense-window attention to absorb the record. A distant final query retrieves
only the marker; no intermediate token receives a pointer or refresh address. A matched `L=1`
control cannot see the source from the marker.

| Model | Length 64 native / oracle | Length 128 native / oracle | Length 256 native / oracle |
|---|---:|---:|---:|
| `L=8` rolling sink | 100.0% / 100.0% | 2.6% / 100.0% | 4.7% / 94.8% |
| `L=1` no rolling | 2.6% / 2.6% | 5.2% / 5.2% | 2.1% / 2.1% |

This demonstrates learned information storage in the rolling region without supplied refresh
pointers. It still uses supervised query-to-marker routing, so it does not demonstrate autonomous
selection of which memories deserve storage. The oracle long-context result shows that stored
content generalizes; native failure is again the address predictor. See
[memory_sink_rolling.json](../artifacts/results/memory_sink_rolling.json) and
[memory_sink_no_rolling.json](../artifacts/results/memory_sink_no_rolling.json).

### Updated assessment

- **Conditioned cascading:** strong in-distribution causal evidence, including paired pointer swaps.
- **Address extrapolation:** mixed-length training fixes much of the first hop, not later hops.
- **Overlap:** partial overlap has demonstrated expressivity at equal retrieval cost.
- **Rolling memory:** a pointer-free sink can learn to store locally visible information for later
  sparse retrieval; autonomous memory prioritization is still untested.
- **Training:** all strongest cascade results still use oracle-consumed routes plus supervised `S`;
  a differentiable or reinforcement-based native-routing objective remains necessary.

Model-only checkpoints for the strict baseline and the principal second-iteration controls are
listed in [artifacts/models/README.md](../artifacts/models/README.md).
