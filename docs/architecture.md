# Architecture specification

## Invariant

At layer `k`, the same pre-normalized incoming state independently produces `Q`, `K`, `V`, and
`S`. `S_k` is not derived from `Q_k`, and every layer owns distinct router parameters. Layer
`k + 1` predicts from the residual state after layer `k` attention, so retrieved content can alter
the next decision without explicit address forwarding or mutable historical KV entries.

For query position `t` (zero based), let `P = t + 1`, dense window `L`, retrieval budget `R`,
window width `W`, address stride `G`, and `J = R / W` address slots.

- If `P <= L + R`, candidates are every position in `[0, t]` exactly once.
- Otherwise, candidates contain `[t - L + 1, t]` plus `J` block-aligned historical windows.
- A router output `a in [0, 1]` maps in constant time to a window starting at
  `G * floor(a * ceil((t - L + 1) / G))`, clamped to the last eligible start.
- Tokens in a partial final block are masked when they cross into the rolling window.
- Windows, and therefore token occurrences, may repeat. No deduplication occurs in the reference
  path.

`G = W` reproduces the initial block-aligned geometry. `G < W` permits partially overlapping
windows without changing the retrieved-token budget.

Selection construction and attention touch at most `L + R` entries per query after the dense
threshold. Historical KV storage remains linear in sequence length. The current batched training
implementation constructs candidate indices for all queries, but the streaming backend processes
bounded query chunks and candidate tiles with online softmax. Activation checkpointing recomputes
those tiles during backward instead of retaining sequence-wide selected K/V tensors. Whole-sequence
training remains linear in sequence length times the fixed selection budget while peak gathered
K/V memory is bounded by the configured tile sizes. Queries through the `L + R` threshold use
optimized causal SDPA directly. The original materialized backend remains available as a numerical
reference. A faster chunked backend materializes selected K/V for one bounded query chunk, flattens
query positions into the batch dimension, and dispatches a single SDPA call. Checkpointing
recomputes each chunk during backward. It uses more bounded temporary memory than the tiled online
softmax path but removes its inner Python/kernel-launch loop. Incremental inference uses a
caller-sized preallocated cache and appends into fresh slots, avoiding an otherwise hidden
`torch.cat` copy of the full history on every token.

## Router training

Hard integer gathers do not transmit gradients into addresses. The first experiments therefore
use explicit route targets and a Smooth L1 auxiliary loss. The initial normalized-coordinate loss
under-penalized one-block errors; later experiments supervise block coordinates directly. During a
bounded warm-up, the first retrieval slot can be teacher-routed while `S` is still evaluated and
trained. `S` executes in FP32 even when Q/K/V use BF16 autocast so scalar precision does not become
the immediate address bottleneck. This is not claimed as semantic retrieval: it isolates whether
direct address prediction and layer-conditioned pointer following are learnable.

The current route target covers only retrieval slot zero. With multiple `S` outputs, hard indexing
also prevents answer gradients from reaching the other slots, so those outputs are unsupervised.
Controlled memory experiments therefore use `R = W = 16`, giving `S` exactly one output. Returning
to multiple windows requires a separate set-valued objective.

The multi-selection control supplies several required regions and trains their addresses as a set.
Every candidate record carries its normalized self-address, while only the query carries the set
that marks candidates as requested. This gives dense attention a content key for a fair
address-matching control without leaking requested status through record tags or values.
For each target, a coverage term is zero for any decoded window start that contains it. A second
one-dimensional order-statistic term sorts both predicted addresses and target intervals before
matching them, making supervision invariant to output-slot permutations while giving every active
output a gradient. Repetition remains possible when there are more outputs than required regions.
The router head can reserve a fixed maximum output capacity so geometries with different active
window counts have identical parameters.

The detached 4K capacity diagnostic isolates the auxiliary router optimizer: the explicit pointer
set is encoded canonically, S reads a detached hidden state, and router/base gradients use separate
clipping and learning rates. Matched corrected controls allow routing gradients into the shared
representation and either retain the full oracle schedule or decay it. An optional two-layer S MLP
tests multi-address decoder capacity without enlarging the transformer. These are supervised
geometry controls, not a claim that semantic address discovery has been solved.

Teacher and oracle controls can mask historical windows independently of their decoded addresses.
The strict oracle sets slot zero to the correct block and disables every other historical slot for
the final query; the rolling window is unchanged. This separates access to the intended cached
entry from interference by native selections.

## Memory write/read diagnostics

The controlled memory tasks place a source outside the marker's retrieved block but inside its
rolling window. The marker must therefore absorb the source through ordinary local attention, and
a distant final query must retrieve the marker through QKVS. An auxiliary marker-token loss is a
training signal, not by itself proof that the marker's cached representation is readable: K/V at
layer `k` are formed from the state entering that layer, before that layer's attention and MLP.

The cache-path probe consequently performs separate forwards in which only one layer may read the
oracle marker window. It also recomputes attention weights from the actual per-layer Q/K inputs and
selected cache indices for source-to-marker and marker-to-query edges. Success with one enabled
layer demonstrates that information was present in that layer's immutable marker K/V, rather than
only in the marker's final classifier state.

## Initial decisions

- Addresses are absolute fractions of the eligible historical-start range.
- One scalar is predicted per retrieved block, shared by all attention heads.
- Rotary position encoding is applied to Q/K; routing uses hidden state directly.
- Duplicate occurrences are retained rather than converted to a log-count bias.
- Synthetic records expose a normalized pointer feature through a small learned encoder.
- Cached K/V tensors are immutable after insertion.
- A query-only retrieval mask exists only as an experimental ablation for removing distributed
  relay paths; it is not part of the proposed production architecture.

## Unresolved decisions and planned ablations

- Address parameterization at much longer lengths: scalar precision, hierarchical/coarse-to-fine,
  or mixed absolute/relative coordinates. Mixed-length Fourier training improves first-hop
  extrapolation but has not solved later-hop composition; a fixed raw-scalar embedding performed
  worse.
- A categorical or hierarchical address representation that avoids high-precision simultaneous
  scalar regression. Set supervision now covers all outputs, but the first eight-address 4K pilot
  was undertrained and its downstream task was confounded; corrected scalar controls come first.
- Straight-through, soft-window, or policy-gradient training without address labels.
- Whether each attention head should own addresses.
- Whether explicit previous-address forwarding helps after controlling for parameter count.
- Kernel layout: checkpointed chunked SDPA is substantially faster than the Python tiled loop, but
  gather, masking, and backward are still not fused. A custom Triton/CUDA path remains necessary.
- Semantic retrieval and practical language modeling are separate later-stage claims.
