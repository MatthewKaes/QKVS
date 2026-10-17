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
reference. Incremental inference uses a caller-sized preallocated cache and appends into fresh
slots, avoiding an otherwise hidden `torch.cat` copy of the full history on every token.

## Router training

Hard integer gathers do not transmit gradients into addresses. The first experiments therefore
use explicit route targets and a Smooth L1 auxiliary loss. The initial normalized-coordinate loss
under-penalized one-block errors; later experiments supervise block coordinates directly. During a
bounded warm-up, the first retrieval slot can be teacher-routed while `S` is still evaluated and
trained. `S` executes in FP32 even when Q/K/V use BF16 autocast so scalar precision does not become
the immediate address bottleneck. This is not claimed as semantic retrieval: it isolates whether
direct address prediction and layer-conditioned pointer following are learnable.

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
- Permutation-invariant supervision when several router slots all carry useful targets.
- Straight-through, soft-window, or policy-gradient training without address labels.
- Whether each attention head should own addresses.
- Whether explicit previous-address forwarding helps after controlling for parameter count.
- Kernel layout: gathered attention is first kept in PyTorch; Triton/CUDA work is justified only
  after correctness and learning evidence. The checkpointed streaming implementation solves peak
  training memory but is substantially slower; fusing online softmax, gather, and backward remains
  necessary for throughput.
- Semantic retrieval and practical language modeling are separate later-stage claims.
