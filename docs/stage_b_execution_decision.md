# Stage B execution decision

## Decision

**Stop before Gate 0.** The preregistered full-length batch-versus-incremental causal-equivalence
audit failed. The causal-oracle dataset evaluation did not run, and the eight-slot semantic
directory was not constructed or trained.

The unmodified raw result is preserved in
[`stage_b_full_length_causal_equivalence.json`](../artifacts/results/stage_b_full_length_causal_equivalence.json).
It was produced from source revision `43f510a5fedbb90f9895e72aa530f97930f54076` with a clean tracked
source tree. The artifact contains the complete frozen configuration, source-file hashes, dataset
checksum, input-token hash, and raw measurements. Those recorded hashes match the repository at
that revision.

## Result

The audit evaluated example 0 at 7,263 tokens on CUDA in BF16. It compared batched execution with
token-by-token cached incremental execution for both the rolling mask and the causally aligned
oracle mask.

| Mode | Hidden max error | Limit | Logit max error | Limit | Last-hidden cosine | Minimum | Top-1 match |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rolling | 15.75 | 0.002 | 0.53125 | 0.0001 | 0.999763131 | 0.999999 | Yes |
| Causal oracle | 15.75 | 0.002 | 0.390625 | 0.0001 | 0.999762058 | 0.999999 | Yes |

The gate required every numerical bound and the top-1 check to pass. Matching top-1 tokens do not
override the three failed numerical conditions. The largest hidden-state error exceeded its bound
by 7,875 times; the rolling and causal-oracle logit errors exceeded their bound by 5,312.5 and
3,906.25 times respectively.

The batched runs took 0.382 seconds for rolling and 0.295 seconds for causal oracle. Their
incremental counterparts took 62.843 and 62.159 seconds. These timings describe the audit paths;
they are not a preregistered throughput comparison.

## Interpretation boundary

Both mask modes have the same maximum hidden-state error and nearly identical mean hidden-state
errors. That pattern suggests the discrepancy lies in execution behavior shared by the batched and
incremental BF16 paths rather than in the oracle window alone. The current artifact does not
localize the error by token or layer, so it cannot distinguish ordinary long-sequence BF16 kernel
drift from a defect in the incremental reference.

The failure must not be relabeled as semantic equivalence based on matching top-1 output. The
thresholds were frozen before execution, and the protocol states that failure blocks subsequent
work. Any diagnostic amendment or revised numerical criterion now requires arbitration as a
substantive protocol change. Until then:

- do not run causal-oracle Gate 0;
- do not implement or train `S`;
- do not change directory capacity, retrieval layer, or execution topology; and
- do not reuse this failed result as positive Stage B evidence.
