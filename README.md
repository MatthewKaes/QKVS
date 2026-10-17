# QKVS: learned direct-address sparse attention

This repository is an independent research prototype for a decoder-only transformer with an
independent `S` projection at every layer. `S` predicts normalized addresses of old KV-cache
windows directly; inference does not scan or score the historical cache to choose them.

The implementation deliberately separates four questions:

1. Is fixed-budget direct-address attention mathematically correct?
2. Can a supervised router learn explicit addresses and extrapolate to longer contexts?
3. Can independently conditioned layer routers follow multi-hop pointer chains?
4. Does bounded attention translate to useful wall-clock behavior on an RTX 5070?

The first prototype uses hard, block-aligned gathers for the forward pass and an auxiliary
routing loss (optionally with teacher-routed warm-up) to train the otherwise nondifferentiable
address choice. Duplicate windows remain duplicated in the attention sequence, preserving their
softmax multiplicity semantics.

## Setup

The pinned CUDA 13.0 PyTorch wheel supports the RTX 5070's `sm_120` architecture. The project is
local and does not modify the system CUDA installation.

```bash
uv sync
uv run pytest
uv run python -m qkvs.environment
```

Run bounded experiments and benchmarks with:

```bash
uv run python -m qkvs.experiment --config configs/smoke.json
uv run python -m qkvs.benchmark --config configs/benchmark.json
```

Every command writes a JSON artifact containing its configuration, seed, environment, and raw
measurements under `artifacts/`. See [docs/architecture.md](docs/architecture.md) for the precise
selection rule and unresolved research decisions. The evidence and conclusions from the initial
study are in [docs/research_report.md](docs/research_report.md).

The reviewer-directed second iteration adds per-hop oracle interventions, paired pointer
counterfactuals, mixed-length address training, true partial-overlap windows, and a pointer-free
rolling-memory sink. Key model-only checkpoints are included under
[artifacts/models](artifacts/models/README.md) for replaying learned routes without optimizer state.

Additional recorded ablations can be reproduced with:

```bash
uv run python -m qkvs.persistence --config configs/persistence.json --device cpu
uv run python -m qkvs.overlap --config configs/overlap.json --device cpu
uv run python -m qkvs.interventions --config configs/interventions.json --device cpu
uv run python -m qkvs.training_benchmark --config configs/training_benchmark.json
```

The local NVIDIA driver has a known history of sporadic `Xid 79` bus-loss events. Training runs are
therefore individually selectable with `--run`, atomically checkpoint every 100 steps, support
`--resume`, and can move evaluation to CPU with `--eval-device cpu`.

Longer runs can be split into independent driver sessions with `--max-steps-this-run N` and resumed
from atomic model/optimizer/RNG checkpoints. The streaming sparse backend uses online softmax,
bounded query/candidate tiles, and activation checkpointing to avoid sequence-wide selected-K/V
materialization during training.

The controlled write/read diagnostics use one supervised historical window, a true oracle that
disables unrelated historical windows, and per-layer probes of the actual selected marker K/V:

```bash
uv run python -m qkvs.gradient_audit
uv run python -m qkvs.experiment --config configs/local_memory_diagnostics.json \
  --run local_memory_general --max-steps-this-run 50
uv run python -m qkvs.memory_diagnostic \
  --config configs/local_memory_diagnostics.json \
  --run local_memory_general \
  --checkpoint artifacts/checkpoints/local_memory_general.pt \
  --output artifacts/results/local_memory_general_diagnostic.json
uv run python -m qkvs.memory_path_probe \
  --config configs/local_memory_diagnostics.json \
  --run local_memory_general \
  --checkpoint artifacts/checkpoints/local_memory_general.pt \
  --output artifacts/results/local_memory_general_kv_probe.json
```

The matching model-only checkpoints and hashes are listed in
[artifacts/models/README.md](artifacts/models/README.md). On the affected host, keep CUDA invocations
short and checkpointed because the driver has a known history of sporadic bus-loss failures.

The corrected multi-selection study fixes `L=R=256` and router capacity at eight outputs while
varying active window count, width, and stride. All 100 candidate records are tagged, populated, and
labeled with their own address; only the explicit query pointer set selects the requested records.
The generator also rejects candidate layouts that collapse multiple records onto one stride
address:

```bash
uv run python -m qkvs.training_benchmark \
  --config configs/training_benchmark_multiselect.json
uv run python -m qkvs.experiment \
  --config configs/multiselect_geometry.json \
  --run v5_regions8_pointer_identity_mixed \
  --max-steps-this-run 100
uv run python -m qkvs.multiselect_diagnostic \
  --config configs/multiselect_geometry.json \
  --run v5_regions8_pointer_identity_mixed \
  --checkpoint \
    artifacts/checkpoints/v5_regions8_pointer_identity_mixed.pt \
  --output \
    artifacts/results/v5_regions8_pointer_identity_mixed_diagnostic.json
```

The V5 identity gate was stopped at step 500. Exact supplied-pointer routing remained at chance
answer accuracy despite all audited windows containing their intended records, so the joint V5 run
is blocked. See the [step-500 report](docs/research_report.md#v5-direct-address-identity-gate-failed-at-step-500)
and [gate audit](artifacts/results/v5_regions8_pointer_identity_mixed_step500_gate_audit.json).

The `chunked` sparse backend is the current training default for this study. It gathers only one
bounded query chunk and uses SDPA, trading more temporary memory for far fewer launches than the
online-softmax reference.

On the affected Linux host, run CUDA commands from the user's terminal rather than through the
Codex tool executor. Ordinary CPU tests and diagnostics remain safe to run through Codex.
