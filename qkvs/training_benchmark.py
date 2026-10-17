from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from qkvs.config import ModelConfig
from qkvs.model import QKVSDecoder
from qkvs.runtime import environment, load_json, write_json


def _step(
    model: QKVSDecoder,
    optimizer: torch.optim.Optimizer,
    values: torch.Tensor,
    pointers: torch.Tensor,
    token_types: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(values, pointers, token_types)
        loss = F.cross_entropy(
            output.logits[:, :-1].reshape(-1, model.config.vocab_size).float(),
            values[:, 1:].reshape(-1),
        )
    loss.backward()
    optimizer.step()
    return loss


def _measure(
    model_config: ModelConfig,
    *,
    seq_len: int,
    batch_size: int,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(19001 + seq_len)
    model = QKVSDecoder(model_config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    values = torch.randint(model_config.vocab_size, (batch_size, seq_len), device=device)
    pointers = torch.full((batch_size, seq_len), -1.0, device=device)
    token_types = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)

    for _ in range(warmup):
        _step(model, optimizer, values, pointers, token_types)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    losses: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = _step(model, optimizer, values, pointers, token_types)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
        losses.append(loss.item())
    peak = torch.cuda.max_memory_allocated(device)
    result = {
        "median_step_ms": statistics.median(timings),
        "minimum_step_ms": min(timings),
        "maximum_step_ms": max(timings),
        "tokens_per_second": batch_size * seq_len * 1000.0 / statistics.median(timings),
        "allocated_baseline_bytes": baseline,
        "peak_allocated_bytes": peak,
        "peak_increment_bytes": max(0, peak - baseline),
        "final_loss": losses[-1],
    }
    del model, optimizer, values, pointers, token_types, loss
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark complete attention training steps")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/benchmarks")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training-step benchmarks")
    specification = load_json(arguments.config)
    device = torch.device("cuda")
    batch_size = specification.get("batch_size", 1)
    common = {
        "vocab_size": specification["vocab_size"],
        "d_model": specification["d_model"],
        "n_heads": specification["n_heads"],
        "n_layers": specification["n_layers"],
        "mlp_ratio": specification["mlp_ratio"],
        "dense_window": specification["dense_window"],
        "retrieval_budget": specification["retrieval_budget"],
        "block_size": specification["block_size"],
        "address_stride": specification["address_stride"],
        "dropout": 0.0,
        "query_chunk_size": specification["query_chunk_size"],
        "candidate_tile_size": specification["candidate_tile_size"],
        "checkpoint_sparse": True,
    }
    results: dict[str, Any] = {}
    for seq_len in specification["sequence_lengths"]:
        length_results: dict[str, Any] = {}
        for name, overrides in specification["backends"].items():
            model_config = ModelConfig(**{**common, **overrides})
            length_results[name] = _measure(
                model_config,
                seq_len=seq_len,
                batch_size=batch_size,
                warmup=specification["warmup"],
                repeats=specification["repeats"],
                device=device,
            )
            if overrides.get("attention_kind") != "dense":
                chunk = min(seq_len, overrides.get("query_chunk_size", common["query_chunk_size"]))
                selected = min(
                    seq_len, specification["dense_window"] + specification["retrieval_budget"]
                )
                length_results[name]["bounded_chunk_selected_kv_bytes"] = (
                    2 * batch_size * chunk * selected * specification["d_model"] * 2
                )
        selected = min(seq_len, specification["dense_window"] + specification["retrieval_budget"])
        length_results["theoretical_full_gather_bytes_per_layer"] = (
            2 * batch_size * seq_len * selected * specification["d_model"] * 2
        )
        length_results["theoretical_streaming_tile_bytes"] = (
            2
            * specification["query_chunk_size"]
            * specification["candidate_tile_size"]
            * specification["d_model"]
            * 2
        )
        results[str(seq_len)] = length_results
    result = {
        "config": specification,
        "environment": environment(),
        "results": results,
        "notes": {
            "step": (
                "AdamW zero-grad, BF16-autocast forward, next-token CE, backward, optimizer step."
            ),
            "streaming": (
                "Query chunks use online softmax over candidate tiles and activation checkpointing."
            ),
        },
    }
    output = Path(arguments.output) / f"{specification['name']}.json"
    write_json(output, result)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
