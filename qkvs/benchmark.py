from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from qkvs.attention import build_query_selection, selected_attention
from qkvs.config import ModelConfig
from qkvs.model import QKVSDecoder
from qkvs.runtime import environment, load_json, write_json


def _measure_cuda(
    operation: Callable[[], Tensor], *, warmup: int, repeats: int
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    measurements: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end))
    peak_increment = max(0, torch.cuda.max_memory_allocated() - baseline_memory)
    return {
        "median_ms": statistics.median(measurements),
        "minimum_ms": min(measurements),
        "maximum_ms": max(measurements),
        "tokens_per_second": 1000.0 / statistics.median(measurements),
        "peak_temporary_bytes": peak_increment,
    }


def _decode_benchmark(specification: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    d_model = specification["d_model"]
    n_heads = specification["n_heads"]
    head_dim = d_model // n_heads
    dense_window = specification["dense_window"]
    retrieval_budget = specification["retrieval_budget"]
    block_size = specification["block_size"]
    n_windows = retrieval_budget // block_size
    dtype = torch.bfloat16
    warmup = specification["warmup"]
    repeats = specification["repeats"]
    results: list[dict[str, Any]] = []

    router = nn.Linear(d_model, n_windows, device=device, dtype=torch.float32)
    hidden = torch.randn(1, d_model, device=device, dtype=torch.float32)
    q = torch.randn(1, n_heads, 1, head_dim, device=device, dtype=dtype)

    for seq_len in specification["decode_lengths"]:
        k = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn_like(k)
        fixed_addresses = torch.sigmoid(router(hidden)).detach()
        selection = build_query_selection(
            fixed_addresses,
            seq_len - 1,
            dense_window=dense_window,
            retrieval_budget=retrieval_budget,
            block_size=block_size,
        )

        def full_attention(current_k: Tensor = k, current_v: Tensor = v) -> Tensor:
            return F.scaled_dot_product_attention(q, current_k, current_v, is_causal=False)

        def address_projection() -> Tensor:
            return torch.sigmoid(router(hidden))

        def address_decode(current_length: int = seq_len) -> Tensor:
            addresses = torch.sigmoid(router(hidden))
            return build_query_selection(
                addresses,
                current_length - 1,
                dense_window=dense_window,
                retrieval_budget=retrieval_budget,
                block_size=block_size,
            ).indices

        def gathered_attention(
            current_k: Tensor = k,
            current_v: Tensor = v,
            current_selection=selection,
        ) -> Tensor:
            return selected_attention(q, current_k, current_v, current_selection)

        def combined(
            current_k: Tensor = k,
            current_v: Tensor = v,
            current_length: int = seq_len,
        ) -> Tensor:
            addresses = torch.sigmoid(router(hidden))
            dynamic_selection = build_query_selection(
                addresses,
                current_length - 1,
                dense_window=dense_window,
                retrieval_budget=retrieval_budget,
                block_size=block_size,
            )
            return selected_attention(q, current_k, current_v, dynamic_selection)

        full = _measure_cuda(full_attention, warmup=warmup, repeats=repeats)
        projection = _measure_cuda(address_projection, warmup=warmup, repeats=repeats)
        decode = _measure_cuda(address_decode, warmup=warmup, repeats=repeats)
        gathered = _measure_cuda(gathered_attention, warmup=warmup, repeats=repeats)
        bounded = _measure_cuda(combined, warmup=warmup, repeats=repeats)
        cache_bytes = 2 * specification["n_layers"] * seq_len * d_model * 2
        results.append(
            {
                "sequence_length": seq_len,
                "full_attention": full,
                "address_projection": projection,
                "address_decode_including_projection": decode,
                "gather_and_attention": gathered,
                "bounded_combined": bounded,
                "kv_cache_bytes": cache_bytes,
                "selected_token_occurrences": min(seq_len, dense_window + retrieval_budget),
                "theoretical_score_ratio_full_over_bounded": seq_len
                / min(seq_len, dense_window + retrieval_budget),
            }
        )
        del k, v
    return results


def _prefill_benchmark(specification: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    common = {
        "vocab_size": 32,
        "d_model": specification["d_model"],
        "n_heads": specification["n_heads"],
        "n_layers": specification["n_layers"],
        "dense_window": specification["dense_window"],
        "retrieval_budget": specification["retrieval_budget"],
        "block_size": specification["block_size"],
        "dropout": 0.0,
    }
    models = {
        kind: QKVSDecoder(ModelConfig(**common, attention_kind=kind)).to(device).eval()
        for kind in ("dense", "qkvs")
    }
    for seq_len in specification["prefill_lengths"]:
        values = torch.randint(32, (1, seq_len), device=device)
        pointers = torch.full((1, seq_len), -1.0, device=device)
        token_types = torch.zeros((1, seq_len), device=device, dtype=torch.long)
        entry: dict[str, Any] = {"sequence_length": seq_len}
        for kind, model in models.items():

            def prefill(
                current_model: QKVSDecoder = model,
                current_values: Tensor = values,
                current_pointers: Tensor = pointers,
                current_types: Tensor = token_types,
            ) -> Tensor:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    return current_model(current_values, current_pointers, current_types).logits

            timing = _measure_cuda(
                prefill,
                warmup=max(1, specification["warmup"] // 2),
                repeats=max(3, specification["prefill_repeats"]),
            )
            timing["tokens_per_second"] = seq_len * 1000.0 / timing["median_ms"]
            entry[kind] = timing
        results.append(entry)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark full and direct-address attention")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/benchmarks")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU timing")
    specification = load_json(arguments.config)
    device = torch.device("cuda")
    torch.manual_seed(specification["seed"])
    result = {
        "config": specification,
        "environment": environment(),
        "decode": _decode_benchmark(specification, device),
        "prefill": _prefill_benchmark(specification, device),
        "notes": {
            "timing": (
                "CUDA events around eager PyTorch operations; medians are not kernel-isolated."
            ),
            "kv_cache": "Theoretical BF16 K+V allocation across all configured layers.",
            "complexity": (
                "Address decoding builds only L+R candidates and never scores historical K."
            ),
        },
    }
    output = Path(arguments.output) / f"{specification['name']}.json"
    write_json(output, result)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
