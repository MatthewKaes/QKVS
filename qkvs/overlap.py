from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from qkvs.attention import duplicate_attention_reference, log_multiplicity_reference
from qkvs.config import ModelConfig
from qkvs.data import make_pointer_batch
from qkvs.model import QKVSDecoder
from qkvs.runtime import environment, load_json, write_json


def _unique_blocks(blocks: torch.Tensor, n_blocks: int) -> torch.Tensor:
    """Replace duplicate blocks by the nearest unused block without changing slot count."""
    result = blocks.clone()
    layers, batch_size, windows = blocks.shape
    for layer in range(layers):
        for item in range(batch_size):
            used: set[int] = set()
            for window in range(windows):
                proposed = int(blocks[layer, item, window])
                if proposed not in used:
                    chosen = proposed
                else:
                    chosen = proposed
                    for distance in range(1, n_blocks):
                        candidates = (proposed - distance, proposed + distance)
                        available = [
                            value
                            for value in candidates
                            if 0 <= value < n_blocks and value not in used
                        ]
                        if available:
                            chosen = available[0]
                            break
                used.add(chosen)
                result[layer, item, window] = chosen
    return result


def _equivalence_error(seed: int, trials: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    maximum = 0.0
    for _ in range(trials):
        q = torch.randn(16, generator=generator)
        k = torch.randn(12, 16, generator=generator)
        v = torch.randn(12, 9, generator=generator)
        indices = torch.randint(12, (24,), generator=generator)
        explicit = duplicate_attention_reference(q, k, v, indices)
        biased = log_multiplicity_reference(q, k, v, indices)
        maximum = max(maximum, (explicit - biased).abs().max().item())
    return maximum


def _evaluate_length(
    model: QKVSDecoder,
    config: ModelConfig,
    *,
    seq_len: int,
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    native_correct = 0
    unique_correct = 0
    total = 0
    duplicate_slots = 0
    total_slots = 0
    examples_with_duplicates = 0
    target_multiplicity = 0
    duplicate_example_correct = 0
    duplicate_example_total = 0

    for _ in range(batches):
        batch = make_pointer_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=config.vocab_size,
            dense_window=config.dense_window,
            block_size=config.block_size,
            n_layers=config.n_layers,
            hops=1,
            generator=generator,
        ).to(device)
        with torch.no_grad():
            native = model(batch.values, batch.pointers, batch.token_types)
        assert native.addresses is not None
        n_blocks = math.ceil((seq_len - config.dense_window) / config.block_size)
        addresses = native.addresses[:, :, -1].float()
        blocks = torch.floor(
            addresses.clamp(0.0, 1.0 - torch.finfo(torch.float32).eps) * n_blocks
        ).to(torch.long)
        unique_counts = torch.stack(
            [
                torch.tensor(
                    [torch.unique(row).numel() for row in layer],
                    device=device,
                )
                for layer in blocks
            ]
        )
        duplicates = config.attention().n_windows - unique_counts
        duplicate_slots += duplicates.sum().item()
        total_slots += duplicates.numel() * config.attention().n_windows
        has_duplicate = (duplicates > 0).any(dim=0)
        examples_with_duplicates += has_duplicate.sum().item()

        target_blocks = torch.div(
            batch.route_positions,
            config.block_size,
            rounding_mode="floor",
        )
        target_multiplicity += (blocks == target_blocks.unsqueeze(-1)).sum().item()
        native_predictions = native.logits[:, -1].argmax(dim=-1)
        native_hits = native_predictions == batch.answers
        native_correct += native_hits.sum().item()
        duplicate_example_correct += native_hits[has_duplicate].sum().item()
        duplicate_example_total += has_duplicate.sum().item()

        forced_blocks = _unique_blocks(blocks.cpu(), n_blocks).to(device)
        forced_addresses = (forced_blocks.float() + 0.5) / n_blocks
        overrides = torch.full(
            (
                config.n_layers,
                batch_size,
                seq_len,
                config.attention().n_windows,
            ),
            torch.nan,
            device=device,
        )
        overrides[:, :, -1] = forced_addresses
        with torch.no_grad():
            unique = model(
                batch.values,
                batch.pointers,
                batch.token_types,
                overrides,
            )
        unique_correct += (unique.logits[:, -1].argmax(dim=-1) == batch.answers).sum().item()
        total += batch_size

    return {
        "native_answer_accuracy": native_correct / total,
        "forced_unique_answer_accuracy": unique_correct / total,
        "duplicate_slot_fraction": duplicate_slots / total_slots,
        "examples_with_any_duplicate_fraction": examples_with_duplicates / total,
        "mean_target_occurrences_across_layers": target_multiplicity / total,
        "native_accuracy_on_duplicate_examples": (
            duplicate_example_correct / duplicate_example_total if duplicate_example_total else None
        ),
        "examples": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate duplicate and unique window behavior")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/results")
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    specification = load_json(arguments.config)
    device = torch.device(arguments.device)
    checkpoint = torch.load(specification["checkpoint"], map_location=device, weights_only=True)
    config = ModelConfig(**checkpoint["model_config"])
    model = QKVSDecoder(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    results = {
        str(seq_len): _evaluate_length(
            model,
            config,
            seq_len=seq_len,
            batch_size=specification["batch_size"],
            batches=specification["batches"],
            seed=specification["seed"] + seq_len,
            device=device,
        )
        for seq_len in specification["sequence_lengths"]
    }
    result = {
        "config": specification,
        "model_config": config.as_dict(),
        "environment": environment(),
        "log_multiplicity_max_absolute_error": _equivalence_error(
            specification["seed"], specification["equivalence_trials"]
        ),
        "results": results,
        "comparison_note": (
            "Forced-unique overrides are derived from the native first pass. Later-layer "
            "addresses therefore reflect native preceding states; this isolates selection "
            "multiplicity but is not a separately trained unique-router baseline."
        ),
    }
    output = Path(arguments.output) / f"{specification['name']}.json"
    write_json(output, result)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
