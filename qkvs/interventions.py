from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from qkvs.attention import normalized_window_addresses
from qkvs.config import ModelConfig
from qkvs.data import (
    FILLER,
    QUERY,
    RECORD,
    PointerBatch,
    make_address_overrides,
    make_pointer_batch,
)
from qkvs.model import QKVSDecoder
from qkvs.runtime import environment, load_json, write_json


def _route_correct(
    addresses: torch.Tensor, batch: PointerBatch, config: ModelConfig
) -> torch.Tensor:
    history_end = batch.values.shape[1] - config.dense_window
    stride = config.attention().stride
    n_addresses = math.ceil(history_end / stride)
    starts = (
        torch.floor(
            addresses[:, :, -1].float().clamp(0.0, 1.0 - torch.finfo(torch.float32).eps)
            * n_addresses
        ).to(torch.long)
        * stride
    )
    targets = batch.route_positions.unsqueeze(-1)
    return ((starts <= targets) & (targets < starts + config.block_size)).any(dim=-1)


def _prefix_evaluation(
    model: QKVSDecoder,
    config: ModelConfig,
    specification: dict[str, Any],
    *,
    seq_len: int,
    prefix_layers: int,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(specification["seed"] + seq_len * 100)
    answer_correct = 0
    total = 0
    route_correct = torch.zeros(config.n_layers, device=device)
    for _ in range(specification["batches"]):
        batch = make_pointer_batch(
            batch_size=specification["batch_size"],
            seq_len=seq_len,
            vocab_size=config.vocab_size,
            dense_window=config.dense_window,
            block_size=config.block_size,
            address_stride=config.attention().stride,
            n_layers=config.n_layers,
            hops=specification["hops"],
            generator=generator,
        ).to(device)
        overrides = make_address_overrides(
            batch,
            seq_len=seq_len,
            n_windows=config.attention().n_windows,
            device=device,
        )
        overrides[prefix_layers:] = torch.nan
        retrieval_mask = torch.zeros_like(batch.values, dtype=torch.bool)
        retrieval_mask[:, -1] = True
        with torch.no_grad():
            output = model(
                batch.values,
                batch.pointers,
                batch.token_types,
                overrides,
                retrieval_mask,
            )
        assert output.addresses is not None
        answer_correct += (output.logits[:, -1].argmax(dim=-1) == batch.answers).sum().item()
        route_correct += _route_correct(output.addresses, batch, config).sum(dim=1)
        total += batch.answers.numel()
    return {
        "answer_accuracy": answer_correct / total,
        "predicted_route_accuracy_by_layer": [value.item() / total for value in route_correct],
        "examples": total,
    }


def _counterfactual_pair(
    *,
    batch_size: int,
    seq_len: int,
    config: ModelConfig,
    generator: torch.Generator,
) -> tuple[PointerBatch, PointerBatch]:
    history_end = seq_len - config.dense_window
    values = torch.randint(config.vocab_size, (batch_size, seq_len), generator=generator)
    base_pointers = torch.full((batch_size, seq_len), -1.0)
    token_types = torch.full((batch_size, seq_len), FILLER, dtype=torch.long)
    original_positions = torch.empty((config.n_layers, batch_size), dtype=torch.long)
    modified_positions = torch.empty_like(original_positions)
    original_answers = torch.empty(batch_size, dtype=torch.long)
    modified_answers = torch.empty_like(original_answers)
    original_pointers = base_pointers.clone()
    modified_pointers = base_pointers.clone()
    stride = config.attention().stride

    for item in range(batch_size):
        a, b, c = torch.randperm(history_end, generator=generator)[:3]
        if values[item, b] == values[item, c]:
            values[item, c] = (values[item, b] + 1) % config.vocab_size
        token_types[item, [a, b, c]] = RECORD
        token_types[item, -1] = QUERY
        address_a = normalized_window_addresses(a, history_end, stride)
        address_b = normalized_window_addresses(b, history_end, stride)
        address_c = normalized_window_addresses(c, history_end, stride)
        original_pointers[item, -1] = address_a
        modified_pointers[item, -1] = address_a
        original_pointers[item, a] = address_b
        modified_pointers[item, a] = address_c
        original_positions[:, item] = b
        modified_positions[:, item] = c
        original_positions[0, item] = a
        modified_positions[0, item] = a
        original_answers[item] = values[item, b]
        modified_answers[item] = values[item, c]

    original_routes = normalized_window_addresses(original_positions, history_end, stride)
    modified_routes = normalized_window_addresses(modified_positions, history_end, stride)
    original = PointerBatch(
        values,
        original_pointers,
        token_types,
        original_answers,
        original_routes,
        original_positions,
        2,
    )
    modified = PointerBatch(
        values.clone(),
        modified_pointers,
        token_types.clone(),
        modified_answers,
        modified_routes,
        modified_positions,
        2,
    )
    return original, modified


def _counterfactual_evaluation(
    model: QKVSDecoder,
    config: ModelConfig,
    specification: dict[str, Any],
    *,
    seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(specification["seed"] + 900_000 + seq_len)
    original_transition_correct = 0
    modified_transition_correct = 0
    paired_transition_correct = 0
    prediction_changed = 0
    original_answer_correct = 0
    modified_answer_correct = 0
    total = 0

    for _ in range(specification["batches"]):
        original, modified = _counterfactual_pair(
            batch_size=specification["batch_size"],
            seq_len=seq_len,
            config=config,
            generator=generator,
        )
        original = original.to(device)
        modified = modified.to(device)
        retrieval_mask = torch.zeros_like(original.values, dtype=torch.bool)
        retrieval_mask[:, -1] = True

        outputs = []
        for batch in (original, modified):
            overrides = make_address_overrides(
                batch,
                seq_len=seq_len,
                n_windows=config.attention().n_windows,
                device=device,
            )
            overrides[1:] = torch.nan
            with torch.no_grad():
                outputs.append(
                    model(
                        batch.values,
                        batch.pointers,
                        batch.token_types,
                        overrides,
                        retrieval_mask,
                    )
                )
        original_output, modified_output = outputs
        assert original_output.addresses is not None
        assert modified_output.addresses is not None
        original_correct = _route_correct(original_output.addresses, original, config)[1]
        modified_correct = _route_correct(modified_output.addresses, modified, config)[1]
        original_transition_correct += original_correct.sum().item()
        modified_transition_correct += modified_correct.sum().item()
        paired_transition_correct += (original_correct & modified_correct).sum().item()

        history_end = seq_len - config.dense_window
        n_addresses = math.ceil(history_end / config.attention().stride)
        original_index = torch.floor(original_output.addresses[1, :, -1, 0] * n_addresses).to(
            torch.long
        )
        modified_index = torch.floor(modified_output.addresses[1, :, -1, 0] * n_addresses).to(
            torch.long
        )
        prediction_changed += (original_index != modified_index).sum().item()
        original_answer_correct += (
            (original_output.logits[:, -1].argmax(dim=-1) == original.answers).sum().item()
        )
        modified_answer_correct += (
            (modified_output.logits[:, -1].argmax(dim=-1) == modified.answers).sum().item()
        )
        total += original.answers.numel()

    return {
        "original_next_route_accuracy": original_transition_correct / total,
        "modified_next_route_accuracy": modified_transition_correct / total,
        "paired_both_routes_correct": paired_transition_correct / total,
        "next_prediction_changed_fraction": prediction_changed / total,
        "original_answer_accuracy": original_answer_correct / total,
        "modified_answer_accuracy": modified_answer_correct / total,
        "examples": total,
        "input_difference": "Only the pointer stored at A differs between each pair.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run per-hop and paired routing interventions")
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

    prefix_results: dict[str, Any] = {}
    counterfactual_results: dict[str, Any] = {}
    for seq_len in specification["sequence_lengths"]:
        prefix_results[str(seq_len)] = {
            str(prefix): _prefix_evaluation(
                model,
                config,
                specification,
                seq_len=seq_len,
                prefix_layers=prefix,
                device=device,
            )
            for prefix in specification["oracle_prefix_layers"]
        }
        counterfactual_results[str(seq_len)] = _counterfactual_evaluation(
            model,
            config,
            specification,
            seq_len=seq_len,
            device=device,
        )

    result = {
        "config": specification,
        "model_config": config.as_dict(),
        "environment": environment(),
        "oracle_prefix_results": prefix_results,
        "paired_counterfactual_results": counterfactual_results,
    }
    output = Path(arguments.output) / f"{specification['name']}.json"
    write_json(output, result)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
