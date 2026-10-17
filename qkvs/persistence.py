from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from qkvs.attention import normalized_window_addresses
from qkvs.config import ModelConfig
from qkvs.data import QUERY, RECORD
from qkvs.model import QKVSDecoder
from qkvs.runtime import environment, load_json, write_json


def _make_batch(
    *,
    batch_size: int,
    seq_len: int,
    config: ModelConfig,
    mode: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.randint(config.vocab_size, (batch_size, seq_len), generator=generator)
    pointers = torch.full((batch_size, seq_len), -1.0)
    token_types = torch.zeros((batch_size, seq_len), dtype=torch.long)
    retrieval_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    answers = torch.empty(batch_size, dtype=torch.long)
    first_access = max(config.dense_window + config.retrieval_budget, seq_len // 2)
    spacing = max(2, config.dense_window // 2)

    for item in range(batch_size):
        remote_limit = max(1, seq_len // 4)
        target = torch.randint(remote_limit, (), generator=generator)
        answers[item] = values[item, target]
        token_types[item, target] = RECORD
        token_types[item, -1] = QUERY

        if mode == "frequent":
            access_positions = list(range(first_access, seq_len - 1, spacing))
        elif mode == "once":
            access_positions = [first_access]
        elif mode in {"never_until_query", "no_access"}:
            access_positions = []
        else:
            raise ValueError(f"unknown persistence mode: {mode}")

        if mode == "never_until_query":
            access_positions.append(seq_len - 1)

        for position in access_positions:
            history_end = position - config.dense_window + 1
            if target >= history_end:
                raise AssertionError("persistence target was not historical")
            pointers[item, position] = normalized_window_addresses(
                target, history_end, config.attention().stride
            )
            token_types[item, position] = QUERY
            retrieval_mask[item, position] = True

    return values, pointers, token_types, retrieval_mask, answers


def _evaluate_mode(
    model: QKVSDecoder,
    config: ModelConfig,
    *,
    mode: str,
    seq_len: int,
    batches: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    correct = 0
    total = 0
    losses: list[float] = []
    for _ in range(batches):
        values, pointers, token_types, retrieval_mask, answers = _make_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            config=config,
            mode=mode,
            generator=generator,
        )
        with torch.no_grad():
            logits = (
                model(
                    values.to(device),
                    pointers.to(device),
                    token_types.to(device),
                    retrieval_mask=retrieval_mask.to(device),
                )
                .logits[:, -1]
                .float()
            )
        answers = answers.to(device)
        losses.append(torch.nn.functional.cross_entropy(logits, answers).item())
        correct += (logits.argmax(dim=-1) == answers).sum().item()
        total += answers.numel()
    return {
        "answer_accuracy": correct / total,
        "answer_loss": sum(losses) / len(losses),
        "examples": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieved-information persistence")
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
    results: dict[str, Any] = {}
    for seq_len in specification["sequence_lengths"]:
        results[str(seq_len)] = {}
        for mode_index, mode in enumerate(specification["modes"]):
            results[str(seq_len)][mode] = _evaluate_mode(
                model,
                config,
                mode=mode,
                seq_len=seq_len,
                batches=specification["batches"],
                batch_size=specification["batch_size"],
                seed=specification["seed"] + seq_len * 10 + mode_index,
                device=device,
            )
    result = {
        "config": specification,
        "model_config": config.as_dict(),
        "environment": environment(),
        "results": results,
        "interpretation_guardrail": (
            "Frequent/once modes disable historical retrieval at the final query. Accuracy above "
            "the no-access control indicates propagation through ordinary layer states, not a "
            "mutable KV cache."
        ),
    }
    output = Path(arguments.output) / f"{specification['name']}.json"
    write_json(output, result)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
