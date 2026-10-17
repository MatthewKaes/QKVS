from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from qkvs.config import ModelConfig
from qkvs.model import QKVSDecoder
from qkvs.runtime import environment, load_json, write_json
from qkvs.training import TrainConfig, _batch, evaluate


def _marker_accuracy(
    model: QKVSDecoder,
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    seq_len: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    correct = 0
    total = 0
    for _ in range(batches):
        batch = _batch(
            model_config,
            train_config,
            seq_len,
            generator,
            batch_size=train_config.eval_batch_size,
        ).to(device)
        retrieval_mask = torch.zeros_like(batch.values, dtype=torch.bool)
        retrieval_mask[:, -1] = True
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
        ):
            output = model(
                batch.values,
                batch.pointers,
                batch.token_types,
                retrieval_mask=retrieval_mask,
                semantic_ids=batch.semantic_ids,
            )
        assert batch.memory_targets is not None
        valid = batch.memory_targets >= 0
        correct += (output.logits.argmax(dim=-1)[valid] == batch.memory_targets[valid]).sum().item()
        total += valid.sum().item()
    return {"accuracy": correct / total, "correct": correct, "examples": total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose scaled memory formation and retrieval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/results/scaled_memory_diagnostic.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=32)
    arguments = parser.parse_args()
    specification = load_json(arguments.config)
    run = next(value for value in specification["runs"] if value["name"] == arguments.run)
    model_values = {**specification["model"], **run.get("model", {})}
    training_values = {
        **specification["training"],
        **run.get("training", {}),
        "name": run["name"],
        "seed": specification["seed"],
        "eval_batches": arguments.batches,
    }
    model_config = ModelConfig(**model_values)
    train_config = TrainConfig.from_dict(training_values)
    device = torch.device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=True)
    model = QKVSDecoder(model_config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    results: dict[str, Any] = {}
    for seq_len in specification["training"]["eval_seq_lens"][:2]:
        seed = specification["seed"] + 50_000 + seq_len
        results[str(seq_len)] = {
            "marker": _marker_accuracy(
                model,
                model_config,
                train_config,
                seq_len=seq_len,
                batches=arguments.batches,
                seed=seed,
                device=device,
            ),
            "native": evaluate(
                model,
                model_config,
                train_config,
                seq_len,
                oracle=False,
                seed=seed,
                device=device,
            ),
            "oracle": evaluate(
                model,
                model_config,
                train_config,
                seq_len,
                oracle=True,
                seed=seed,
                device=device,
            ),
        }
    result = {
        "config": specification,
        "run": run,
        "checkpoint_step": checkpoint.get("step"),
        "environment": environment(),
        "results": results,
        "interpretation": (
            "Marker accuracy diagnoses local memory formation. Oracle final-query routes isolate "
            "stored-content capacity from native routing."
        ),
    }
    output = Path(arguments.output)
    write_json(output, result)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
