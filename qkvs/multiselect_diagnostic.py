from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from qkvs.config import ModelConfig
from qkvs.data import (
    PointerBatch,
    make_address_overrides,
    make_multi_region_counterfactual_pair,
    make_oracle_window_masks,
    make_query_pointer_overrides,
)
from qkvs.model import QKVSDecoder
from qkvs.runtime import environment, load_json, write_json
from qkvs.training import TrainConfig, _retrieval_mask, evaluate


def _counterfactual_predictions(
    model: QKVSDecoder,
    batch: PointerBatch,
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    oracle_source: str | None,
    device: torch.device,
) -> torch.Tensor:
    batch = batch.to(device)
    overrides = None
    window_masks = None
    if oracle_source is not None and model_config.attention_kind == "qkvs":
        if oracle_source == "coverage_oracle":
            overrides = make_address_overrides(
                batch,
                seq_len=batch.values.shape[1],
                n_windows=model_config.attention().n_windows,
                device=device,
            )
        elif oracle_source == "query_pointers":
            overrides = make_query_pointer_overrides(
                batch,
                seq_len=batch.values.shape[1],
                n_windows=model_config.attention().n_windows,
                device=device,
            )
        else:
            raise ValueError(f"unknown counterfactual oracle source: {oracle_source}")
        window_masks = make_oracle_window_masks(
            batch,
            seq_len=batch.values.shape[1],
            n_windows=model_config.attention().n_windows,
            device=device,
        )
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
            address_overrides=overrides,
            retrieval_mask=_retrieval_mask(batch, train_config),
            historical_window_masks=window_masks,
        )
    return output.logits[:, -1].argmax(dim=-1).cpu()


def evaluate_counterfactual(
    model: QKVSDecoder,
    model_config: ModelConfig,
    train_config: TrainConfig,
    seq_len: int,
    *,
    oracle_source: str | None,
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    """Measure whether changing only the requested pointer set changes the predicted count."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    pairs = 0
    correct = 0
    both_correct = 0
    predictions_changed = 0
    for _ in range(train_config.eval_batches):
        batch_size = train_config.eval_batch_size or train_config.batch_size
        first, second = make_multi_region_counterfactual_pair(
            batch_size=batch_size,
            seq_len=seq_len,
            dense_window=model_config.dense_window,
            block_size=model_config.block_size,
            address_stride=model_config.attention().stride,
            n_windows=model_config.attention().n_windows,
            n_layers=model_config.n_layers,
            n_records=train_config.region_records,
            required_regions=train_config.required_regions,
            generator=generator,
        )
        first_predictions = _counterfactual_predictions(
            model,
            first,
            model_config,
            train_config,
            oracle_source=oracle_source,
            device=device,
        )
        second_predictions = _counterfactual_predictions(
            model,
            second,
            model_config,
            train_config,
            oracle_source=oracle_source,
            device=device,
        )
        first_correct = first_predictions == first.answers
        second_correct = second_predictions == second.answers
        correct += first_correct.sum().item() + second_correct.sum().item()
        both_correct += (first_correct & second_correct).sum().item()
        predictions_changed += (first_predictions != second_predictions).sum().item()
        pairs += batch_size
    return {
        "answer_accuracy": correct / (2 * pairs),
        "both_answers_correct": both_correct / pairs,
        "prediction_changed": predictions_changed / pairs,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-region routing and answers")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--seq-len", type=int, action="append", dest="seq_lens")
    arguments = parser.parse_args()
    specification = load_json(arguments.config)
    run_index, run = next(
        (index, value)
        for index, value in enumerate(specification["runs"])
        if value["name"] == arguments.run
    )
    model_config = ModelConfig(**{**specification["model"], **run.get("model", {})})
    train_config = TrainConfig.from_dict(
        {
            **specification["training"],
            **run.get("training", {}),
            "name": run["name"],
            "seed": run.get("seed", specification["seed"] + run_index),
            "eval_batches": arguments.batches,
        }
    )
    device = torch.device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=True)
    model = QKVSDecoder(model_config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    seq_lens = arguments.seq_lens or list(train_config.eval_seq_lens)
    results: dict[str, Any] = {}
    for seq_len in seq_lens:
        seed = train_config.seed + 80_000 + seq_len
        results[str(seq_len)] = {
            "native": evaluate(
                model,
                model_config,
                train_config,
                seq_len,
                oracle=False,
                seed=seed,
                device=device,
            ),
            "paired_counterfactual_native": evaluate_counterfactual(
                model,
                model_config,
                train_config,
                seq_len,
                oracle_source=None,
                seed=seed + 10_000,
                device=device,
            ),
        }
        if model_config.attention_kind == "qkvs":
            results[str(seq_len)]["coverage_oracle"] = evaluate(
                model,
                model_config,
                train_config,
                seq_len,
                oracle=True,
                seed=seed,
                device=device,
            )
            results[str(seq_len)]["paired_counterfactual_oracle"] = evaluate_counterfactual(
                model,
                model_config,
                train_config,
                seq_len,
                oracle_source="coverage_oracle",
                seed=seed + 10_000,
                device=device,
            )
            results[str(seq_len)]["pointer_identity"] = evaluate(
                model,
                model_config,
                train_config,
                seq_len,
                oracle=True,
                oracle_source="query_pointers",
                seed=seed,
                device=device,
            )
            results[str(seq_len)]["paired_counterfactual_pointer_identity"] = (
                evaluate_counterfactual(
                    model,
                    model_config,
                    train_config,
                    seq_len,
                    oracle_source="query_pointers",
                    seed=seed + 10_000,
                    device=device,
                )
            )
    result = {
        "checkpoint_step": checkpoint.get("step"),
        "run": run,
        "environment": environment(),
        "balanced_class_baseline": 1.0 / (train_config.required_regions + 1),
        "results": results,
    }
    write_json(Path(arguments.output), result)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
