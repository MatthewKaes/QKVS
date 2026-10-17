from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from qkvs.config import ModelConfig
from qkvs.model import QKVSDecoder
from qkvs.runtime import environment, load_json, write_json
from qkvs.training import TrainConfig, evaluate


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
            )
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
