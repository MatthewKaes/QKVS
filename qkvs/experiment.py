from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from qkvs.config import ModelConfig
from qkvs.runtime import load_json, write_json
from qkvs.training import TrainConfig, train_run


def _merged(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    return {**base, **override}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded QKVS pointer experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-device")
    parser.add_argument("--run", action="append", dest="runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps-this-run", type=int)
    arguments = parser.parse_args()
    specification = load_json(arguments.config)
    output_directory = Path(arguments.output)
    device = torch.device(arguments.device)
    evaluation_device = torch.device(arguments.eval_device or arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    summaries: list[dict[str, Any]] = []
    for index, run in enumerate(specification["runs"]):
        if arguments.runs and run["name"] not in arguments.runs:
            continue
        model_values = _merged(specification["model"], run.get("model", {}))
        training_values = _merged(specification["training"], run.get("training", {}))
        training_values["name"] = run["name"]
        training_values["seed"] = run.get("seed", specification.get("seed", 1) + index)
        if "hops" in run:
            training_values["hops"] = run["hops"]
        result = train_run(
            ModelConfig(**model_values),
            TrainConfig.from_dict(training_values),
            output_directory=output_directory,
            device=device,
            evaluation_device=evaluation_device,
            resume=arguments.resume,
            max_steps_this_run=arguments.max_steps_this_run,
        )
        summaries.append(
            {
                "name": result["name"],
                "elapsed_seconds": result["elapsed_seconds"],
                "evaluations": result["evaluations"],
            }
        )
    summary_name = specification["name"]
    if arguments.runs:
        summary_name += "_" + "_".join(arguments.runs)
    write_json(output_directory / f"{summary_name}_summary.json", summaries)


if __name__ == "__main__":
    main()
