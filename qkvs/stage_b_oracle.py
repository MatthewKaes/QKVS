from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from qkvs.causal_reference import (
    causal_oracle_mask_plan,
    qwen_forward_with_layer_masks,
    rolling_mask_plan,
    stage_a_oracle_mask_plan,
)
from qkvs.runtime import environment, load_json, write_json
from qkvs.utility_gate import (
    UtilityExample,
    _bootstrap_mean_interval,
    _download,
    _load_model,
    _load_tokenizer,
    _mean,
    _sha256,
    _validate_resume,
    build_decisive_examples,
)


def _source_identity(config_path: str | Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    config = Path(config_path).resolve()
    files = [
        root / "qkvs" / "causal_reference.py",
        root / "qkvs" / "stage_b_oracle.py",
        root / "qkvs" / "utility_gate.py",
        root / "qkvs" / "runtime.py",
        root / "pyproject.toml",
        root / "uv.lock",
        config,
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relative_files = [str(path.relative_to(root)) for path in files]
    tracked_clean = all(
        subprocess.run(
            ["git", "diff", "--quiet", mode, "--", *relative_files],
            cwd=root,
            check=False,
        ).returncode
        == 0
        for mode in ("HEAD", "--cached")
    )
    return {
        "git_revision": revision,
        "tracked_files_clean": tracked_clean,
        "files": {str(path.relative_to(root)): _sha256(path) for path in files},
    }


def _score(
    model: Any,
    example: UtilityExample,
    masks: tuple[Tensor, ...],
    *,
    device: torch.device,
) -> dict[str, Any]:
    input_ids = torch.tensor(example.input_ids, dtype=torch.long, device=device)[None]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        hidden_states, _ = qwen_forward_with_layer_masks(model, input_ids, masks)
        logits = model.lm_head(hidden_states[:, -1])[0, list(example.label_token_ids)].float()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    log_probabilities = logits.log_softmax(dim=-1)
    prediction = int(logits.argmax().item())
    return {
        "loss": float(-log_probabilities[example.gold_label].item()),
        "prediction": prediction,
        "correct": prediction == example.gold_label,
        "label_log_probabilities": [float(value) for value in log_probabilities.tolist()],
        "elapsed_seconds": elapsed,
    }


def evaluate_example(
    model: Any,
    example: UtilityExample,
    *,
    device: torch.device,
    model_dtype: torch.dtype,
    rolling_tokens: int,
    retrieval_layer: int,
) -> dict[str, Any]:
    common = {
        "length": len(example.input_ids),
        "n_layers": model.config.num_hidden_layers,
        "rolling_tokens": rolling_tokens,
        "global_prefix_end": example.global_prefix_end,
        "dtype": model_dtype,
        "device": device,
    }
    selected_range = example.block_ranges[example.target_block]
    plans = {
        "rolling": rolling_mask_plan(**common),
        "causal_oracle": causal_oracle_mask_plan(
            **common,
            selected_range=selected_range,
            retrieval_layer=retrieval_layer,
        ),
        "stage_a_oracle": stage_a_oracle_mask_plan(
            **common,
            query_start=example.query_start,
            selected_range=selected_range,
        ),
    }
    modes = {
        name: _score(model, example, plan.masks, device=device) for name, plan in plans.items()
    }
    rolling_loss = modes["rolling"]["loss"]
    for mode in modes.values():
        mode["utility_nats"] = rolling_loss - mode["loss"]
    return {
        "index": example.index,
        "squad_id": example.squad_id,
        "gold_label": example.gold_label,
        "target_block": example.target_block,
        "sequence_tokens": len(example.input_ids),
        "query_start": example.query_start,
        "decision_position": len(example.input_ids) - 1,
        "retrieval_layer": retrieval_layer,
        "selected_range": list(selected_range),
        "modes": modes,
    }


def summarize(
    examples: list[dict[str, Any]],
    *,
    configured_examples: int,
    seed: int,
    minimum_accuracy: float,
    minimum_lower_bound: float,
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for name in ("rolling", "causal_oracle", "stage_a_oracle"):
        values = [example["modes"][name] for example in examples]
        utilities = [float(value["utility_nats"]) for value in values]
        modes[name] = {
            "accuracy": _mean([float(value["correct"]) for value in values]) if values else None,
            "mean_loss": _mean([float(value["loss"]) for value in values]) if values else None,
            "mean_utility_nats": _mean(utilities) if utilities else None,
            "utility_95pct_bootstrap": (
                _bootstrap_mean_interval(utilities, seed=seed + len(modes)) if utilities else None
            ),
        }
    complete = len(examples) == configured_examples
    comparisons: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    recovered_utility = None
    if complete:
        causal = [example["modes"]["causal_oracle"]["utility_nats"] for example in examples]
        stage_a = [example["modes"]["stage_a_oracle"]["utility_nats"] for example in examples]
        comparisons = {
            "causal_oracle_vs_rolling": {
                "mean_utility_nats": _mean(causal),
                "bootstrap_95pct": _bootstrap_mean_interval(causal, seed=seed + 101),
            },
            "stage_a_oracle_vs_rolling": {
                "mean_utility_nats": _mean(stage_a),
                "bootstrap_95pct": _bootstrap_mean_interval(stage_a, seed=seed + 102),
            },
        }
        denominator = _mean(stage_a)
        recovered_utility = _mean(causal) / denominator if denominator > 0 else None
        checks = {
            "causal_oracle_reader_accuracy": modes["causal_oracle"]["accuracy"] >= minimum_accuracy,
            "causal_oracle_beats_rolling": comparisons["causal_oracle_vs_rolling"][
                "bootstrap_95pct"
            ][0]
            >= minimum_lower_bound,
        }
    return {
        "complete": complete,
        "completed_examples": len(examples),
        "configured_examples": configured_examples,
        "modes": modes,
        "comparisons": comparisons,
        "causal_oracle_recovered_stage_a_utility": recovered_utility,
        "gate": {
            "minimum_causal_oracle_accuracy": minimum_accuracy,
            "minimum_causal_oracle_vs_rolling_lower_bound_nats": minimum_lower_bound,
            "checks": checks,
            "passed": complete and all(checks.values()),
            "decision": (
                "permit_s_training"
                if complete and all(checks.values())
                else "stop_before_s_training"
                if complete
                else "incomplete"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage B causal-oracle parity gate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--max-examples-this-run", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    specification = load_json(args.config)
    dataset = specification["datasets"]["evaluation"]
    oracle = specification["oracle_parity"]
    geometry = specification["geometry"]
    _download(dataset["url"], Path(dataset["cache_path"]), dataset["sha256"])
    tokenizer = _load_tokenizer(specification)
    with Path(dataset["cache_path"]).open() as source:
        payload = json.load(source)
    examples = build_decisive_examples(
        tokenizer,
        payload,
        count=oracle["examples"],
        seed=oracle["dataset_seed"],
        n_history_blocks=geometry["history_blocks"],
        candidate_history_blocks=len(geometry["candidate_positions"]),
        block_tokens=geometry["block_tokens"],
        rolling_tokens=geometry["rolling_tokens"],
        rolling_layers=geometry["rolling_layers"],
        target_seed=oracle["target_position_seed"],
        label_seed=oracle["answer_label_seed"],
        baseline_seed=oracle["random_baseline_seed"],
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepared_examples": len(examples),
                    "sequence_token_range": [
                        min(len(example.input_ids) for example in examples),
                        max(len(example.input_ids) for example in examples),
                    ],
                    "decision_position": "final_prompt_token",
                    "retrieval_layer": geometry["retrieval_layer"],
                    "dataset_sha256": _sha256(Path(dataset["cache_path"])),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    output = Path(args.output or oracle["output"])
    identity = _source_identity(args.config)
    if not identity["tracked_files_clean"]:
        raise ValueError("oracle-parity source files must be committed before measurement")
    expected = {
        "config": specification,
        "dataset_sha256": _sha256(Path(dataset["cache_path"])),
        "device": args.device,
        "source_identity": identity,
    }
    if output.exists():
        result = load_json(output)
        _validate_resume(result, expected)
    else:
        result = {
            "schema_version": 1,
            **expected,
            "environment": environment(),
            "examples": [],
        }
    model, model_dtype = _load_model(specification, torch.device(args.device))
    completed = {int(example["index"]) for example in result["examples"]}
    remaining = [example for example in examples if example.index not in completed]
    if args.max_examples_this_run is not None:
        remaining = remaining[: args.max_examples_this_run]
    for example in remaining:
        result["examples"].append(
            evaluate_example(
                model,
                example,
                device=torch.device(args.device),
                model_dtype=model_dtype,
                rolling_tokens=geometry["rolling_tokens"],
                retrieval_layer=geometry["retrieval_layer"],
            )
        )
        result["examples"].sort(key=lambda value: value["index"])
        result["summary"] = summarize(
            result["examples"],
            configured_examples=oracle["examples"],
            seed=oracle["dataset_seed"],
            minimum_accuracy=oracle["minimum_causal_oracle_accuracy"],
            minimum_lower_bound=oracle["minimum_causal_oracle_vs_rolling_lower_bound_nats"],
        )
        write_json(output, result)
        print(
            f"example={example.index + 1}/{oracle['examples']} "
            f"rolling={result['examples'][-1]['modes']['rolling']['correct']} "
            f"causal={result['examples'][-1]['modes']['causal_oracle']['correct']} "
            f"stage_a={result['examples'][-1]['modes']['stage_a_oracle']['correct']}"
        )
    if not remaining:
        result["summary"] = summarize(
            result["examples"],
            configured_examples=oracle["examples"],
            seed=oracle["dataset_seed"],
            minimum_accuracy=oracle["minimum_causal_oracle_accuracy"],
            minimum_lower_bound=oracle["minimum_causal_oracle_vs_rolling_lower_bound_nats"],
        )
        write_json(output, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
