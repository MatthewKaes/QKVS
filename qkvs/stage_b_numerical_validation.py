from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import Tensor, nn
from transformers.cache_utils import Cache, DynamicCache

from qkvs.causal_reference import (
    causal_oracle_mask_plan,
    qwen_forward_with_layer_masks,
    rolling_mask_plan,
)
from qkvs.runtime import environment, load_json, write_json
from qkvs.stage_b_numerical_diagnostic import (
    _backend_context,
    _batched_layer_zero_kv,
    _final_comparison,
    _forward_trace,
    _incremental_trace,
    _load_model,
    _mask_structure,
    _point_metrics,
    _prefix_summary,
    _sequence_projection_summary,
    _state_names,
    _state_point_summary,
    _token_hash,
    _trace_summary,
)
from qkvs.utility_gate import _download, _load_tokenizer, _sha256, build_decisive_examples

_OUTPUT = Path("artifacts/results/stage_b_numerical_validation_replacement_v1.json")
_DIAGNOSTIC_CONFIG = Path("configs/stage_b_numerical_diagnostic_v2.json")


def _source_identity(proposal_path: str | Path, proposal: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    proposal_file = Path(proposal_path).resolve()
    diagnostic_config = root / _DIAGNOSTIC_CONFIG
    diagnostic = load_json(diagnostic_config)
    base_config = root / diagnostic["base_config_path"]
    files = [
        root / "AGENTS.md",
        root / "README.md",
        root / "docs" / "stage_b_preregistration.md",
        root / "docs" / "stage_b_execution_decision.md",
        root / "qkvs" / "causal_reference.py",
        root / "qkvs" / "stage_b_numerical_diagnostic.py",
        root / "qkvs" / "stage_b_numerical_validation.py",
        root / "qkvs" / "utility_gate.py",
        root / "qkvs" / "runtime.py",
        root / "pyproject.toml",
        root / "uv.lock",
        proposal_file,
        diagnostic_config,
        base_config,
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
        "proposal_status": proposal["status"],
    }


def _validate_proposal(proposal: dict[str, Any]) -> None:
    expected_examples = [1, 2, 4, 6]
    if proposal["protocol"] != "stage_b_numerical_validation_replacement_v1":
        raise ValueError("unexpected numerical-validation protocol")
    if proposal["status"] != "approved_for_numerical_validation_only":
        raise ValueError("replacement numerical validation is not authorized")
    if proposal["stage_b_status"] != "blocked":
        raise ValueError("Stage B must remain blocked during numerical validation")
    if proposal["validation_examples"]["indices"] != expected_examples:
        raise ValueError("validation example indices differ from the approved set")
    if proposal["conditional_gate_0_amendment"]["authorized_by_this_proposal"]:
        raise ValueError("replacement validation must not authorize Gate 0")


def _validate_preserved_artifacts(proposal: dict[str, Any]) -> dict[str, Any]:
    diagnostic = load_json(_DIAGNOSTIC_CONFIG)
    original = diagnostic["original_failed_gate"]
    calibration = proposal["calibration_evidence"]
    records = {
        "original_failed_audit": {
            "path": original["path"],
            "expected_sha256": original["sha256"],
            "source_revision": original["source_revision"],
        },
        "calibration_diagnostic": {
            "path": calibration["path"],
            "expected_sha256": calibration["sha256"],
            "source_revision": calibration["source_revision"],
        },
    }
    for record in records.values():
        actual = _sha256(Path(record["path"]))
        if actual != record["expected_sha256"]:
            raise ValueError(f"preserved artifact changed: {record['path']}")
        record["actual_sha256"] = actual
        artifact = load_json(record["path"])
        if artifact["source_identity"]["git_revision"] != record["source_revision"]:
            raise ValueError(f"preserved artifact source changed: {record['path']}")
    return records


def _execution_conditions(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostic = load_json(_DIAGNOSTIC_CONFIG)
    available = {condition["name"]: condition for condition in diagnostic["conditions"]}
    names = [
        proposal["bf16_task_sensitivity"]["condition"],
        *proposal["fp32_mechanical_validation"]["conditions"],
    ]
    if len(names) != len(set(names)):
        raise ValueError("replacement execution conditions are not unique")
    try:
        return [available[name] for name in names]
    except KeyError as error:
        raise ValueError(f"missing frozen execution condition: {error.args[0]}") from error


def _prepare_examples(
    proposal: dict[str, Any], specification: dict[str, Any]
) -> tuple[list[Any], list[dict[str, Any]]]:
    dataset = specification["datasets"]["evaluation"]
    dataset_path = Path(dataset["cache_path"])
    _download(dataset["url"], dataset_path, dataset["sha256"])
    tokenizer = _load_tokenizer(specification)
    with dataset_path.open() as source:
        payload = json.load(source)
    geometry = specification["geometry"]
    oracle = specification["oracle_parity"]
    all_examples = build_decisive_examples(
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
    indices = proposal["validation_examples"]["indices"]
    examples = [all_examples[index] for index in indices]
    labels = [example.gold_label for example in examples]
    targets = [example.target_block for example in examples]
    if labels != proposal["validation_examples"]["gold_labels"]:
        raise ValueError("prepared gold labels differ from the frozen proposal")
    if targets != proposal["validation_examples"]["target_positions"]:
        raise ValueError("prepared target positions differ from the frozen proposal")
    preparations = []
    for example in examples:
        selected_range = example.block_ranges[example.target_block]
        preparations.append(
            {
                "example_index": example.index,
                "squad_id": example.squad_id,
                "sequence_tokens": len(example.input_ids),
                "input_token_sha256": _token_hash(example.input_ids),
                "gold_label": example.gold_label,
                "target_position": example.target_block,
                "selected_range": list(selected_range),
                "global_prefix_end": example.global_prefix_end,
                "query_start": example.query_start,
                "decision_position": len(example.input_ids) - 1,
                "label_token_ids": list(example.label_token_ids),
            }
        )
    return examples, preparations


def _future_cutoffs(proposal: dict[str, Any], example: Any) -> dict[str, int]:
    selected_range = example.block_ranges[example.target_block]
    values = {
        "global_prefix_end_minus_one": example.global_prefix_end - 1,
        "selected_block_end_minus_one": selected_range[1] - 1,
        "history_midpoint": example.query_start // 2,
        "query_start_minus_one": example.query_start - 1,
    }
    names = proposal["causal_semantics"]["future_counterfactual_cutoffs"]
    if set(names) != set(values):
        raise ValueError("future-counterfactual names differ from the implementation")
    return {name: values[name] for name in names}


def _incremental_hidden_cache(
    model: nn.Module,
    input_ids: Tensor,
    layer_masks: tuple[Tensor, ...],
) -> tuple[Tensor, Cache, dict[str, Any]]:
    cache = DynamicCache(config=model.config)
    outputs: list[Tensor] = []
    supplied_positions: list[int] = []
    first_length_mismatch = None
    for position in range(input_ids.shape[1]):
        supplied_positions.append(position)
        token_masks = tuple(
            mask[:, :, position : position + 1, : position + 1] for mask in layer_masks
        )
        hidden_states, updated_cache = qwen_forward_with_layer_masks(
            model,
            input_ids[:, position : position + 1],
            token_masks,
            past_key_values=cache,
            cache_position=torch.tensor([position], device=input_ids.device),
            use_cache=True,
        )
        if updated_cache is None:
            raise RuntimeError("incremental execution did not return a cache")
        cache = updated_cache
        outputs.append(hidden_states)
        if first_length_mismatch is None and cache.get_seq_length() != position + 1:
            first_length_mismatch = {
                "position": position,
                "expected": position + 1,
                "actual": cache.get_seq_length(),
            }
    layer_lengths = [cache.get_seq_length(index) for index in range(len(cache.layers))]
    return (
        torch.cat(outputs, dim=1),
        cache,
        {
            "first_length_mismatch": first_length_mismatch,
            "final_layer_lengths": layer_lengths,
            "all_layers_have_full_length": all(
                length == input_ids.shape[1] for length in layer_lengths
            ),
            "supplied_cache_positions": {
                "count": len(supplied_positions),
                "first": supplied_positions[0] if supplied_positions else None,
                "last": supplied_positions[-1] if supplied_positions else None,
                "contiguous_zero_based": supplied_positions == list(range(input_ids.shape[1])),
            },
        },
    )


def _cache_structure_passed(summary: dict[str, Any]) -> bool:
    positions = summary["supplied_cache_positions"]
    return (
        summary["first_length_mismatch"] is None
        and summary["all_layers_have_full_length"]
        and positions["contiguous_zero_based"]
        and positions["count"] == summary["final_layer_lengths"][0]
    )


def _projection_samples(example: Any) -> list[int]:
    selected_range = example.block_ranges[example.target_block]
    return sorted(
        {
            0,
            example.global_prefix_end - 1,
            selected_range[0],
            selected_range[1] - 1,
            example.query_start - 1,
            len(example.input_ids) - 1,
        }
    )


def _layer_zero_kv_comparison(
    model: nn.Module,
    input_ids: Tensor,
    cache: Cache,
    *,
    example: Any,
) -> tuple[dict[str, Any], Tensor, Tensor]:
    batched_key, batched_value = _batched_layer_zero_kv(model, input_ids)
    cached_key = cache.layers[0].keys
    cached_value = cache.layers[0].values
    sample_errors = {}
    for position in _projection_samples(example):
        sample_errors[str(position)] = {
            "key_max_absolute_error": float(
                (batched_key[:, :, position].float() - cached_key[:, :, position].float())
                .abs()
                .max()
                .item()
            ),
            "value_max_absolute_error": float(
                (batched_value[:, :, position].float() - cached_value[:, :, position].float())
                .abs()
                .max()
                .item()
            ),
        }
    expected_key_shape = [
        input_ids.shape[0],
        model.config.num_key_value_heads,
        input_ids.shape[1],
        model.config.hidden_size // model.config.num_attention_heads,
    ]
    expected_value_shape = expected_key_shape
    structure_checks = {
        "batched_key_shape_expected": list(batched_key.shape) == expected_key_shape,
        "cached_key_shape_expected": list(cached_key.shape) == expected_key_shape,
        "batched_value_shape_expected": list(batched_value.shape) == expected_value_shape,
        "cached_value_shape_expected": list(cached_value.shape) == expected_value_shape,
        "all_finite": bool(
            torch.isfinite(batched_key).all()
            and torch.isfinite(cached_key).all()
            and torch.isfinite(batched_value).all()
            and torch.isfinite(cached_value).all()
        ),
    }
    result = {
        "projection_semantics": {
            "input": "token_embeddings_after_layer_0_input_layernorm",
            "key": "k_proj_then_qwen_rotary_position_embedding",
            "value": "v_proj_without_rotary_position_embedding",
            "batched_positions": "zero_based_contiguous_arange",
            "cached_positions": "explicit_zero_based_position_per_incremental_step",
        },
        "expected_key_shape": expected_key_shape,
        "expected_value_shape": expected_value_shape,
        "structure_checks": structure_checks,
        "structure_passed": all(structure_checks.values()),
        "key": _sequence_projection_summary(batched_key, cached_key, []),
        "value": _sequence_projection_summary(batched_value, cached_value, []),
        "sampled_aligned_token_errors": sample_errors,
        "numerical_acceptance": "report_only_no_authorized_tolerance",
    }
    return result, cached_key.detach().cpu(), cached_value.detach().cpu()


def _cache_repeat_summary(left: Cache, right: Cache) -> dict[str, Any]:
    if len(left.layers) != len(right.layers):
        return {"layer_count_match": False, "exact_equal": False}
    layers = []
    for index, (left_layer, right_layer) in enumerate(zip(left.layers, right.layers, strict=True)):
        key = _point_metrics(left_layer.keys, right_layer.keys)
        value = _point_metrics(left_layer.values, right_layer.values)
        layers.append({"layer": index, "key": key, "value": value})
    return {
        "layer_count_match": True,
        "exact_equal": all(
            layer[projection]["exact_equal"] for layer in layers for projection in ("key", "value")
        ),
        "maximum_absolute_error": max(
            layer[projection]["max_absolute_error"]
            for layer in layers
            for projection in ("key", "value")
        ),
        "layers": layers,
    }


def _trace_maximum(summary: dict[str, Any]) -> float:
    return max(state["max_absolute_error"] for state in summary["states"].values())


def _utility_difference(modes: dict[str, Any]) -> float:
    batch_utility = (
        modes["rolling"]["four_label_answer"]["batched"]["loss"]
        - modes["causal_oracle"]["four_label_answer"]["batched"]["loss"]
    )
    incremental_utility = (
        modes["rolling"]["four_label_answer"]["incremental"]["loss"]
        - modes["causal_oracle"]["four_label_answer"]["incremental"]["loss"]
    )
    return incremental_utility - batch_utility


def _fp32_checks(
    proposal: dict[str, Any],
    *,
    trace_summary: dict[str, Any],
    prefix_replays: dict[str, Any],
    modes: dict[str, Any],
) -> dict[str, bool]:
    thresholds = proposal["fp32_mechanical_validation"]["thresholds"]
    checks = {
        "rolling_all_token_layer_hidden": _trace_maximum(trace_summary)
        <= thresholds["maximum_all_token_layer_hidden_error"],
        "uncached_prefix_vs_cached_hidden": max(
            replay["max_absolute_error"] for replay in prefix_replays.values()
        )
        <= thresholds["maximum_uncached_prefix_vs_cached_hidden_error"],
        "utility_path_error": abs(_utility_difference(modes))
        <= thresholds["maximum_utility_path_error_nats"],
    }
    for mode_name, mode in modes.items():
        hidden = mode["hidden"]
        logits = mode["full_vocabulary_logits"]
        answer = mode["four_label_answer"]
        prefix = mode_name.replace("causal_oracle", "oracle")
        checks.update(
            {
                f"{prefix}_final_hidden_error": hidden["max_absolute_error"]
                <= thresholds["maximum_final_hidden_error"],
                f"{prefix}_final_hidden_cosine": hidden["cosine"]
                >= thresholds["minimum_final_hidden_cosine"],
                f"{prefix}_full_vocabulary_logit_error": logits["max_absolute_error"]
                <= thresholds["maximum_full_vocabulary_logit_error"],
                f"{prefix}_four_label_loss_error": abs(answer["incremental_minus_batched_loss"])
                <= thresholds["maximum_four_label_loss_error_nats"],
                f"{prefix}_label_log_probability_error": answer["max_label_log_probability_error"]
                <= thresholds["maximum_label_log_probability_error"],
                f"{prefix}_four_label_total_variation": answer["total_variation_distance"]
                <= thresholds["maximum_four_label_total_variation"],
                f"{prefix}_four_label_prediction_match": answer["prediction_match"]
                if thresholds["require_four_label_prediction_match"]
                else True,
                f"{prefix}_full_vocabulary_top1_match": logits["top1_match"]
                if thresholds["require_full_vocabulary_top1_match"]
                else True,
                f"{prefix}_all_finite": hidden["all_finite"]
                if thresholds["require_all_finite"]
                else True,
            }
        )
    return checks


def _backend_agreement(proposal: dict[str, Any], example_result: dict[str, Any]) -> dict[str, Any]:
    conditions = example_result["conditions"]
    if not {"fp32_sdpa_math", "fp32_eager"}.issubset(conditions):
        return {"complete": False, "passed": None}
    left = conditions["fp32_sdpa_math"]["modes"]
    right = conditions["fp32_eager"]["modes"]
    loss_differences = {}
    for mode in ("rolling", "causal_oracle"):
        for path in ("batched", "incremental"):
            loss_differences[f"{mode}.{path}"] = abs(
                left[mode]["four_label_answer"][path]["loss"]
                - right[mode]["four_label_answer"][path]["loss"]
            )
    utility_differences = {}
    for path in ("batched", "incremental"):
        left_utility = (
            left["rolling"]["four_label_answer"][path]["loss"]
            - left["causal_oracle"]["four_label_answer"][path]["loss"]
        )
        right_utility = (
            right["rolling"]["four_label_answer"][path]["loss"]
            - right["causal_oracle"]["four_label_answer"][path]["loss"]
        )
        utility_differences[path] = abs(left_utility - right_utility)
    limits = proposal["fp32_mechanical_validation"]["backend_agreement"]
    checks = {
        "four_label_loss": max(loss_differences.values())
        <= limits["maximum_four_label_loss_difference_nats"],
        "utility": max(utility_differences.values()) <= limits["maximum_utility_difference_nats"],
    }
    return {
        "complete": True,
        "loss_differences_nats": loss_differences,
        "utility_differences_nats": utility_differences,
        "thresholds": limits,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _run_condition(
    proposal: dict[str, Any],
    specification: dict[str, Any],
    condition: dict[str, Any],
    example: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
    name = condition["name"]
    print(f"example={example.index} condition={name} phase=load", flush=True)
    model = _load_model(specification, condition, device)
    names = _state_names(model)
    input_ids = torch.tensor(example.input_ids, dtype=torch.long, device=device)[None]
    geometry = specification["geometry"]
    selected_range = example.block_ranges[example.target_block]
    common = {
        "length": len(example.input_ids),
        "n_layers": model.config.num_hidden_layers,
        "rolling_tokens": geometry["rolling_tokens"],
        "global_prefix_end": example.global_prefix_end,
        "dtype": model.dtype,
        "device": device,
    }
    rolling_plan = rolling_mask_plan(**common)
    oracle_plan = causal_oracle_mask_plan(
        **common,
        selected_range=selected_range,
        retrieval_layer=geometry["retrieval_layer"],
    )
    mask_structure = _mask_structure(
        rolling_plan.masks,
        oracle_plan.masks,
        retrieval_layer=geometry["retrieval_layer"],
        selected_range=selected_range,
    )
    is_fp32 = name in proposal["fp32_mechanical_validation"]["conditions"]
    is_bf16 = name == proposal["bf16_task_sensitivity"]["condition"]
    has_causal_checks = name in proposal["causal_semantics"]["conditions"]
    cutoffs = _future_cutoffs(proposal, example)
    probe_positions = sorted({*cutoffs.values(), len(example.input_ids) - 1})
    timings: dict[str, float] = {}

    with _backend_context(condition), torch.inference_mode():
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        rolling_batched, _, rolling_batched_trace = _forward_trace(
            model,
            input_ids,
            rolling_plan.masks,
            trace_device=torch.device("cpu"),
        )
        torch.cuda.synchronize(device)
        timings["rolling_batched_seconds"] = time.perf_counter() - started
        print(f"example={example.index} condition={name} phase=rolling_batch", flush=True)

        started = time.perf_counter()
        if is_fp32:
            rolling_incremental, rolling_cache, rolling_incremental_trace, rolling_cache_summary = (
                _incremental_trace(model, input_ids, rolling_plan.masks)
            )
            rolling_incremental_trace = rolling_incremental_trace.cpu()
        else:
            rolling_incremental, rolling_cache, rolling_cache_summary = _incremental_hidden_cache(
                model, input_ids, rolling_plan.masks
            )
            rolling_incremental_trace = None
        torch.cuda.synchronize(device)
        timings["rolling_incremental_seconds"] = time.perf_counter() - started
        rolling_layer_zero, rolling_key, rolling_value = _layer_zero_kv_comparison(
            model, input_ids, rolling_cache, example=example
        )
        print(f"example={example.index} condition={name} phase=rolling_incremental", flush=True)

        future_counterfactuals = {}
        if has_causal_checks:
            for cutoff_name, cutoff in cutoffs.items():
                changed_ids = input_ids.clone()
                suffix = changed_ids[:, cutoff + 1 :]
                changed_ids[:, cutoff + 1 :] = (suffix + 1) % model.config.vocab_size
                if torch.equal(changed_ids[:, cutoff + 1 :], input_ids[:, cutoff + 1 :]):
                    raise RuntimeError("future counterfactual did not change the suffix")
                _, _, changed_trace = _forward_trace(
                    model,
                    changed_ids,
                    rolling_plan.masks,
                    trace_device=torch.device("cpu"),
                )
                future_counterfactuals[cutoff_name] = {
                    "cutoff": cutoff,
                    **_prefix_summary(
                        rolling_batched_trace,
                        changed_trace,
                        end_exclusive=cutoff + 1,
                        names=names,
                    ),
                }
                del changed_ids, changed_trace
            print(f"example={example.index} condition={name} phase=future_checks", flush=True)

        trace_summary = None
        prefix_replays = {}
        if is_fp32:
            if rolling_incremental_trace is None:
                raise RuntimeError("FP32 validation requires the incremental layer trace")
            trace_summary = _trace_summary(
                rolling_batched_trace,
                rolling_incremental_trace,
                names=names,
                thresholds=[
                    proposal["fp32_mechanical_validation"]["thresholds"][
                        "maximum_all_token_layer_hidden_error"
                    ]
                ],
                probe_positions=probe_positions,
                top_count=16,
            )
            for probe in probe_positions:
                prefix_masks = tuple(
                    mask[:, :, : probe + 1, : probe + 1] for mask in rolling_plan.masks
                )
                _, _, prefix_trace = _forward_trace(
                    model,
                    input_ids[:, : probe + 1],
                    prefix_masks,
                    trace_device=torch.device("cpu"),
                )
                prefix_replays[str(probe)] = _state_point_summary(
                    prefix_trace[:, 0, -1],
                    rolling_incremental_trace[:, 0, probe],
                    names,
                )
                del prefix_trace
            print(f"example={example.index} condition={name} phase=fp32_trace", flush=True)

        repeatability = None
        if is_bf16:
            repeat_hidden, repeat_cache, repeat_cache_summary = _incremental_hidden_cache(
                model, input_ids, rolling_plan.masks
            )
            repeatability = {
                "rolling": {
                    "hidden": _point_metrics(rolling_incremental, repeat_hidden),
                    "cache": _cache_repeat_summary(rolling_cache, repeat_cache),
                    "cache_structure": repeat_cache_summary,
                }
            }
            del repeat_hidden, repeat_cache

        rolling_final = _final_comparison(
            model,
            rolling_batched,
            rolling_incremental,
            label_token_ids=example.label_token_ids,
            gold_label=example.gold_label,
        )

        torch.cuda.synchronize(device)
        started = time.perf_counter()
        oracle_batched, _ = qwen_forward_with_layer_masks(model, input_ids, oracle_plan.masks)
        torch.cuda.synchronize(device)
        timings["causal_oracle_batched_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        oracle_incremental, oracle_cache, oracle_cache_summary = _incremental_hidden_cache(
            model, input_ids, oracle_plan.masks
        )
        torch.cuda.synchronize(device)
        timings["causal_oracle_incremental_seconds"] = time.perf_counter() - started
        oracle_layer_zero, oracle_key, oracle_value = _layer_zero_kv_comparison(
            model, input_ids, oracle_cache, example=example
        )
        print(f"example={example.index} condition={name} phase=oracle_paths", flush=True)

        if is_bf16:
            oracle_repeat_hidden, oracle_repeat_cache, oracle_repeat_cache_summary = (
                _incremental_hidden_cache(model, input_ids, oracle_plan.masks)
            )
            if repeatability is None:
                raise RuntimeError("BF16 repeatability result was not initialized")
            repeatability["causal_oracle"] = {
                "hidden": _point_metrics(oracle_incremental, oracle_repeat_hidden),
                "cache": _cache_repeat_summary(oracle_cache, oracle_repeat_cache),
                "cache_structure": oracle_repeat_cache_summary,
            }
            del oracle_repeat_hidden, oracle_repeat_cache

        oracle_final = _final_comparison(
            model,
            oracle_batched,
            oracle_incremental,
            label_token_ids=example.label_token_ids,
            gold_label=example.gold_label,
        )

    modes = {"rolling": rolling_final, "causal_oracle": oracle_final}
    layer_zero = {
        "rolling_incremental": rolling_layer_zero,
        "causal_oracle_incremental": oracle_layer_zero,
        "incremental_mode_agreement": {
            "key": _sequence_projection_summary(rolling_key, oracle_key, []),
            "value": _sequence_projection_summary(rolling_value, oracle_value, []),
            "numerical_acceptance": "report_only_no_authorized_tolerance",
        },
    }
    causal_checks = {
        "mask_structure": mask_structure["passed"],
        "rolling_cache_order_positions_and_lengths": _cache_structure_passed(rolling_cache_summary),
        "oracle_cache_order_positions_and_lengths": _cache_structure_passed(oracle_cache_summary),
        "future_prefixes_exact": all(
            result["exact_equal"] for result in future_counterfactuals.values()
        )
        if has_causal_checks
        else True,
    }
    layer_zero_checks = {
        "rolling_structure_and_projection_shapes": rolling_layer_zero["structure_passed"],
        "oracle_structure_and_projection_shapes": oracle_layer_zero["structure_passed"],
        "mode_shapes_match": layer_zero["incremental_mode_agreement"]["key"]["shape"]
        == layer_zero["incremental_mode_agreement"]["value"]["shape"],
    }
    checks = {**causal_checks, **layer_zero_checks}
    if is_fp32:
        if trace_summary is None:
            raise RuntimeError("FP32 trace summary was not initialized")
        checks.update(
            _fp32_checks(
                proposal,
                trace_summary=trace_summary,
                prefix_replays=prefix_replays,
                modes=modes,
            )
        )
    if is_bf16:
        if repeatability is None:
            raise RuntimeError("BF16 repeatability result was not initialized")
        require_repeat = proposal["bf16_task_sensitivity"]["require_incremental_repeat_exact"]
        require_prediction = proposal["bf16_task_sensitivity"][
            "require_batch_incremental_four_label_prediction_match"
        ]
        for mode_name in ("rolling", "causal_oracle"):
            checks[f"{mode_name}_incremental_hidden_repeat_exact"] = (
                repeatability[mode_name]["hidden"]["exact_equal"] if require_repeat else True
            )
            checks[f"{mode_name}_incremental_cache_repeat_exact"] = (
                repeatability[mode_name]["cache"]["exact_equal"] if require_repeat else True
            )
            checks[f"{mode_name}_four_label_prediction_match"] = (
                modes[mode_name]["four_label_answer"]["prediction_match"]
                if require_prediction
                else True
            )

    result = {
        "condition": condition,
        "example_index": example.index,
        "sequence_tokens": len(example.input_ids),
        "input_token_sha256": _token_hash(example.input_ids),
        "model_attention_implementation": model.config._attn_implementation,
        "mask_structure": mask_structure,
        "cache_structure": {
            "rolling": rolling_cache_summary,
            "causal_oracle": oracle_cache_summary,
        },
        "layer_zero_batched_vs_cached_kv": layer_zero,
        "future_counterfactuals": future_counterfactuals,
        "rolling_all_token_layer_trace": trace_summary,
        "uncached_prefix_replays": prefix_replays,
        "incremental_repeatability": repeatability,
        "modes": modes,
        "utility_path_error_nats": _utility_difference(modes),
        "timings": timings,
        "checks": checks,
        "passed": all(checks.values()),
    }
    del (
        model,
        input_ids,
        rolling_plan,
        oracle_plan,
        rolling_batched,
        rolling_incremental,
        rolling_batched_trace,
        rolling_cache,
        oracle_batched,
        oracle_incremental,
        oracle_cache,
        rolling_key,
        rolling_value,
        oracle_key,
        oracle_value,
    )
    if rolling_incremental_trace is not None:
        del rolling_incremental_trace
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _refresh_decision(
    result: dict[str, Any],
    *,
    examples: list[Any],
    conditions: list[dict[str, Any]],
) -> None:
    required_units = {
        (str(example.index), condition["name"]) for example in examples for condition in conditions
    }
    completed_units = {
        (example_index, condition_name)
        for example_index, example_result in result["examples"].items()
        for condition_name in example_result["conditions"]
    }
    condition_failures = [
        {"example_index": int(example_index), "condition": condition_name}
        for example_index, example_result in result["examples"].items()
        for condition_name, condition in example_result["conditions"].items()
        if not condition["passed"]
    ]
    backend_failures = []
    for example_index, example_result in result["examples"].items():
        agreement = _backend_agreement(result["proposal"], example_result)
        example_result["fp32_backend_agreement"] = agreement
        if agreement["complete"] and not agreement["passed"]:
            backend_failures.append(int(example_index))
    complete = completed_units == required_units
    passed = complete and not condition_failures and not backend_failures
    if condition_failures or backend_failures:
        outcome = "failed_stop"
    elif complete:
        outcome = "passed_return_for_independent_arbitration"
    else:
        outcome = "incomplete"
    result["decision"] = {
        "complete": complete,
        "passed": passed if complete or outcome == "failed_stop" else None,
        "outcome": outcome,
        "condition_failures": condition_failures,
        "backend_agreement_failures": backend_failures,
        "stage_b_status": "blocked",
        "gate_0_authorized": False,
    }


def _validate_resume(result: dict[str, Any], expected: dict[str, Any]) -> None:
    actual = {key: result.get(key) for key in expected}
    if actual != expected:
        raise ValueError("existing validation output does not match source or protocol")
    if result["decision"]["outcome"] == "failed_stop":
        raise ValueError("validation already reached a frozen stop condition")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run approved Stage B numerical validation")
    parser.add_argument(
        "--proposal",
        default="configs/stage_b_numerical_validation_replacement_proposal.json",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output")
    parser.add_argument("--max-conditions-this-run", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    proposal = load_json(args.proposal)
    _validate_proposal(proposal)
    diagnostic = load_json(_DIAGNOSTIC_CONFIG)
    specification = load_json(diagnostic["base_config_path"])
    preserved_artifacts = _validate_preserved_artifacts(proposal)
    conditions = _execution_conditions(proposal)
    examples, preparations = _prepare_examples(proposal, specification)
    dataset_path = Path(specification["datasets"]["evaluation"]["cache_path"])
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "protocol": proposal["protocol"],
                    "status": proposal["status"],
                    "conditions": conditions,
                    "examples": preparations,
                    "dataset_sha256": _sha256(dataset_path),
                    "preserved_artifacts": preserved_artifacts,
                    "gate_0_authorized": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    identity = _source_identity(args.proposal, proposal)
    if not identity["tracked_files_clean"]:
        raise ValueError("numerical-validation source files must be committed before measurement")
    if not torch.cuda.is_available():
        raise RuntimeError("approved numerical validation requires CUDA")
    expected = {
        "proposal": proposal,
        "base_config": specification,
        "source_identity": identity,
        "preserved_artifacts": preserved_artifacts,
        "dataset_sha256": _sha256(dataset_path),
        "device": args.device,
    }
    output = Path(args.output) if args.output else _OUTPUT
    if output.exists():
        result = load_json(output)
        _validate_resume(result, expected)
    else:
        result = {
            "schema_version": 1,
            "test": proposal["protocol"],
            **expected,
            "environment": {
                **environment(),
                "transformers": transformers.__version__,
                "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
                "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
                "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
            },
            "preparations": preparations,
            "examples": {
                str(example.index): {"preparation": preparation, "conditions": {}}
                for example, preparation in zip(examples, preparations, strict=True)
            },
            "decision": {
                "complete": False,
                "passed": None,
                "outcome": "incomplete",
                "stage_b_status": "blocked",
                "gate_0_authorized": False,
            },
        }
        write_json(output, result)

    remaining = [
        (example, condition)
        for example in examples
        for condition in conditions
        if condition["name"] not in result["examples"][str(example.index)]["conditions"]
    ][: args.max_conditions_this_run]
    for example, condition in remaining:
        measured = _run_condition(
            proposal,
            specification,
            condition,
            example,
            device=torch.device(args.device),
        )
        result["examples"][str(example.index)]["conditions"][condition["name"]] = measured
        _refresh_decision(result, examples=examples, conditions=conditions)
        write_json(output, result)
        print(
            json.dumps(
                {
                    "completed_example": example.index,
                    "completed_condition": condition["name"],
                    "condition_passed": measured["passed"],
                    "decision": result["decision"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        if result["decision"]["outcome"] == "failed_stop":
            raise SystemExit(1)

    if not remaining:
        _refresh_decision(result, examples=examples, conditions=conditions)
        write_json(output, result)
        print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
