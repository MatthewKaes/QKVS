from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from qkvs.causal_reference import (
    causal_oracle_mask_plan,
    qwen_forward_with_layer_masks,
    qwen_incremental_with_layer_masks,
    rolling_mask_plan,
)
from qkvs.runtime import environment, load_json, write_json
from qkvs.utility_gate import (
    _download,
    _load_tokenizer,
    _sha256,
    build_decisive_examples,
)


def _source_identity(diagnostic_path: str | Path, diagnostic: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    diagnostic_config = Path(diagnostic_path).resolve()
    base_config = root / diagnostic["base_config_path"]
    files = [
        root / "AGENTS.md",
        root / "README.md",
        root / "docs" / "stage_b_preregistration.md",
        root / "docs" / "stage_b_execution_decision.md",
        root / "qkvs" / "causal_reference.py",
        root / "qkvs" / "stage_b_numerical_diagnostic.py",
        root / "qkvs" / "utility_gate.py",
        root / "qkvs" / "runtime.py",
        root / "pyproject.toml",
        root / "uv.lock",
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
    }


def _token_hash(input_ids: tuple[int, ...]) -> str:
    return hashlib.sha256(b"".join(token.to_bytes(4, "little") for token in input_ids)).hexdigest()


def _state_names(model: nn.Module) -> list[str]:
    return [
        "embedding",
        *(f"layer_{index:02d}_output" for index in range(model.config.num_hidden_layers)),
        "final_norm",
    ]


class _Progress:
    def __init__(
        self,
        condition: str,
        *,
        condition_index: int,
        condition_count: int,
        step_count: int,
    ) -> None:
        self.condition = condition
        self.condition_index = condition_index
        self.condition_count = condition_count
        self.step_count = step_count
        self.step = 0
        self.started = time.perf_counter()

    def report(self, phase: str, *, advance: bool = False) -> None:
        if advance:
            self.step += 1
        width = 24
        filled = width * self.step // self.step_count
        bar = "#" * filled + "-" * (width - filled)
        elapsed = int(time.perf_counter() - self.started)
        print(
            f"[condition {self.condition_index}/{self.condition_count}] "
            f"[{bar}] {self.step}/{self.step_count} {self.condition}: {phase} "
            f"({elapsed // 60:02d}:{elapsed % 60:02d})",
            flush=True,
        )


def _forward_trace(
    model: nn.Module,
    input_ids: Tensor,
    layer_masks: tuple[Tensor, ...],
    *,
    past_key_values: Cache | None = None,
    cache_position: Tensor | None = None,
    use_cache: bool = False,
    trace_device: torch.device | None = None,
) -> tuple[Tensor, Cache | None, Tensor]:
    decoder = model.model
    if len(layer_masks) != decoder.config.num_hidden_layers:
        raise ValueError("one attention mask is required per decoder layer")
    hidden_states = decoder.embed_tokens(input_ids)
    if trace_device is None:
        trace_device = hidden_states.device
    trace = torch.empty(
        (
            decoder.config.num_hidden_layers + 2,
            input_ids.shape[0],
            input_ids.shape[1],
            decoder.config.hidden_size,
        ),
        dtype=hidden_states.dtype,
        device=trace_device,
    )
    trace[0].copy_(hidden_states)
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=decoder.config)
    if cache_position is None:
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_length,
            past_length + input_ids.shape[1],
            device=input_ids.device,
        )
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = decoder.rotary_emb(hidden_states, position_ids)
    for layer_index, (layer, attention_mask) in enumerate(
        zip(decoder.layers, layer_masks, strict=True),
        start=1,
    ):
        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        trace[layer_index].copy_(hidden_states)
    hidden_states = decoder.norm(hidden_states)
    trace[-1].copy_(hidden_states)
    return hidden_states, past_key_values if use_cache else None, trace


def _incremental_trace(
    model: nn.Module,
    input_ids: Tensor,
    layer_masks: tuple[Tensor, ...],
) -> tuple[Tensor, Cache, Tensor, dict[str, Any]]:
    cache = DynamicCache(config=model.config)
    state_count = model.config.num_hidden_layers + 2
    trace = torch.empty(
        (state_count, input_ids.shape[0], input_ids.shape[1], model.config.hidden_size),
        dtype=model.dtype,
        device=input_ids.device,
    )
    first_length_mismatch = None
    for position in range(input_ids.shape[1]):
        token_masks = tuple(
            mask[:, :, position : position + 1, : position + 1] for mask in layer_masks
        )
        hidden_states, updated_cache, token_trace = _forward_trace(
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
        trace[:, :, position : position + 1].copy_(token_trace)
        if first_length_mismatch is None and cache.get_seq_length() != position + 1:
            first_length_mismatch = {
                "position": position,
                "expected": position + 1,
                "actual": cache.get_seq_length(),
            }
    layer_lengths = [cache.get_seq_length(index) for index in range(len(cache.layers))]
    final_hidden_states = trace[-1].clone()
    return (
        final_hidden_states,
        cache,
        trace,
        {
            "first_length_mismatch": first_length_mismatch,
            "final_layer_lengths": layer_lengths,
            "all_layers_have_full_length": all(
                length == input_ids.shape[1] for length in layer_lengths
            ),
        },
    )


def _backend_context(condition: dict[str, Any]) -> Any:
    if condition["sdpa_backend"] == "math":
        return sdpa_kernel(SDPBackend.MATH)
    return nullcontext()


def _load_model(
    specification: dict[str, Any],
    condition: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[condition["dtype"]]
    model_specification = specification["model"]
    model = AutoModelForCausalLM.from_pretrained(
        model_specification["id"],
        revision=model_specification["revision"],
        trust_remote_code=False,
        local_files_only=True,
        dtype=dtype,
        attn_implementation=condition["attention_implementation"],
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _point_metrics(left: Tensor, right: Tensor) -> dict[str, Any]:
    left_float = left.float().reshape(-1)
    right_float = right.float().reshape(-1)
    error = (left_float - right_float).abs()
    return {
        "max_absolute_error": float(error.max().item()),
        "mean_absolute_error": float(error.mean().item()),
        "cosine": float(F.cosine_similarity(left_float[None], right_float[None]).item()),
        "exact_equal": bool(torch.equal(left, right)),
        "all_finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
    }


def _trace_summary(
    batched: Tensor,
    incremental: Tensor,
    *,
    names: list[str],
    thresholds: list[float],
    probe_positions: list[int],
    top_count: int,
) -> dict[str, Any]:
    if batched.shape != incremental.shape:
        raise ValueError("trace shapes do not match")
    sequence_tokens = batched.shape[2]
    hidden_size = batched.shape[3]
    token_maxima: list[Tensor] = []
    states: dict[str, Any] = {}
    for state_index, name in enumerate(names):
        left = batched[state_index, 0].float()
        right = incremental[state_index, 0].float()
        error = (left - right).abs()
        per_token_max = error.amax(dim=-1)
        token_maxima.append(per_token_max.cpu())
        flat_index = int(error.argmax().item())
        max_token, max_feature = divmod(flat_index, hidden_size)
        earliest: dict[str, Any] = {}
        for threshold in thresholds:
            crossing = torch.nonzero(per_token_max > threshold, as_tuple=False)
            earliest[str(threshold)] = int(crossing[0].item()) if crossing.numel() else None
        states[name] = {
            "max_absolute_error": float(error.max().item()),
            "mean_absolute_error": float(error.mean().item()),
            "max_error_token": max_token,
            "max_error_feature": max_feature,
            "earliest_token_above": earliest,
            "final_token": _point_metrics(left[-1], right[-1]),
        }

    stacked_maxima = torch.stack(token_maxima)
    earliest_global: dict[str, Any] = {}
    for threshold in thresholds:
        crossing = torch.nonzero(stacked_maxima > threshold, as_tuple=False)
        if crossing.numel():
            first = min(
                ((int(item[1]), int(item[0])) for item in crossing),
                key=lambda item: (item[0], item[1]),
            )
            token, state_index = first
            difference = (
                batched[state_index, 0, token].float() - incremental[state_index, 0, token].float()
            ).abs()
            earliest_global[str(threshold)] = {
                "token": token,
                "state": names[state_index],
                "feature": int(difference.argmax().item()),
                "max_absolute_error": float(difference.max().item()),
            }
        else:
            earliest_global[str(threshold)] = None

    count = min(top_count, stacked_maxima.numel())
    top_values, top_indices = torch.topk(stacked_maxima.flatten(), count)
    top: list[dict[str, Any]] = []
    for value, flat_index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
        state_index, token = divmod(flat_index, sequence_tokens)
        difference = (
            batched[state_index, 0, token].float() - incremental[state_index, 0, token].float()
        ).abs()
        top.append(
            {
                "state": names[state_index],
                "token": token,
                "feature": int(difference.argmax().item()),
                "max_absolute_error": float(value),
            }
        )

    probes = {}
    for position in probe_positions:
        if not 0 <= position < sequence_tokens:
            raise ValueError(f"trace probe {position} lies outside the input")
        state_values = stacked_maxima[:, position]
        state_index = int(state_values.argmax().item())
        probes[str(position)] = {
            "largest_state": names[state_index],
            "max_absolute_error": float(state_values[state_index].item()),
            "final_norm": _point_metrics(batched[-1, 0, position], incremental[-1, 0, position]),
        }
    return {
        "shape": list(batched.shape),
        "all_finite": bool(torch.isfinite(batched).all() and torch.isfinite(incremental).all()),
        "earliest_global_above": earliest_global,
        "states": states,
        "top_token_state_divergences": top,
        "probes": probes,
    }


def _state_point_summary(left: Tensor, right: Tensor, names: list[str]) -> dict[str, Any]:
    states = {name: _point_metrics(left[index], right[index]) for index, name in enumerate(names)}
    maximum_name = max(states, key=lambda name: states[name]["max_absolute_error"])
    return {
        "largest_state": maximum_name,
        "max_absolute_error": states[maximum_name]["max_absolute_error"],
        "states": states,
    }


def _prefix_summary(
    left: Tensor,
    right: Tensor,
    *,
    end_exclusive: int,
    names: list[str],
) -> dict[str, Any]:
    states: dict[str, Any] = {}
    global_maximum = (-1.0, "", -1, -1)
    for index, name in enumerate(names):
        left_state = left[index, 0, :end_exclusive].float()
        right_state = right[index, 0, :end_exclusive].float()
        error = (left_state - right_state).abs()
        flat_index = int(error.argmax().item())
        token, feature = divmod(flat_index, error.shape[-1])
        maximum = float(error.max().item())
        states[name] = {
            "max_absolute_error": maximum,
            "mean_absolute_error": float(error.mean().item()),
            "cosine": float(
                F.cosine_similarity(left_state.flatten()[None], right_state.flatten()[None]).item()
            ),
            "exact_equal": bool(torch.equal(left_state, right_state)),
            "max_error_token": token,
            "max_error_feature": feature,
        }
        if maximum > global_maximum[0]:
            global_maximum = (maximum, name, token, feature)
    return {
        "tokens_compared": end_exclusive,
        "max_absolute_error": global_maximum[0],
        "largest_state": global_maximum[1],
        "max_error_token": global_maximum[2],
        "max_error_feature": global_maximum[3],
        "exact_equal": all(state["exact_equal"] for state in states.values()),
        "states": states,
    }


def _sequence_projection_summary(
    left: Tensor,
    right: Tensor,
    thresholds: list[float],
) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError("projection shapes do not match")
    left_by_token = left.movedim(-2, 0).reshape(left.shape[-2], -1).float()
    right_by_token = right.movedim(-2, 0).reshape(right.shape[-2], -1).float()
    error = (left_by_token - right_by_token).abs()
    per_token_max = error.amax(dim=-1)
    flat_index = int(error.argmax().item())
    token, feature = divmod(flat_index, error.shape[-1])
    earliest = {}
    for threshold in thresholds:
        crossing = torch.nonzero(per_token_max > threshold, as_tuple=False)
        earliest[str(threshold)] = int(crossing[0].item()) if crossing.numel() else None
    return {
        "shape": list(left.shape),
        "max_absolute_error": float(error.max().item()),
        "mean_absolute_error": float(error.mean().item()),
        "max_error_token": token,
        "max_error_feature": feature,
        "earliest_token_above": earliest,
        "exact_equal": bool(torch.equal(left, right)),
    }


def _batched_layer_zero_kv(model: nn.Module, input_ids: Tensor) -> tuple[Tensor, Tensor]:
    decoder = model.model
    hidden_states = decoder.embed_tokens(input_ids)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    position_embeddings = decoder.rotary_emb(hidden_states, positions)
    attention = decoder.layers[0].self_attn
    hidden_shape = (*hidden_states.shape[:-1], -1, attention.head_dim)
    query = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cosine, sine = position_embeddings
    _, key = apply_rotary_pos_emb(query, key, cosine, sine)
    return key, value


def _answer_score(
    model: nn.Module,
    hidden_states: Tensor,
    label_token_ids: tuple[int, ...],
    gold_label: int,
) -> tuple[dict[str, Any], Tensor]:
    logits = model.lm_head(hidden_states[:, -1]).float()[0]
    label_logits = logits[list(label_token_ids)]
    label_log_probabilities = label_logits.log_softmax(dim=-1)
    label_probabilities = label_log_probabilities.exp()
    prediction = int(label_logits.argmax().item())
    return (
        {
            "label_token_ids": list(label_token_ids),
            "label_logits": [float(value) for value in label_logits.tolist()],
            "label_log_probabilities": [float(value) for value in label_log_probabilities.tolist()],
            "label_probabilities": [float(value) for value in label_probabilities.tolist()],
            "loss": float(-label_log_probabilities[gold_label].item()),
            "prediction": prediction,
            "gold_label": gold_label,
            "correct": prediction == gold_label,
        },
        logits,
    )


def _final_comparison(
    model: nn.Module,
    batched: Tensor,
    incremental: Tensor,
    *,
    label_token_ids: tuple[int, ...],
    gold_label: int,
) -> dict[str, Any]:
    batched_answer, batched_logits = _answer_score(model, batched, label_token_ids, gold_label)
    incremental_answer, incremental_logits = _answer_score(
        model, incremental, label_token_ids, gold_label
    )
    logit_error = (batched_logits - incremental_logits).abs()
    batched_probabilities = torch.tensor(batched_answer["label_probabilities"])
    incremental_probabilities = torch.tensor(incremental_answer["label_probabilities"])
    label_log_probability_error = (
        torch.tensor(batched_answer["label_log_probabilities"])
        - torch.tensor(incremental_answer["label_log_probabilities"])
    ).abs()
    return {
        "hidden": _point_metrics(batched[:, -1], incremental[:, -1]),
        "full_vocabulary_logits": {
            "max_absolute_error": float(logit_error.max().item()),
            "mean_absolute_error": float(logit_error.mean().item()),
            "batched_top1_token": int(batched_logits.argmax().item()),
            "incremental_top1_token": int(incremental_logits.argmax().item()),
            "top1_match": int(batched_logits.argmax().item())
            == int(incremental_logits.argmax().item()),
        },
        "four_label_answer": {
            "batched": batched_answer,
            "incremental": incremental_answer,
            "incremental_minus_batched_loss": incremental_answer["loss"] - batched_answer["loss"],
            "max_label_log_probability_error": float(label_log_probability_error.max().item()),
            "total_variation_distance": float(
                0.5 * (batched_probabilities - incremental_probabilities).abs().sum().item()
            ),
            "prediction_match": batched_answer["prediction"] == incremental_answer["prediction"],
        },
    }


def _mask_structure(
    rolling: tuple[Tensor, ...],
    causal_oracle: tuple[Tensor, ...],
    *,
    retrieval_layer: int,
    selected_range: tuple[int, int],
) -> dict[str, Any]:
    rolling_visible = rolling[0][0, 0] == 0
    oracle_visible = causal_oracle[retrieval_layer][0, 0] == 0
    rolling_future_visible = int(torch.triu(rolling_visible, diagonal=1).sum().item())
    oracle_future_visible = int(torch.triu(oracle_visible, diagonal=1).sum().item())
    actual_difference = oracle_visible ^ rolling_visible
    expected_difference = torch.zeros_like(actual_difference)
    start, end = selected_range
    expected_difference[-1, start:end] = True
    other_layers_equal = all(
        torch.equal(rolling[index], causal_oracle[index])
        for index in range(len(rolling))
        if index != retrieval_layer
    )
    return {
        "rolling_future_visible_cells": rolling_future_visible,
        "causal_oracle_future_visible_cells": oracle_future_visible,
        "oracle_difference_cells": int(actual_difference.sum().item()),
        "expected_oracle_difference_cells": end - start,
        "oracle_difference_exactly_final_retrieval_row": bool(
            torch.equal(actual_difference, expected_difference)
        ),
        "all_nonretrieval_layers_equal_rolling": other_layers_equal,
        "passed": rolling_future_visible == 0
        and oracle_future_visible == 0
        and torch.equal(actual_difference, expected_difference)
        and other_layers_equal,
    }


def _run_final_mode(
    model: nn.Module,
    input_ids: Tensor,
    masks: tuple[Tensor, ...],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        batched, _ = qwen_forward_with_layer_masks(model, input_ids, masks)
    torch.cuda.synchronize(device)
    batched_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode():
        incremental = qwen_incremental_with_layer_masks(model, input_ids, masks)
    torch.cuda.synchronize(device)
    incremental_seconds = time.perf_counter() - started
    return (
        batched,
        incremental,
        {
            "batched_seconds": batched_seconds,
            "incremental_seconds": incremental_seconds,
        },
    )


def _run_condition(
    specification: dict[str, Any],
    diagnostic: dict[str, Any],
    condition: dict[str, Any],
    example: Any,
    *,
    device: torch.device,
    condition_index: int,
    condition_count: int,
) -> dict[str, Any]:
    progress = _Progress(
        condition["name"],
        condition_index=condition_index,
        condition_count=condition_count,
        step_count=17,
    )
    progress.report("loading frozen model")
    model = _load_model(specification, condition, device)
    progress.report("model loaded", advance=True)
    names = _state_names(model)
    input_ids = torch.tensor(example.input_ids, dtype=torch.long, device=device)[None]
    geometry = specification["geometry"]
    common = {
        "length": len(example.input_ids),
        "n_layers": model.config.num_hidden_layers,
        "rolling_tokens": geometry["rolling_tokens"],
        "global_prefix_end": example.global_prefix_end,
        "dtype": model.dtype,
        "device": device,
    }
    selected_range = example.block_ranges[example.target_block]
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
    progress.report("causal masks verified", advance=True)
    thresholds = [float(value) for value in diagnostic["absolute_error_thresholds"]]
    probes = [int(value) for value in diagnostic["prefix_probe_positions"]]
    cutoffs = [int(value) for value in diagnostic["future_invariance_cutoffs"]]

    with _backend_context(condition), torch.inference_mode():
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        rolling_batched, _, batched_trace = _forward_trace(
            model,
            input_ids,
            rolling_plan.masks,
            trace_device=torch.device("cpu"),
        )
        torch.cuda.synchronize(device)
        batched_seconds = time.perf_counter() - started
        progress.report("rolling batched trace captured", advance=True)

        started = time.perf_counter()
        rolling_incremental, cache, incremental_trace, cache_lengths = _incremental_trace(
            model, input_ids, rolling_plan.masks
        )
        torch.cuda.synchronize(device)
        incremental_seconds = time.perf_counter() - started
        progress.report("rolling incremental trace captured", advance=True)

        batched_key, batched_value = _batched_layer_zero_kv(model, input_ids)
        cached_key = cache.layers[0].keys
        cached_value = cache.layers[0].values
        layer_zero_cache = {
            "key": _sequence_projection_summary(batched_key, cached_key, thresholds),
            "value": _sequence_projection_summary(batched_value, cached_value, thresholds),
            "lengths": cache_lengths,
        }
        del cache, batched_key, batched_value, cached_key, cached_value
        incremental_trace_cpu = incremental_trace.cpu()
        del incremental_trace
        incremental_trace = incremental_trace_cpu
        torch.cuda.empty_cache()
        trace_summary = _trace_summary(
            batched_trace,
            incremental_trace,
            names=names,
            thresholds=thresholds,
            probe_positions=probes,
            top_count=int(diagnostic["top_divergences"]),
        )
        progress.report("divergence and layer-0 cache localized", advance=True)

        prefix_replays: dict[str, Any] = {}
        for probe_index, position in enumerate(probes, start=1):
            prefix_masks = tuple(
                mask[:, :, : position + 1, : position + 1] for mask in rolling_plan.masks
            )
            _, _, prefix_trace = _forward_trace(
                model,
                input_ids[:, : position + 1],
                prefix_masks,
                trace_device=torch.device("cpu"),
            )
            prefix_point = prefix_trace[:, 0, -1]
            full_point = batched_trace[:, 0, position]
            cached_point = incremental_trace[:, 0, position]
            prefix_replays[str(position)] = {
                "uncached_prefix_vs_full_batch": _state_point_summary(
                    prefix_point, full_point, names
                ),
                "uncached_prefix_vs_cached_incremental": _state_point_summary(
                    prefix_point, cached_point, names
                ),
                "full_batch_vs_cached_incremental": _state_point_summary(
                    full_point, cached_point, names
                ),
            }
            del prefix_trace
            progress.report(
                f"uncached-prefix probe {probe_index}/{len(probes)} at token {position}",
                advance=True,
            )

        _, _, repeat_trace = _forward_trace(
            model,
            input_ids,
            rolling_plan.masks,
            trace_device=torch.device("cpu"),
        )
        repeatability = {
            str(cutoff): _prefix_summary(
                batched_trace,
                repeat_trace,
                end_exclusive=cutoff + 1,
                names=names,
            )
            for cutoff in cutoffs
        }
        del repeat_trace
        progress.report("same-input repeatability measured", advance=True)

        future_counterfactuals: dict[str, Any] = {}
        for cutoff_index, cutoff in enumerate(cutoffs, start=1):
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
            comparison = _prefix_summary(
                batched_trace,
                changed_trace,
                end_exclusive=cutoff + 1,
                names=names,
            )
            repeat = repeatability[str(cutoff)]
            comparison["consistent_with_same_input_repeatability"] = all(
                comparison["states"][name]["max_absolute_error"]
                <= repeat["states"][name]["max_absolute_error"]
                for name in names
            )
            future_counterfactuals[str(cutoff)] = comparison
            del changed_ids, changed_trace
            progress.report(
                f"future-invariance probe {cutoff_index}/{len(cutoffs)} at token {cutoff}",
                advance=True,
            )

        rolling_final = _final_comparison(
            model,
            rolling_batched,
            rolling_incremental,
            label_token_ids=example.label_token_ids,
            gold_label=example.gold_label,
        )
        oracle_batched, oracle_incremental, oracle_timings = _run_final_mode(
            model,
            input_ids,
            oracle_plan.masks,
            device=device,
        )
        progress.report("causal-oracle batch/incremental pair measured", advance=True)
        oracle_final = _final_comparison(
            model,
            oracle_batched,
            oracle_incremental,
            label_token_ids=example.label_token_ids,
            gold_label=example.gold_label,
        )

    batch_utility = (
        rolling_final["four_label_answer"]["batched"]["loss"]
        - oracle_final["four_label_answer"]["batched"]["loss"]
    )
    incremental_utility = (
        rolling_final["four_label_answer"]["incremental"]["loss"]
        - oracle_final["four_label_answer"]["incremental"]["loss"]
    )
    result = {
        "condition": condition,
        "model_attention_implementation": model.config._attn_implementation,
        "mask_structure": mask_structure,
        "timings": {
            "rolling_batched_seconds": batched_seconds,
            "rolling_incremental_seconds": incremental_seconds,
            "causal_oracle_batched_seconds": oracle_timings["batched_seconds"],
            "causal_oracle_incremental_seconds": oracle_timings["incremental_seconds"],
        },
        "rolling_trace": trace_summary,
        "layer_zero_batch_vs_cache": layer_zero_cache,
        "prefix_replays": prefix_replays,
        "same_input_repeatability": repeatability,
        "future_counterfactuals": future_counterfactuals,
        "causal_invariance": {
            "mask_structure_passed": mask_structure["passed"],
            "all_future_counterfactuals_consistent_with_repeatability": all(
                value["consistent_with_same_input_repeatability"]
                for value in future_counterfactuals.values()
            ),
        },
        "modes": {
            "rolling": rolling_final,
            "causal_oracle": oracle_final,
        },
        "four_label_utility_sensitivity": {
            "batched_causal_oracle_minus_rolling_utility_nats": batch_utility,
            "incremental_causal_oracle_minus_rolling_utility_nats": incremental_utility,
            "incremental_minus_batched_utility_nats": incremental_utility - batch_utility,
        },
        "complete": True,
    }
    progress.report("condition checkpoint ready", advance=True)
    del (
        model,
        input_ids,
        rolling_plan,
        oracle_plan,
        rolling_batched,
        rolling_incremental,
        batched_trace,
        incremental_trace,
        oracle_batched,
        oracle_incremental,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _validate_original_failure(diagnostic: dict[str, Any]) -> dict[str, Any]:
    original = diagnostic["original_failed_gate"]
    path = Path(original["path"])
    if _sha256(path) != original["sha256"]:
        raise ValueError("the original failed-gate artifact changed")
    result = load_json(path)
    if result.get("passed") is not False:
        raise ValueError("the original Stage B gate is not a preserved failure")
    if result["source_identity"]["git_revision"] != original["source_revision"]:
        raise ValueError("the original failed-gate source revision changed")
    if result["input_token_sha256"] != original["input_token_sha256"]:
        raise ValueError("the original failed-gate input hash changed")
    return result


def _validate_superseded_partial(diagnostic: dict[str, Any]) -> None:
    partial = diagnostic.get("supersedes_partial_diagnostic")
    if partial is None:
        return
    path = Path(partial["path"])
    if _sha256(path) != partial["sha256"]:
        raise ValueError("the superseded partial diagnostic changed")
    result = load_json(path)
    if result.get("complete") is not False:
        raise ValueError("the superseded diagnostic is not an incomplete artifact")
    if result["source_identity"]["git_revision"] != partial["source_revision"]:
        raise ValueError("the superseded diagnostic source revision changed")
    if list(result["conditions"]) != partial["completed_conditions"]:
        raise ValueError("the superseded diagnostic condition set changed")


def _validate_resume(result: dict[str, Any], expected: dict[str, Any]) -> None:
    actual = {key: result.get(key) for key in expected}
    if actual != expected:
        raise ValueError("existing diagnostic output does not match source or configuration")


def main() -> None:
    parser = argparse.ArgumentParser(description="Localize Stage B batch/incremental divergence")
    parser.add_argument("--diagnostic-config", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output")
    parser.add_argument("--max-conditions-this-run", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    diagnostic = load_json(args.diagnostic_config)
    if diagnostic["trace_mode"] != "rolling" or diagnostic["answer_modes"] != [
        "rolling",
        "causal_oracle",
    ]:
        raise ValueError("diagnostic modes do not match the bounded implementation")
    specification = load_json(diagnostic["base_config_path"])
    original_failure = _validate_original_failure(diagnostic)
    _validate_superseded_partial(diagnostic)
    dataset = specification["datasets"]["evaluation"]
    dataset_path = Path(dataset["cache_path"])
    _download(dataset["url"], dataset_path, dataset["sha256"])
    tokenizer = _load_tokenizer(specification)
    with dataset_path.open() as source:
        payload = json.load(source)
    geometry = specification["geometry"]
    oracle = specification["oracle_parity"]
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
    example = examples[diagnostic["example_index"]]
    input_hash = _token_hash(example.input_ids)
    if input_hash != diagnostic["original_failed_gate"]["input_token_sha256"]:
        raise ValueError("the diagnostic example does not reproduce the failed-gate input")
    preparation = {
        "sequence_tokens": len(example.input_ids),
        "input_token_sha256": input_hash,
        "squad_id": example.squad_id,
        "global_prefix_end": example.global_prefix_end,
        "query_start": example.query_start,
        "selected_range": list(example.block_ranges[example.target_block]),
        "decision_position": len(example.input_ids) - 1,
        "retrieval_layer": geometry["retrieval_layer"],
        "label_token_ids": list(example.label_token_ids),
        "gold_label": example.gold_label,
    }
    if args.prepare_only:
        print(json.dumps(preparation, indent=2, sort_keys=True))
        return

    identity = _source_identity(args.diagnostic_config, diagnostic)
    if not identity["tracked_files_clean"]:
        raise ValueError("diagnostic source files must be committed before measurement")
    if not torch.cuda.is_available():
        raise RuntimeError("the authorized diagnostic requires CUDA")
    device = torch.device(args.device)
    expected = {
        "diagnostic_config": diagnostic,
        "base_config": specification,
        "source_identity": identity,
        "original_failed_gate_sha256": diagnostic["original_failed_gate"]["sha256"],
        "dataset_sha256": _sha256(dataset_path),
        "input_token_sha256": input_hash,
        "device": args.device,
        "decision": diagnostic["decision"],
    }
    output = Path(args.output or diagnostic["output"])
    if output.exists():
        result = load_json(output)
        _validate_resume(result, expected)
    else:
        result = {
            "schema_version": 1,
            "test": diagnostic["diagnostic"],
            **expected,
            "environment": {
                **environment(),
                "transformers": transformers.__version__,
                "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
                "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
                "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
            },
            "preparation": preparation,
            "original_failure_summary": {
                "passed": original_failure["passed"],
                "source_revision": original_failure["source_identity"]["git_revision"],
                "modes": original_failure["modes"],
            },
            "conditions": {},
            "complete": False,
        }
        write_json(output, result)

    completed = set(result["conditions"])
    remaining = [
        condition for condition in diagnostic["conditions"] if condition["name"] not in completed
    ]
    maximum = args.max_conditions_this_run
    if maximum is None:
        maximum = diagnostic["execution"]["default_max_conditions_per_invocation"]
    remaining = remaining[:maximum]
    condition_positions = {
        condition["name"]: index
        for index, condition in enumerate(diagnostic["conditions"], start=1)
    }
    for condition in remaining:
        measured = _run_condition(
            specification,
            diagnostic,
            condition,
            example,
            device=device,
            condition_index=condition_positions[condition["name"]],
            condition_count=len(diagnostic["conditions"]),
        )
        result["conditions"][condition["name"]] = measured
        result["complete"] = len(result["conditions"]) == len(diagnostic["conditions"])
        write_json(output, result)
        print(
            json.dumps(
                {
                    "completed_condition": condition["name"],
                    "completed_conditions": list(result["conditions"]),
                    "complete": result["complete"],
                    "stage_b_status": "blocked",
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
