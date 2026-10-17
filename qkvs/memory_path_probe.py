from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from qkvs.attention import AttentionOutput, _apply_rope, _gather_cache
from qkvs.config import ModelConfig
from qkvs.data import make_address_overrides, make_oracle_window_masks
from qkvs.model import QKVSDecoder
from qkvs.runtime import environment, load_json, write_json
from qkvs.training import TrainConfig, _batch, _retrieval_mask


def _attention_weight(
    attention,
    hidden: torch.Tensor,
    output: AttentionOutput,
    *,
    query_positions: torch.Tensor,
    target_positions: torch.Tensor,
) -> float:
    assert output.selection is not None
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    q = _apply_rope(attention._split_heads(attention.q_proj(hidden)), positions)
    k = _apply_rope(attention._split_heads(attention.k_proj(hidden)), positions)
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    indices = output.selection.indices[batch_indices, query_positions].unsqueeze(1)
    valid = output.selection.valid[batch_indices, query_positions].unsqueeze(1)
    selected_k = _gather_cache(k, indices)
    selected_q = q[batch_indices, :, query_positions].unsqueeze(2)
    logits = torch.einsum(
        "bhqd,bhqmd->bhqm",
        selected_q,
        selected_k,
    ) / math.sqrt(q.shape[-1])
    weights = torch.softmax(logits.masked_fill(~valid.unsqueeze(1), -torch.inf), dim=-1)
    target = (indices == target_positions.view(-1, 1, 1)).unsqueeze(1)
    return (weights * target).sum(dim=-1).mean().item()


def _controlled_forward(
    model: QKVSDecoder,
    batch,
    model_config: ModelConfig,
    train_config: TrainConfig,
    enabled_layers: set[int] | None,
) -> tuple[torch.Tensor, list[float], list[float]]:
    seq_len = batch.values.shape[1]
    overrides = None
    window_masks = None
    if enabled_layers is not None:
        overrides = make_address_overrides(
            batch,
            seq_len=seq_len,
            n_windows=model_config.attention().n_windows,
            device=batch.values.device,
        )
        window_masks = make_oracle_window_masks(
            batch,
            seq_len=seq_len,
            n_windows=model_config.attention().n_windows,
            device=batch.values.device,
        )
        window_masks[:, :, -1] = False
        for layer_index in enabled_layers:
            window_masks[layer_index, :, -1, 0] = True

    hidden_inputs: list[torch.Tensor] = []
    attention_outputs: list[AttentionOutput] = []
    hooks = []
    for layer in model.layers:
        hooks.append(
            layer.attention.register_forward_pre_hook(
                lambda _module, inputs: hidden_inputs.append(inputs[0].detach())
            )
        )
        hooks.append(
            layer.attention.register_forward_hook(
                lambda _module, _inputs, output: attention_outputs.append(output)
            )
        )
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=batch.values.device.type,
            dtype=torch.bfloat16,
            enabled=batch.values.device.type == "cuda",
        ),
    ):
        output = model(
            batch.values,
            batch.pointers,
            batch.token_types,
            address_overrides=overrides,
            retrieval_mask=_retrieval_mask(batch, train_config),
            semantic_ids=batch.semantic_ids,
            historical_window_masks=window_masks,
        )
    for hook in hooks:
        hook.remove()

    markers = batch.route_positions[0]
    if train_config.task == "memory_bank":
        source_offset = train_config.memory_bank_source_offset
    else:
        source_offset = int(train_config.memory_anchor_source_offset or 1)
    sources = markers - source_offset
    final_queries = torch.full_like(markers, seq_len - 1)
    source_weights = []
    marker_weights = []
    for layer_index, layer in enumerate(model.layers):
        source_weights.append(
            _attention_weight(
                layer.attention,
                hidden_inputs[layer_index],
                attention_outputs[layer_index],
                query_positions=markers,
                target_positions=sources,
            )
        )
        marker_weights.append(
            _attention_weight(
                layer.attention,
                hidden_inputs[layer_index],
                attention_outputs[layer_index],
                query_positions=final_queries,
                target_positions=markers,
            )
        )
    return output.logits[:, -1], source_weights, marker_weights


def _probe(
    model: QKVSDecoder,
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    batches: int,
    fixed_batch: bool,
    seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(train_config.seed + 60_000)
    fixed = None
    if fixed_batch:
        fixed = _batch(
            model_config,
            train_config,
            seq_len,
            torch.Generator().manual_seed(train_config.seed + 404),
        )
    mode_names: dict[str, set[int] | None] = {
        "native": None,
        "no_historical_read": set(),
        "all_layer_true_oracle": set(range(model_config.n_layers)),
        **{f"only_layer_{index}": {index} for index in range(model_config.n_layers)},
    }
    correct = {name: 0 for name in mode_names}
    total = 0
    source_weights = torch.zeros(model_config.n_layers)
    marker_weights = torch.zeros(model_config.n_layers)

    for _ in range(batches):
        batch = (
            fixed
            if fixed is not None
            else _batch(
                model_config,
                train_config,
                seq_len,
                generator,
            )
        ).to(device)
        for name, enabled_layers in mode_names.items():
            logits, source, marker = _controlled_forward(
                model, batch, model_config, train_config, enabled_layers
            )
            correct[name] += (logits.argmax(dim=-1) == batch.answers).sum().item()
            if name == "all_layer_true_oracle":
                source_weights += torch.tensor(source)
                marker_weights += torch.tensor(marker)
        total += batch.answers.numel()

    return {
        "answer_accuracy": {name: value / total for name, value in correct.items()},
        "mean_source_to_marker_attention_by_layer": (source_weights / batches).tolist(),
        "mean_marker_to_query_attention_by_layer": (marker_weights / batches).tolist(),
        "examples": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe layer-specific marker KV retrievability")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--fixed-batch", action="store_true")
    parser.add_argument("--seq-len", type=int)
    arguments = parser.parse_args()
    specification = load_json(arguments.config)
    run = next(value for value in specification["runs"] if value["name"] == arguments.run)
    model_config = ModelConfig(**{**specification["model"], **run.get("model", {})})
    train_config = TrainConfig.from_dict(
        {
            **specification["training"],
            **run.get("training", {}),
            "name": run["name"],
            "seed": specification["seed"],
        }
    )
    device = torch.device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=True)
    model = QKVSDecoder(model_config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    seq_len = arguments.seq_len or train_config.train_seq_len
    if arguments.fixed_batch and seq_len != train_config.train_seq_len:
        raise ValueError("the fixed training batch exists only at the training length")
    result = {
        "environment": environment(),
        "checkpoint_step": checkpoint.get("step"),
        "fixed_training_batch": arguments.fixed_batch,
        "sequence_length": seq_len,
        "probe": _probe(
            model,
            model_config,
            train_config,
            batches=arguments.batches,
            fixed_batch=arguments.fixed_batch,
            seq_len=seq_len,
            device=device,
        ),
        "interpretation": (
            "only_layer_N enables exactly one oracle marker window at layer N; all unrelated "
            "historical windows are disabled. Attention weights are measured on actual layer "
            "Q/K and selected cache entries."
        ),
    }
    write_json(Path(arguments.output), result)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
