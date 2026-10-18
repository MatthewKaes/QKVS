from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache, DynamicCache

from qkvs.utility_gate import build_attention_mask


@dataclass(frozen=True)
class LayerMaskPlan:
    masks: tuple[Tensor, ...]
    decision_position: int
    retrieval_layer: int | None
    selected_range: tuple[int, int] | None


def rolling_mask_plan(
    *,
    length: int,
    n_layers: int,
    rolling_tokens: int,
    global_prefix_end: int,
    dtype: torch.dtype,
    device: torch.device,
) -> LayerMaskPlan:
    mask = build_attention_mask(
        length,
        dtype=dtype,
        device=device,
        rolling_tokens=rolling_tokens,
        global_ranges=[(0, global_prefix_end)],
    )
    return LayerMaskPlan(
        masks=tuple(mask for _ in range(n_layers)),
        decision_position=length - 1,
        retrieval_layer=None,
        selected_range=None,
    )


def causal_oracle_mask_plan(
    *,
    length: int,
    n_layers: int,
    rolling_tokens: int,
    global_prefix_end: int,
    selected_range: tuple[int, int],
    retrieval_layer: int,
    dtype: torch.dtype,
    device: torch.device,
) -> LayerMaskPlan:
    if not 0 < retrieval_layer < n_layers:
        raise ValueError("retrieval layer must follow at least one causal representation layer")
    base = rolling_mask_plan(
        length=length,
        n_layers=n_layers,
        rolling_tokens=rolling_tokens,
        global_prefix_end=global_prefix_end,
        dtype=dtype,
        device=device,
    )
    masks = list(base.masks)
    selected = masks[retrieval_layer].clone()
    start, end = selected_range
    selected[:, :, base.decision_position, start:end] = 0.0
    masks[retrieval_layer] = selected
    return LayerMaskPlan(
        masks=tuple(masks),
        decision_position=base.decision_position,
        retrieval_layer=retrieval_layer,
        selected_range=selected_range,
    )


def stage_a_oracle_mask_plan(
    *,
    length: int,
    n_layers: int,
    rolling_tokens: int,
    global_prefix_end: int,
    query_start: int,
    selected_range: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> LayerMaskPlan:
    mask = build_attention_mask(
        length,
        dtype=dtype,
        device=device,
        rolling_tokens=rolling_tokens,
        query_start=query_start,
        global_ranges=[(0, global_prefix_end)],
        selected_ranges=[selected_range],
    )
    return LayerMaskPlan(
        masks=tuple(mask for _ in range(n_layers)),
        decision_position=length - 1,
        retrieval_layer=None,
        selected_range=selected_range,
    )


def qwen_forward_with_layer_masks(
    model: nn.Module,
    input_ids: Tensor,
    layer_masks: Sequence[Tensor],
    *,
    past_key_values: Cache | None = None,
    cache_position: Tensor | None = None,
    use_cache: bool = False,
) -> tuple[Tensor, Cache | None]:
    decoder = model.model
    if len(layer_masks) != decoder.config.num_hidden_layers:
        raise ValueError("one attention mask is required per decoder layer")
    hidden_states = decoder.embed_tokens(input_ids)
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
    for layer, attention_mask in zip(decoder.layers, layer_masks, strict=True):
        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    return decoder.norm(hidden_states), past_key_values if use_cache else None


def qwen_incremental_with_layer_masks(
    model: nn.Module,
    input_ids: Tensor,
    layer_masks: Sequence[Tensor],
) -> Tensor:
    cache = DynamicCache(config=model.config)
    outputs: list[Tensor] = []
    for position in range(input_ids.shape[1]):
        token_masks = [mask[:, :, position : position + 1, : position + 1] for mask in layer_masks]
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
    return torch.cat(outputs, dim=1)
