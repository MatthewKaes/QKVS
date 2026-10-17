from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from qkvs.config import AttentionConfig


@dataclass(frozen=True)
class Selection:
    """Token occurrences selected for each query; duplicate indices are significant."""

    indices: Tensor  # [batch, queries, slots]
    valid: Tensor  # [batch, queries, slots]


@dataclass(frozen=True)
class AttentionOutput:
    hidden: Tensor
    addresses: Tensor | None
    selection: Selection | None


@dataclass(frozen=True)
class KVCache:
    keys: Tensor  # [batch, heads, capacity, head_dim]
    values: Tensor
    length: int


def normalized_block_centers(token_indices: Tensor, history_end: int, block_size: int) -> Tensor:
    """Return normalized addresses that decode to the blocks containing token_indices."""
    return normalized_window_addresses(token_indices, history_end, block_size)


def normalized_window_addresses(
    token_indices: Tensor, history_end: int, address_stride: int
) -> Tensor:
    """Return normalized addresses for canonical windows containing token_indices."""
    if history_end <= 0:
        raise ValueError("history_end must be positive")
    n_starts = math.ceil(history_end / address_stride)
    starts = torch.div(token_indices, address_stride, rounding_mode="floor")
    return (starts.to(torch.float32) + 0.5) / n_starts


def _selection_for_positions(
    addresses: Tensor,
    positions: Tensor,
    *,
    dense_window: int,
    retrieval_budget: int,
    block_size: int,
    address_stride: int,
    full_below_threshold: bool,
) -> Selection:
    """Build fixed-width candidate lists without inspecting historical K or V tensors."""
    if addresses.ndim != 3:
        raise ValueError("addresses must have shape [batch, queries, windows]")
    batch, queries, n_windows = addresses.shape
    if n_windows * block_size != retrieval_budget:
        raise ValueError("address count does not match retrieval budget")
    if positions.shape != (queries,):
        raise ValueError("positions must contain one absolute position per query")

    device = addresses.device
    positions = positions.to(device=device, dtype=torch.long)
    query_positions = positions.view(1, queries, 1)
    history_end = query_positions - dense_window + 1
    address_count = torch.div(
        history_end.clamp_min(0) + address_stride - 1,
        address_stride,
        rounding_mode="floor",
    )

    scaled = addresses.clamp(0.0, 1.0 - torch.finfo(addresses.dtype).eps)
    address_indices = torch.floor(scaled * address_count.to(addresses.dtype)).to(torch.long)
    address_indices = torch.minimum(address_indices, (address_count - 1).clamp_min(0))
    offsets = torch.arange(block_size, device=device).view(1, 1, 1, block_size)
    historical = address_indices.unsqueeze(-1) * address_stride + offsets
    historical = historical.reshape(batch, queries, retrieval_budget)
    historical_valid = (historical >= 0) & (historical < history_end)

    recent_offsets = torch.arange(dense_window, device=device) - dense_window + 1
    recent = query_positions + recent_offsets.view(1, 1, dense_window)
    recent = recent.expand(batch, -1, -1)
    recent_valid = (recent >= 0) & (recent <= query_positions)

    sparse_indices = torch.cat((historical, recent), dim=-1)
    sparse_valid = torch.cat((historical_valid, recent_valid), dim=-1)

    if full_below_threshold:
        budget = dense_window + retrieval_budget
        slots = torch.arange(budget, device=device).view(1, 1, budget)
        dense_indices = slots.expand(batch, queries, -1)
        dense_valid = dense_indices <= query_positions
        use_dense = (query_positions + 1) <= budget
        sparse_indices = torch.where(use_dense, dense_indices, sparse_indices)
        sparse_valid = torch.where(use_dense, dense_valid, sparse_valid)

    safe_indices = torch.minimum(sparse_indices.clamp_min(0), query_positions)
    return Selection(indices=safe_indices, valid=sparse_valid)


def build_selection(
    addresses: Tensor,
    *,
    dense_window: int,
    retrieval_budget: int,
    block_size: int,
    address_stride: int | None = None,
) -> Selection:
    """Build causal selections for every sequence position."""
    positions = torch.arange(addresses.shape[1], device=addresses.device)
    return _selection_for_positions(
        addresses,
        positions,
        dense_window=dense_window,
        retrieval_budget=retrieval_budget,
        block_size=block_size,
        address_stride=block_size if address_stride is None else address_stride,
        full_below_threshold=True,
    )


def build_query_selection(
    addresses: Tensor,
    query_position: int,
    *,
    dense_window: int,
    retrieval_budget: int,
    block_size: int,
    address_stride: int | None = None,
) -> Selection:
    """Build one incremental query's selection in O(L + R) work."""
    if addresses.ndim != 2:
        raise ValueError("addresses must have shape [batch, windows]")
    positions = torch.tensor([query_position], device=addresses.device)
    return _selection_for_positions(
        addresses.unsqueeze(1),
        positions,
        dense_window=dense_window,
        retrieval_budget=retrieval_budget,
        block_size=block_size,
        address_stride=block_size if address_stride is None else address_stride,
        full_below_threshold=True,
    )


def build_rolling_selection(
    batch: int, seq_len: int, window: int, device: torch.device
) -> Selection:
    positions = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    offsets = torch.arange(window, device=device).view(1, 1, window) - window + 1
    indices = (positions + offsets).expand(batch, -1, -1)
    valid = (indices >= 0) & (indices <= positions)
    return Selection(indices=indices.clamp_min(0), valid=valid)


def _gather_cache(cache: Tensor, indices: Tensor) -> Tensor:
    """Gather [B,H,T,D] cache entries into [B,H,Q,M,D] without a T-by-T expansion."""
    batch = cache.shape[0]
    cache_time_major = cache.transpose(1, 2)
    batch_index = torch.arange(batch, device=cache.device).view(batch, 1, 1)
    gathered = cache_time_major[batch_index, indices]
    return gathered.permute(0, 3, 1, 2, 4)


def selected_attention(q: Tensor, k: Tensor, v: Tensor, selection: Selection) -> Tensor:
    """Scaled dot-product attention over selected token occurrences."""
    selected_k = _gather_cache(k, selection.indices)
    selected_v = _gather_cache(v, selection.indices)
    logits = torch.einsum("bhqd,bhqmd->bhqm", q, selected_k) / math.sqrt(q.shape[-1])
    logits = logits.masked_fill(~selection.valid.unsqueeze(1), -torch.inf)
    weights = torch.softmax(logits, dim=-1)
    return torch.einsum("bhqm,bhqmd->bhqd", weights, selected_v)


def _streaming_attention_chunk(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    indices: Tensor,
    valid: Tensor,
    candidate_tile_size: int,
) -> Tensor:
    """Online softmax over candidate tiles without a full selected-K/V tensor."""
    batch, heads, queries, head_dim = q.shape
    running_max = torch.full(
        (batch, heads, queries),
        -torch.inf,
        device=q.device,
        dtype=torch.float32,
    )
    running_sum = torch.zeros_like(running_max)
    accumulator = torch.zeros(
        (batch, heads, queries, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    scale = 1.0 / math.sqrt(head_dim)

    for start in range(0, indices.shape[-1], candidate_tile_size):
        end = min(indices.shape[-1], start + candidate_tile_size)
        tile_indices = indices[..., start:end]
        tile_valid = valid[..., start:end]
        selected_k = _gather_cache(k, tile_indices)
        selected_v = _gather_cache(v, tile_indices)
        logits = torch.einsum("bhqd,bhqmd->bhqm", q, selected_k).float() * scale
        logits = logits.masked_fill(~tile_valid.unsqueeze(1), -torch.inf)
        tile_max = logits.amax(dim=-1)
        updated_max = torch.maximum(running_max, tile_max)
        safe_max = torch.where(torch.isfinite(updated_max), updated_max, 0.0)
        previous_scale = torch.where(
            torch.isfinite(running_max), torch.exp(running_max - safe_max), 0.0
        )
        weights = torch.where(
            tile_valid.unsqueeze(1),
            torch.exp(logits - safe_max.unsqueeze(-1)),
            0.0,
        )
        running_sum = running_sum * previous_scale + weights.sum(dim=-1)
        accumulator = accumulator * previous_scale.unsqueeze(-1) + torch.einsum(
            "bhqm,bhqmd->bhqd", weights, selected_v.float()
        )
        running_max = updated_max

    return (accumulator / running_sum.unsqueeze(-1)).to(q.dtype)


def streaming_selected_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    selection: Selection,
    *,
    query_chunk_size: int,
    candidate_tile_size: int,
    checkpoint_chunks: bool,
) -> Tensor:
    """Bound peak selected-K/V memory in both forward and backward."""
    outputs: list[Tensor] = []
    for start in range(0, q.shape[2], query_chunk_size):
        end = min(q.shape[2], start + query_chunk_size)
        arguments = (
            q[:, :, start:end],
            k,
            v,
            selection.indices[:, start:end],
            selection.valid[:, start:end],
        )

        def attend_chunk(
            chunk_q: Tensor,
            all_k: Tensor,
            all_v: Tensor,
            chunk_indices: Tensor,
            chunk_valid: Tensor,
        ) -> Tensor:
            return _streaming_attention_chunk(
                chunk_q,
                all_k,
                all_v,
                chunk_indices,
                chunk_valid,
                candidate_tile_size,
            )

        if checkpoint_chunks and torch.is_grad_enabled():
            output = checkpoint(
                attend_chunk,
                *arguments,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            output = attend_chunk(*arguments)
        outputs.append(output)
    return torch.cat(outputs, dim=2)


def duplicate_attention_reference(q: Tensor, k: Tensor, v: Tensor, indices: Tensor) -> Tensor:
    """Small single-query reference that intentionally preserves repeated occurrences."""
    selected_k = k[indices]
    selected_v = v[indices]
    logits = selected_k @ q / math.sqrt(q.numel())
    return torch.softmax(logits, dim=0) @ selected_v


def log_multiplicity_reference(q: Tensor, k: Tensor, v: Tensor, indices: Tensor) -> Tensor:
    """Equivalent deduplicated reference using an additive log occurrence count."""
    unique, counts = torch.unique(indices, sorted=True, return_counts=True)
    logits = k[unique] @ q / math.sqrt(q.numel())
    logits = logits + counts.to(logits.dtype).log()
    return torch.softmax(logits, dim=0) @ v[unique]


def _apply_rope(x: Tensor, positions: Tensor) -> Tensor:
    head_dim = x.shape[-1]
    rotary_dim = head_dim - head_dim % 2
    if rotary_dim == 0:
        return x
    frequencies = torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32)
    frequencies = 1.0 / (10000.0 ** (frequencies / rotary_dim))
    angles = positions.to(torch.float32).unsqueeze(-1) * frequencies
    cos = angles.cos().to(x.dtype).view(1, 1, -1, rotary_dim // 2)
    sin = angles.sin().to(x.dtype).view(1, 1, -1, rotary_dim // 2)
    paired = x[..., :rotary_dim].reshape(*x.shape[:-1], rotary_dim // 2, 2)
    even, odd = paired.unbind(-1)
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    rotated = rotated.flatten(-2)
    if rotary_dim == head_dim:
        return rotated
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        if config.kind == "qkvs" and config.router_mode == "learned":
            self.s_proj: nn.Module | None = nn.Linear(config.d_model, config.n_windows, bias=True)
        else:
            self.s_proj = None

        if config.kind == "qkvs" and config.router_mode == "unconditioned":
            self.router_logits = nn.Parameter(torch.zeros(config.n_windows))
        else:
            self.register_parameter("router_logits", None)

        if config.kind == "qkvs" and config.router_mode == "random":
            self.register_buffer("fixed_addresses", torch.rand(config.n_windows))
        else:
            self.register_buffer("fixed_addresses", None)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        return x.view(batch, time, self.config.n_heads, self.config.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, time, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, time, self.config.d_model)

    def _addresses(self, hidden: Tensor) -> Tensor | None:
        if self.config.kind != "qkvs":
            return None
        if self.config.router_mode == "learned":
            assert self.s_proj is not None
            # Address precision determines which discrete block is gathered. Keep S in FP32 even
            # when Q/K/V run under autocast; BF16 cannot represent enough bins at long contexts.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                return torch.sigmoid(self.s_proj(hidden.float()))
        if self.config.router_mode == "unconditioned":
            assert self.router_logits is not None
            return (
                torch.sigmoid(self.router_logits)
                .view(1, 1, -1)
                .expand(hidden.shape[0], hidden.shape[1], -1)
            )
        assert self.fixed_addresses is not None
        return self.fixed_addresses.view(1, 1, -1).expand(hidden.shape[0], hidden.shape[1], -1)

    def _selected(self, q: Tensor, k: Tensor, v: Tensor, selection: Selection) -> Tensor:
        if self.config.sparse_backend == "materialized":
            return selected_attention(q, k, v, selection)
        return streaming_selected_attention(
            q,
            k,
            v,
            selection,
            query_chunk_size=self.config.query_chunk_size,
            candidate_tile_size=self.config.candidate_tile_size,
            checkpoint_chunks=self.config.checkpoint_sparse and self.training,
        )

    @staticmethod
    def _apply_override(addresses: Tensor, override: Tensor | None) -> Tensor:
        if override is None:
            return addresses
        if override.shape != addresses.shape:
            raise ValueError("address override shape must match predicted addresses")
        return torch.where(torch.isfinite(override), override, addresses)

    def forward(
        self,
        hidden: Tensor,
        address_override: Tensor | None = None,
        retrieval_mask: Tensor | None = None,
        historical_window_mask: Tensor | None = None,
    ) -> AttentionOutput:
        batch, time, _ = hidden.shape
        positions = torch.arange(time, device=hidden.device)
        q = _apply_rope(self._split_heads(self.q_proj(hidden)), positions)
        k = _apply_rope(self._split_heads(self.k_proj(hidden)), positions)
        v = self._split_heads(self.v_proj(hidden))
        addresses = self._addresses(hidden)

        selection: Selection | None = None
        if self.config.kind == "dense":
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.config.dropout if self.training else 0.0,
            )
        elif self.config.kind == "rolling":
            selection = build_rolling_selection(
                batch, time, self.config.dense_window, hidden.device
            )
            attended = self._selected(q, k, v, selection)
        else:
            assert addresses is not None
            routed_addresses = self._apply_override(addresses, address_override)
            selection = build_selection(
                routed_addresses,
                dense_window=self.config.dense_window,
                retrieval_budget=self.config.retrieval_budget,
                block_size=self.config.block_size,
                address_stride=self.config.stride,
            )
            if retrieval_mask is not None:
                if retrieval_mask.shape != (batch, time):
                    raise ValueError("retrieval mask must have shape [batch, time]")
                below_threshold = (
                    torch.arange(time, device=hidden.device) + 1
                    <= self.config.dense_window + self.config.retrieval_budget
                )
                retrieval_enabled = retrieval_mask | below_threshold.unsqueeze(0)
                valid = selection.valid.clone()
                valid[..., : self.config.retrieval_budget] &= retrieval_enabled.unsqueeze(-1)
                selection = Selection(selection.indices, valid)
            if historical_window_mask is not None:
                expected = (batch, time, self.config.n_windows)
                if historical_window_mask.shape != expected:
                    raise ValueError(f"historical window mask must have shape {expected}")
                token_mask = historical_window_mask.repeat_interleave(
                    self.config.block_size, dim=-1
                )
                valid = selection.valid.clone()
                valid[..., : self.config.retrieval_budget] &= token_mask
                selection = Selection(selection.indices, valid)
            threshold = min(time, self.config.dense_window + self.config.retrieval_budget)
            dense_prefix = F.scaled_dot_product_attention(
                q[:, :, :threshold],
                k[:, :, :threshold],
                v[:, :, :threshold],
                is_causal=True,
                dropout_p=self.config.dropout if self.training else 0.0,
            )
            if threshold == time:
                attended = dense_prefix
            else:
                sparse_selection = Selection(
                    selection.indices[:, threshold:], selection.valid[:, threshold:]
                )
                sparse_suffix = self._selected(q[:, :, threshold:], k, v, sparse_selection)
                attended = torch.cat((dense_prefix, sparse_suffix), dim=2)

        output = self.o_proj(self._merge_heads(attended))
        return AttentionOutput(hidden=output, addresses=addresses, selection=selection)

    def step(
        self,
        hidden: Tensor,
        cache: KVCache | None,
        cache_capacity: int,
        address_override: Tensor | None = None,
    ) -> tuple[AttentionOutput, KVCache]:
        if hidden.shape[1] != 1:
            raise ValueError("incremental attention expects exactly one token")
        position = 0 if cache is None else cache.length
        positions = torch.tensor([position], device=hidden.device)
        q = _apply_rope(self._split_heads(self.q_proj(hidden)), positions)
        new_k = _apply_rope(self._split_heads(self.k_proj(hidden)), positions)
        new_v = self._split_heads(self.v_proj(hidden))
        if cache is None:
            if cache_capacity <= 0:
                raise ValueError("cache capacity must be positive")
            shape = (*new_k.shape[:2], cache_capacity, new_k.shape[-1])
            key_storage = torch.empty(shape, device=new_k.device, dtype=new_k.dtype)
            value_storage = torch.empty(shape, device=new_v.device, dtype=new_v.dtype)
            cache = KVCache(key_storage, value_storage, 0)
        if position >= cache.keys.shape[2]:
            raise ValueError("KV cache capacity exceeded")
        # Incremental decoding is an inference path. Append into a fresh slot; historical entries
        # are never overwritten or recomputed.
        cache.keys[:, :, position : position + 1].copy_(new_k)
        cache.values[:, :, position : position + 1].copy_(new_v)
        updated_cache = KVCache(cache.keys, cache.values, position + 1)
        k = updated_cache.keys[:, :, : updated_cache.length]
        v = updated_cache.values[:, :, : updated_cache.length]
        addresses = self._addresses(hidden)
        selection: Selection | None = None

        if self.config.kind == "dense":
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        elif self.config.kind == "rolling":
            start = max(0, position - self.config.dense_window + 1)
            indices = torch.arange(start, position + 1, device=hidden.device)
            indices = indices.view(1, 1, -1).expand(hidden.shape[0], -1, -1)
            selection = Selection(indices, torch.ones_like(indices, dtype=torch.bool))
            attended = selected_attention(q, k, v, selection)
        else:
            assert addresses is not None
            routed = self._apply_override(addresses, address_override)
            selection = build_query_selection(
                routed[:, 0],
                position,
                dense_window=self.config.dense_window,
                retrieval_budget=self.config.retrieval_budget,
                block_size=self.config.block_size,
                address_stride=self.config.stride,
            )
            attended = selected_attention(q, k, v, selection)

        output = self.o_proj(self._merge_heads(attended))
        return AttentionOutput(output, addresses, selection), updated_cache
