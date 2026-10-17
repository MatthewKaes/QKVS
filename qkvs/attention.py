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
    logit_bias: Tensor | None = None  # [batch, queries, slots]


@dataclass(frozen=True)
class AttentionOutput:
    hidden: Tensor
    addresses: Tensor | None
    selection: Selection | None
    router_logits: Tensor | None = None
    router_offsets: Tensor | None = None


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
    dense_threshold_retrieval_budget: int | None = None,
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
        slots = torch.arange(dense_window + retrieval_budget, device=device).view(1, 1, -1)
        dense_indices = slots.expand(batch, queries, -1)
        dense_valid = dense_indices <= query_positions
        threshold_budget = (
            retrieval_budget
            if dense_threshold_retrieval_budget is None
            else dense_threshold_retrieval_budget
        )
        use_dense = (query_positions + 1) <= dense_window + threshold_budget
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


def build_soft_neighbor_selection(
    addresses: Tensor,
    offsets: Tensor,
    *,
    dense_window: int,
    retrieval_budget: int,
    block_size: int,
    address_stride: int,
) -> Selection:
    """Retrieve adjacent windows with differentiable occurrence weights during training."""
    if addresses.shape != offsets.shape:
        raise ValueError("soft-neighbor addresses and offsets must have matching shapes")
    positions = torch.arange(addresses.shape[1], device=addresses.device)
    history_end = (positions - dense_window + 1).clamp_min(0)
    address_count = torch.div(
        history_end + address_stride - 1,
        address_stride,
        rounding_mode="floor",
    ).clamp_min(1)
    base_cells = torch.floor(
        addresses.clamp(0.0, 1.0 - torch.finfo(addresses.dtype).eps) * address_count.view(1, -1, 1)
    ).to(torch.long)
    neighbor_cells = torch.minimum(base_cells + 1, address_count.view(1, -1, 1) - 1)
    cells = torch.stack((base_cells, neighbor_cells), dim=-1).flatten(-2)
    expanded_addresses = (cells.to(torch.float32) + 0.5) / address_count.view(1, -1, 1)
    selection = _selection_for_positions(
        expanded_addresses,
        positions,
        dense_window=dense_window,
        retrieval_budget=2 * retrieval_budget,
        block_size=block_size,
        address_stride=address_stride,
        full_below_threshold=True,
        dense_threshold_retrieval_budget=retrieval_budget,
    )
    weights = torch.stack((1.0 - offsets, offsets), dim=-1).flatten(-2)
    historical_bias = weights.clamp_min(1e-8).log().repeat_interleave(block_size, dim=-1)
    recent_bias = torch.zeros(
        *historical_bias.shape[:-1],
        dense_window,
        device=addresses.device,
        dtype=historical_bias.dtype,
    )
    logit_bias = torch.cat((historical_bias, recent_bias), dim=-1)
    use_dense = (positions + 1) <= dense_window + retrieval_budget
    logit_bias = torch.where(use_dense.view(1, -1, 1), 0.0, logit_bias)
    return Selection(selection.indices, selection.valid, logit_bias)


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
    if selection.logit_bias is not None:
        logits = logits + selection.logit_bias.unsqueeze(1).to(logits.dtype)
    logits = logits.masked_fill(~selection.valid.unsqueeze(1), -torch.inf)
    weights = torch.softmax(logits, dim=-1)
    return torch.einsum("bhqm,bhqmd->bhqd", weights, selected_v)


def _streaming_attention_chunk(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    indices: Tensor,
    valid: Tensor,
    logit_bias: Tensor,
    candidate_tile_size: int,
) -> Tensor:
    """Online softmax over candidate tiles without a full selected-K/V tensor."""
    batch, heads, queries, query_dim = q.shape
    value_dim = v.shape[-1]
    running_max = torch.full(
        (batch, heads, queries),
        -torch.inf,
        device=q.device,
        dtype=torch.float32,
    )
    running_sum = torch.zeros_like(running_max)
    accumulator = torch.zeros(
        (batch, heads, queries, value_dim),
        device=q.device,
        dtype=torch.float32,
    )
    scale = 1.0 / math.sqrt(query_dim)

    for start in range(0, indices.shape[-1], candidate_tile_size):
        end = min(indices.shape[-1], start + candidate_tile_size)
        tile_indices = indices[..., start:end]
        tile_valid = valid[..., start:end]
        tile_bias = logit_bias[..., start:end]
        selected_k = _gather_cache(k, tile_indices)
        selected_v = _gather_cache(v, tile_indices)
        logits = torch.einsum("bhqd,bhqmd->bhqm", q, selected_k).float() * scale
        logits = logits + tile_bias.unsqueeze(1).float()
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
    logit_bias = selection.logit_bias
    if logit_bias is None:
        logit_bias = torch.zeros_like(selection.indices, dtype=q.dtype)
    for start in range(0, q.shape[2], query_chunk_size):
        end = min(q.shape[2], start + query_chunk_size)
        arguments = (
            q[:, :, start:end],
            k,
            v,
            selection.indices[:, start:end],
            selection.valid[:, start:end],
            logit_bias[:, start:end],
        )

        def attend_chunk(
            chunk_q: Tensor,
            all_k: Tensor,
            all_v: Tensor,
            chunk_indices: Tensor,
            chunk_valid: Tensor,
            chunk_logit_bias: Tensor,
        ) -> Tensor:
            return _streaming_attention_chunk(
                chunk_q,
                all_k,
                all_v,
                chunk_indices,
                chunk_valid,
                chunk_logit_bias,
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


def _chunked_sdpa_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    indices: Tensor,
    valid: Tensor,
    logit_bias: Tensor,
) -> Tensor:
    """Materialize one bounded query chunk and dispatch its selections through SDPA."""
    selected_k = _gather_cache(k, indices)
    selected_v = _gather_cache(v, indices)
    batch, heads, queries, query_dim = q.shape
    value_dim = v.shape[-1]
    candidates = indices.shape[-1]
    flat_q = q.permute(0, 2, 1, 3).reshape(batch * queries, heads, 1, query_dim)
    flat_k = selected_k.permute(0, 2, 1, 3, 4).reshape(
        batch * queries, heads, candidates, query_dim
    )
    flat_v = selected_v.permute(0, 2, 1, 3, 4).reshape(
        batch * queries, heads, candidates, value_dim
    )
    mask = logit_bias.reshape(batch * queries, 1, 1, candidates).to(flat_q.dtype)
    mask = mask.masked_fill(
        ~valid.reshape(batch * queries, 1, 1, candidates),
        -torch.inf,
    )
    attended = F.scaled_dot_product_attention(
        flat_q,
        flat_k,
        flat_v,
        attn_mask=mask,
    )
    return attended.reshape(batch, queries, heads, value_dim).permute(0, 2, 1, 3)


def chunked_selected_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    selection: Selection,
    *,
    query_chunk_size: int,
    checkpoint_chunks: bool,
) -> Tensor:
    """Use fused SDPA over bounded materialized query chunks."""
    outputs: list[Tensor] = []
    logit_bias = selection.logit_bias
    if logit_bias is None:
        logit_bias = torch.zeros_like(selection.indices, dtype=q.dtype)
    for start in range(0, q.shape[2], query_chunk_size):
        end = min(q.shape[2], start + query_chunk_size)
        arguments = (
            q[:, :, start:end],
            k,
            v,
            selection.indices[:, start:end],
            selection.valid[:, start:end],
            logit_bias[:, start:end],
        )
        if checkpoint_chunks and torch.is_grad_enabled():
            output = checkpoint(
                _chunked_sdpa_attention,
                *arguments,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            output = _chunked_sdpa_attention(*arguments)
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


def router_coordinate_features(
    positions: Tensor,
    *,
    dense_window: int,
    address_stride: int,
    n_frequencies: int = 4,
) -> Tensor:
    """Encode absolute query position and the size of its eligible address space for S."""
    position_values = positions.to(torch.float32)
    history_end = (position_values - dense_window + 1).clamp_min(0.0)
    address_count = torch.ceil(history_end / address_stride).clamp_min(1.0)
    frequencies = 2.0 ** torch.arange(
        n_frequencies,
        device=positions.device,
        dtype=torch.float32,
    )
    position_angles = (
        2.0 * torch.pi * position_values.unsqueeze(-1) / (address_stride * frequencies)
    )
    count_angles = 2.0 * torch.pi * address_count.unsqueeze(-1) / frequencies
    history_ratio = history_end / (position_values + 1.0)
    valid_history = (history_end > 0).to(torch.float32)
    return torch.cat(
        (
            torch.log2(position_values + 1.0).unsqueeze(-1) / 32.0,
            torch.log2(address_count + 1.0).unsqueeze(-1) / 16.0,
            history_ratio.unsqueeze(-1),
            valid_history.unsqueeze(-1),
            torch.sin(position_angles),
            torch.cos(position_angles),
            torch.sin(count_angles),
            torch.cos(count_angles),
        ),
        dim=-1,
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        if config.unrotated_address_matching:
            self.address_q_proj: nn.Module | None = nn.Linear(
                config.d_model, config.d_model, bias=False
            )
            self.address_k_proj: nn.Module | None = nn.Linear(
                config.d_model, config.d_model, bias=False
            )
        else:
            self.address_q_proj = None
            self.address_k_proj = None

        if config.kind == "qkvs" and config.router_position_encoding == "absolute_fourier":
            position_width = 4 + 4 * 4
            self.router_position_projection: nn.Module | None = nn.Sequential(
                nn.Linear(position_width, config.d_model),
                nn.SiLU(),
                nn.Linear(config.d_model, config.d_model),
            )
        else:
            self.router_position_projection = None

        if config.kind == "qkvs" and config.router_pointer_input:
            self.router_pointer_projection: nn.Module | None = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.SiLU(),
                nn.Linear(config.d_model, config.d_model),
            )
        else:
            self.router_pointer_projection = None

        if config.kind == "qkvs" and config.router_mode == "learned":
            router_capacity = config.router_capacity or config.n_windows
            router_outputs = (
                router_capacity
                if config.router_output_encoding == "scalar"
                else router_capacity * 2 * config.router_radix
            )
            if config.router_mlp_ratio:
                router_hidden = config.d_model * config.router_mlp_ratio
                self.s_proj: nn.Module | None = nn.Sequential(
                    nn.Linear(config.d_model, router_hidden),
                    nn.SiLU(),
                    nn.Linear(router_hidden, router_outputs),
                )
            else:
                self.s_proj = nn.Linear(config.d_model, router_outputs, bias=True)
        else:
            self.s_proj = None

        if config.kind == "qkvs" and config.router_mode == "learned" and config.router_local_offset:
            router_capacity = config.router_capacity or config.n_windows
            self.s_offset_proj: nn.Module | None = nn.Linear(config.d_model, router_capacity)
        else:
            self.s_offset_proj = None

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

    def _router_input(
        self,
        hidden: Tensor,
        positions: Tensor | None,
        router_context: Tensor | None,
    ) -> Tensor:
        router_input = hidden.detach() if self.config.router_detach_input else hidden
        router_input = router_input.float()
        if self.router_position_projection is not None:
            if positions is None:
                positions = torch.arange(hidden.shape[1], device=hidden.device)
            if positions.shape != (hidden.shape[1],):
                raise ValueError("router positions must contain one entry per token")
            coordinates = router_coordinate_features(
                positions,
                dense_window=self.config.dense_window,
                address_stride=self.config.stride,
            )
            router_input = router_input + self.router_position_projection(coordinates)
        if self.router_pointer_projection is not None:
            if router_context is None or router_context.shape != hidden.shape:
                raise ValueError("direct router pointer input must match the hidden-state shape")
            router_input = router_input + self.router_pointer_projection(router_context.float())
        return router_input

    def _router_logits(
        self,
        hidden: Tensor,
        positions: Tensor | None = None,
        router_context: Tensor | None = None,
    ) -> Tensor | None:
        if self.config.kind != "qkvs":
            return None
        if self.config.router_mode == "learned":
            assert self.s_proj is not None
            # Address precision determines which discrete block is gathered. Keep S in FP32 even
            # when Q/K/V run under autocast; BF16 cannot represent enough bins at long contexts.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                router_input = self._router_input(hidden, positions, router_context)
                logits = self.s_proj(router_input)
                if self.config.router_output_encoding == "scalar":
                    return logits[..., : self.config.n_windows]
                router_capacity = self.config.router_capacity or self.config.n_windows
                logits = logits.view(
                    *logits.shape[:-1],
                    router_capacity,
                    2,
                    self.config.router_radix,
                )
                return logits[..., : self.config.n_windows, :, :]
        return None

    def _router_offsets(
        self,
        hidden: Tensor,
        positions: Tensor | None = None,
        router_context: Tensor | None = None,
    ) -> Tensor | None:
        if self.s_offset_proj is None:
            return None
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            router_input = self._router_input(hidden, positions, router_context)
            return torch.sigmoid(self.s_offset_proj(router_input))[..., : self.config.n_windows]

    def _decode_router_logits(
        self,
        logits: Tensor,
        positions: Tensor,
        offsets: Tensor | None = None,
    ) -> Tensor:
        if self.config.router_output_encoding == "scalar":
            return torch.sigmoid(logits)
        coarse = logits[..., 0, :].argmax(dim=-1)
        fine = logits[..., 1, :].argmax(dim=-1)
        cells = coarse * self.config.router_radix + fine
        if offsets is not None:
            cells = cells + (offsets > 0.5).to(torch.long)
        history_end = (positions - self.config.dense_window + 1).clamp_min(0)
        address_count = torch.div(
            history_end + self.config.stride - 1,
            self.config.stride,
            rounding_mode="floor",
        ).clamp_min(1)
        maximum_cells = self.config.router_radix * self.config.router_radix
        if int(address_count.max()) > maximum_cells:
            raise ValueError("hierarchical router radix cannot represent this address space")
        cells = torch.minimum(cells, address_count.view(1, -1, 1) - 1)
        return (cells.to(torch.float32) + 0.5) / address_count.view(1, -1, 1)

    def _addresses(
        self,
        hidden: Tensor,
        positions: Tensor | None = None,
        router_context: Tensor | None = None,
    ) -> Tensor | None:
        if self.config.kind != "qkvs":
            return None
        if self.config.router_mode == "learned":
            if positions is None:
                positions = torch.arange(hidden.shape[1], device=hidden.device)
            logits = self._router_logits(hidden, positions, router_context)
            assert logits is not None
            return self._decode_router_logits(logits, positions)
        if self.config.router_mode == "unconditioned":
            assert self.router_logits is not None
            return (
                torch.sigmoid(self.router_logits)
                .view(1, 1, -1)
                .expand(hidden.shape[0], hidden.shape[1], -1)
            )
        assert self.fixed_addresses is not None
        return self.fixed_addresses.view(1, 1, -1).expand(hidden.shape[0], hidden.shape[1], -1)

    def router_output_projection(self) -> nn.Linear | None:
        if isinstance(self.s_proj, nn.Linear):
            return self.s_proj
        if isinstance(self.s_proj, nn.Sequential):
            projection = self.s_proj[-1]
            assert isinstance(projection, nn.Linear)
            return projection
        return None

    def _selected(self, q: Tensor, k: Tensor, v: Tensor, selection: Selection) -> Tensor:
        if self.config.sparse_backend == "materialized":
            return selected_attention(q, k, v, selection)
        if self.config.sparse_backend == "chunked":
            return chunked_selected_attention(
                q,
                k,
                v,
                selection,
                query_chunk_size=self.config.query_chunk_size,
                checkpoint_chunks=self.config.checkpoint_sparse and self.training,
            )
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
        router_context: Tensor | None = None,
    ) -> AttentionOutput:
        batch, time, _ = hidden.shape
        positions = torch.arange(time, device=hidden.device)
        q = _apply_rope(self._split_heads(self.q_proj(hidden)), positions)
        k = _apply_rope(self._split_heads(self.k_proj(hidden)), positions)
        v = self._split_heads(self.v_proj(hidden))
        if self.address_q_proj is not None and self.address_k_proj is not None:
            if router_context is None:
                raise ValueError("unrotated address matching requires direct pointer context")
            q = torch.cat((q, self._split_heads(self.address_q_proj(router_context))), dim=-1)
            k = torch.cat((k, self._split_heads(self.address_k_proj(router_context))), dim=-1)
        router_logits = self._router_logits(hidden, positions, router_context)
        router_offsets = self._router_offsets(hidden, positions, router_context)
        addresses = (
            self._decode_router_logits(router_logits, positions, router_offsets)
            if router_logits is not None
            else self._addresses(hidden, positions, router_context)
        )

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
            if self.training and self.config.soft_neighbor_training:
                assert router_logits is not None and router_offsets is not None
                base_addresses = self._decode_router_logits(router_logits, positions)
                routed_addresses = self._apply_override(base_addresses, address_override)
                effective_offsets = router_offsets
                if address_override is not None:
                    effective_offsets = torch.where(
                        torch.isfinite(address_override),
                        torch.zeros_like(effective_offsets),
                        effective_offsets,
                    )
                selection = build_soft_neighbor_selection(
                    routed_addresses,
                    effective_offsets,
                    dense_window=self.config.dense_window,
                    retrieval_budget=self.config.retrieval_budget,
                    block_size=self.config.block_size,
                    address_stride=self.config.stride,
                )
            else:
                routed_addresses = self._apply_override(addresses, address_override)
                selection = build_selection(
                    routed_addresses,
                    dense_window=self.config.dense_window,
                    retrieval_budget=self.config.retrieval_budget,
                    block_size=self.config.block_size,
                    address_stride=self.config.stride,
                )
            historical_budget = selection.indices.shape[-1] - self.config.dense_window
            if retrieval_mask is not None:
                if retrieval_mask.shape != (batch, time):
                    raise ValueError("retrieval mask must have shape [batch, time]")
                below_threshold = (
                    torch.arange(time, device=hidden.device) + 1
                    <= self.config.dense_window + self.config.retrieval_budget
                )
                retrieval_enabled = retrieval_mask | below_threshold.unsqueeze(0)
                valid = selection.valid.clone()
                valid[..., :historical_budget] &= retrieval_enabled.unsqueeze(-1)
                selection = Selection(selection.indices, valid, selection.logit_bias)
            if historical_window_mask is not None:
                expected = (batch, time, self.config.n_windows)
                if historical_window_mask.shape != expected:
                    raise ValueError(f"historical window mask must have shape {expected}")
                window_mask = historical_window_mask
                if historical_budget == 2 * self.config.retrieval_budget:
                    window_mask = window_mask.repeat_interleave(2, dim=-1)
                token_mask = window_mask.repeat_interleave(self.config.block_size, dim=-1)
                valid = selection.valid.clone()
                valid[..., :historical_budget] &= token_mask
                selection = Selection(selection.indices, valid, selection.logit_bias)
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
                    selection.indices[:, threshold:],
                    selection.valid[:, threshold:],
                    (None if selection.logit_bias is None else selection.logit_bias[:, threshold:]),
                )
                sparse_suffix = self._selected(q[:, :, threshold:], k, v, sparse_selection)
                attended = torch.cat((dense_prefix, sparse_suffix), dim=2)

        output = self.o_proj(self._merge_heads(attended))
        return AttentionOutput(
            hidden=output,
            addresses=addresses,
            selection=selection,
            router_logits=router_logits,
            router_offsets=router_offsets,
        )

    def step(
        self,
        hidden: Tensor,
        cache: KVCache | None,
        cache_capacity: int,
        address_override: Tensor | None = None,
        router_context: Tensor | None = None,
    ) -> tuple[AttentionOutput, KVCache]:
        if hidden.shape[1] != 1:
            raise ValueError("incremental attention expects exactly one token")
        position = 0 if cache is None else cache.length
        positions = torch.tensor([position], device=hidden.device)
        q = _apply_rope(self._split_heads(self.q_proj(hidden)), positions)
        new_k = _apply_rope(self._split_heads(self.k_proj(hidden)), positions)
        new_v = self._split_heads(self.v_proj(hidden))
        if self.address_q_proj is not None and self.address_k_proj is not None:
            if router_context is None:
                raise ValueError("unrotated address matching requires direct pointer context")
            q = torch.cat((q, self._split_heads(self.address_q_proj(router_context))), dim=-1)
            new_k = torch.cat(
                (new_k, self._split_heads(self.address_k_proj(router_context))),
                dim=-1,
            )
        if cache is None:
            if cache_capacity <= 0:
                raise ValueError("cache capacity must be positive")
            key_shape = (*new_k.shape[:2], cache_capacity, new_k.shape[-1])
            value_shape = (*new_v.shape[:2], cache_capacity, new_v.shape[-1])
            key_storage = torch.empty(key_shape, device=new_k.device, dtype=new_k.dtype)
            value_storage = torch.empty(value_shape, device=new_v.device, dtype=new_v.dtype)
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
        router_logits = self._router_logits(hidden, positions, router_context)
        router_offsets = self._router_offsets(hidden, positions, router_context)
        addresses = (
            self._decode_router_logits(router_logits, positions, router_offsets)
            if router_logits is not None
            else self._addresses(hidden, positions, router_context)
        )
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
        return AttentionOutput(
            output,
            addresses,
            selection,
            router_logits,
            router_offsets,
        ), updated_cache
