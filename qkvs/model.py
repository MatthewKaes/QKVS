from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from qkvs.selection import (
    LearnedHeatFieldPolicy,
    PolicyState,
    SelectionContext,
    SelectionPolicy,
    resolve_proposals,
)


@dataclass(frozen=True)
class TokenStream:
    stream_id: str
    tokens: tuple[int, ...]
    support_positions: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) < 2:
            raise ValueError("token streams need at least two tokens")
        if len(self.support_positions) != len(self.tokens) - 1:
            raise ValueError("support positions must align with next-token scores")


@dataclass(frozen=True)
class DensePrefillSnapshot:
    """Immutable continuation state after the dense prefix [0, activation_tokens)."""

    identity: dict[str, Any]
    prefix_tokens: tuple[int, ...]
    next_position: int
    key_cache: tuple[Tensor, ...]
    value_cache: tuple[Tensor, ...]
    dense_losses: tuple[float, ...]
    dense_predictions: tuple[int, ...]
    boundary_logits: Tensor


@dataclass
class HeatfieldRolloutState:
    """Mutable causal continuation owned by one training stream."""

    stream_id: str
    next_position: int
    key_cache: list[_LayerKVCache]
    value_cache: list[_LayerKVCache]
    policy_states: list[PolicyState]


@dataclass(frozen=True)
class AccessRecord:
    token_position: int
    layer: int
    rolling_range: tuple[int, int]
    historical_windows_encoded: bytes
    attention_mode: str
    historical_occurrences: int
    historical_unique_tokens: int
    duplicate_historical_occurrences: int
    rolling_overlap_occurrences: int
    proposal_centers: tuple[float, ...]
    proposal_spreads: tuple[float, ...]
    proposal_weights: tuple[float, ...]
    proposal_active: tuple[bool, ...]
    support_position: int | None
    history_end: int
    cache_length: int
    gathered_key_occurrences: int
    gathered_value_occurrences: int

    @property
    def rolling_indices(self) -> tuple[int, ...]:
        return tuple(range(*self.rolling_range))

    @property
    def historical_indices(self) -> tuple[int, ...]:
        return tuple(
            index for start, stop in self.historical_windows for index in range(start, stop)
        )

    @property
    def historical_windows(self) -> tuple[tuple[int, int], ...]:
        return _unpack_windows(self.historical_windows_encoded)

    @property
    def actual_indices(self) -> tuple[int, ...]:
        return (*self.historical_indices, *self.rolling_indices)

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_position": self.token_position,
            "layer": self.layer,
            "rolling_range": self.rolling_range,
            "historical_windows_varint_base64": base64.b64encode(
                self.historical_windows_encoded
            ).decode("ascii"),
            "attention_mode": self.attention_mode,
            "historical_occurrences": self.historical_occurrences,
            "historical_unique_tokens": self.historical_unique_tokens,
            "duplicate_historical_occurrences": self.duplicate_historical_occurrences,
            "rolling_overlap_occurrences": self.rolling_overlap_occurrences,
            "proposal_centers": self.proposal_centers,
            "proposal_spreads": self.proposal_spreads,
            "proposal_weights": self.proposal_weights,
            "proposal_active": self.proposal_active,
            "support_position": self.support_position,
            "history_end": self.history_end,
            "cache_length": self.cache_length,
            "gathered_key_occurrences": self.gathered_key_occurrences,
            "gathered_value_occurrences": self.gathered_value_occurrences,
        }


@dataclass(frozen=True)
class TokenScore:
    position: int
    target: int
    loss: float
    prediction: int
    accesses: tuple[AccessRecord, ...]
    phase: str = "qkvs"

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "target": self.target,
            "loss": self.loss,
            "prediction": self.prediction,
            "accesses": [access.as_dict() for access in self.accesses],
            "phase": self.phase,
        }


def _access_record(
    *,
    position: int,
    layer: int,
    rolling: tuple[int, ...],
    historical: tuple[int, ...],
    historical_windows: tuple[tuple[int, int], ...] | None = None,
    attention_mode: str = "bounded",
    proposal_centers: tuple[float, ...],
    proposal_spreads: tuple[float, ...],
    proposal_weights: tuple[float, ...],
    proposal_active: tuple[bool, ...],
    support_position: int | None,
    history_end: int,
    cache_length: int,
    gathered_key_occurrences: int,
    gathered_value_occurrences: int,
    historical_occurrence_count: int | None = None,
    historical_unique_count: int | None = None,
) -> AccessRecord:
    if not rolling:
        raise ValueError("an attention access must include working memory")
    if historical_windows is None:
        ranges: list[tuple[int, int]] = []
        start = None
        previous = None
        for index in historical:
            if start is None or previous is None or index != previous + 1:
                if start is not None and previous is not None:
                    ranges.append((start, previous + 1))
                start = index
            previous = index
        if start is not None and previous is not None:
            ranges.append((start, previous + 1))
        historical_windows = tuple(ranges)
    rolling_set = set(rolling)
    unique_historical = len(set(historical))
    occurrence_count = (
        len(historical) if historical_occurrence_count is None else historical_occurrence_count
    )
    unique_count = unique_historical if historical_unique_count is None else historical_unique_count
    return AccessRecord(
        token_position=position,
        layer=layer,
        rolling_range=(rolling[0], rolling[-1] + 1),
        historical_windows_encoded=_pack_windows(historical_windows),
        attention_mode=attention_mode,
        historical_occurrences=occurrence_count,
        historical_unique_tokens=unique_count,
        duplicate_historical_occurrences=occurrence_count - unique_count,
        rolling_overlap_occurrences=sum(index in rolling_set for index in historical),
        proposal_centers=proposal_centers,
        proposal_spreads=proposal_spreads,
        proposal_weights=proposal_weights,
        proposal_active=proposal_active,
        support_position=support_position,
        history_end=history_end,
        cache_length=cache_length,
        gathered_key_occurrences=gathered_key_occurrences,
        gathered_value_occurrences=gathered_value_occurrences,
    )


def _encode_varint(value: int, output: bytearray) -> None:
    if value < 0:
        raise ValueError("window coordinates must be nonnegative")
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)


def _decode_varint(encoded: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(encoded):
        byte = encoded[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            break
    raise ValueError("invalid packed historical window")


def _pack_windows(windows: tuple[tuple[int, int], ...]) -> bytes:
    encoded = bytearray()
    for start, stop in windows:
        if stop <= start:
            raise ValueError("historical windows must be nonempty")
        _encode_varint(start, encoded)
        _encode_varint(stop - start, encoded)
    return bytes(encoded)


def _unpack_windows(encoded: bytes) -> tuple[tuple[int, int], ...]:
    windows = []
    offset = 0
    while offset < len(encoded):
        start, offset = _decode_varint(encoded, offset)
        length, offset = _decode_varint(encoded, offset)
        if length <= 0:
            raise ValueError("packed historical windows must be nonempty")
        windows.append((start, start + length))
    return tuple(windows)


class _LayerKVCache:
    """Preallocated layer cache supporting O(selected occurrences) direct gathers."""

    def __init__(
        self,
        storage: Tensor,
        length: int = 0,
        *,
        writable: bool = True,
        overlay: list[Tensor] | None = None,
    ) -> None:
        if storage.ndim != 3 or length < 0 or length > storage.shape[1]:
            raise ValueError("invalid layer K/V cache storage")
        self.storage = storage
        self.length = length
        self.writable = writable
        self.overlay = overlay or []

    @classmethod
    def empty(
        cls,
        *,
        heads: int,
        capacity: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _LayerKVCache:
        return cls(torch.empty((heads, capacity, head_dim), device=device, dtype=dtype))

    @classmethod
    def from_prefix(
        cls,
        prefix: Tensor,
        *,
        capacity: int,
        device: torch.device,
    ) -> _LayerKVCache:
        prefix = prefix.to(device)
        cache = cls.empty(
            heads=prefix.shape[0],
            capacity=capacity,
            head_dim=prefix.shape[2],
            device=device,
            dtype=prefix.dtype,
        )
        cache.storage[:, : prefix.shape[1]].copy_(prefix)
        cache.length = prefix.shape[1]
        return cache

    def __len__(self) -> int:
        return self.length + len(self.overlay)

    def append(self, value: Tensor) -> None:
        if value.shape != self.storage[:, 0].shape:
            raise ValueError("layer K/V cache append is outside its fixed storage")
        if not self.writable:
            self.overlay.append(value)
            return
        if self.length >= self.storage.shape[1]:
            raise ValueError("layer K/V cache append is outside its fixed storage")
        self.storage[:, self.length].copy_(value)
        self.length += 1

    def gather(
        self,
        indices: tuple[int, ...] | Tensor | None,
        *,
        occurrence_limit: int | None,
    ) -> Tensor:
        if indices is None:
            if occurrence_limit is not None:
                raise ValueError("full-cache gather cannot claim a bounded occurrence limit")
            base = self.storage[:, : self.length]
            if not self.overlay:
                return base
            return torch.cat((base, torch.stack(self.overlay, dim=1)), dim=1)
        occurrence_count = len(indices)
        if occurrence_limit is not None and occurrence_count > occurrence_limit:
            raise ValueError("actual K/V gather exceeds the bounded occurrence limit")
        if isinstance(indices, Tensor):
            if indices.numel() == 0:
                raise ValueError("actual K/V gather must include the rolling window")
            index = indices.to(device=self.storage.device, dtype=torch.long)
        else:
            if not indices:
                raise ValueError("actual K/V gather must include the rolling window")
            if any(value < 0 or value >= len(self) for value in indices):
                raise ValueError("actual K/V gather index is outside the layer cache")
            index = torch.tensor(indices, device=self.storage.device, dtype=torch.long)
        if not self.overlay:
            return self.storage[:, : self.length].index_select(1, index)
        base_mask = index < self.length
        if bool(base_mask.all()):
            return self.storage[:, : self.length].index_select(1, index)
        if bool(base_mask.any()) and not bool(
            torch.equal(index, torch.sort(index, stable=True).values)
        ):
            raise ValueError("branched cache indices must keep base entries before overlays")
        pieces = []
        if bool(base_mask.any()):
            pieces.append(self.storage[:, : self.length].index_select(1, index[base_mask]))
        overlay_indices = (index[~base_mask] - self.length).detach().cpu().tolist()
        pieces.append(torch.stack([self.overlay[value] for value in overlay_indices], dim=1))
        return torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]

    def fork(self) -> _LayerKVCache:
        return _LayerKVCache(self.storage, self.length, writable=False)

    def snapshot(self) -> Tensor:
        if self.overlay:
            raise ValueError("branched K/V cache cannot be checkpointed")
        return self.storage[:, : self.length].detach().cpu().clone()


def _gather_cache_occurrences(
    cache: list[Tensor] | _LayerKVCache,
    indices: tuple[int, ...] | Tensor | None,
    *,
    occurrence_limit: int | None,
) -> Tensor:
    """Stack only executed cache occurrences, preserving order and duplicates."""

    if isinstance(cache, _LayerKVCache):
        return cache.gather(indices, occurrence_limit=occurrence_limit)
    if indices is None:
        if occurrence_limit is not None:
            raise ValueError("full-cache gather cannot claim a bounded occurrence limit")
        indices = tuple(range(len(cache)))

    if occurrence_limit is not None and len(indices) > occurrence_limit:
        raise ValueError("actual K/V gather exceeds the bounded occurrence limit")
    cache_length = len(cache)
    if any(index < 0 or index >= cache_length for index in indices):
        raise ValueError("actual K/V gather index is outside the layer cache")
    selected = [cache[index] for index in indices]
    if not selected:
        raise ValueError("actual K/V gather must include the rolling window")
    return torch.stack(selected, dim=1)


def _collective_soft_occupancy(
    soft_parameters: Tensor,
    *,
    history_end: int,
    window_width: int,
    beta: float,
) -> Tensor:
    if history_end < 0 or window_width <= 0 or beta <= 0.0:
        raise ValueError("soft occupancy geometry is invalid")
    if soft_parameters.ndim != 2 or soft_parameters.shape[1] != 4:
        raise ValueError("soft proposal parameters must have center, spread, weight, and gate")
    historical_positions = torch.arange(
        history_end,
        device=soft_parameters.device,
        dtype=soft_parameters.dtype,
    )
    centers = soft_parameters[:, 0:1]
    spreads = (window_width * soft_parameters[:, 1:2] / beta).clamp_min(0.5)
    importance = soft_parameters[:, 2:3]
    gates = soft_parameters[:, 3:4]
    distance = (historical_positions.unsqueeze(0) - centers) / spreads
    kernels = torch.exp(-0.5 * distance.square())
    proposal_occupancy = gates * (1.0 - torch.exp(-importance * kernels))
    proposal_occupancy = proposal_occupancy.clamp(0.0, 1.0 - 1e-6)
    return 1.0 - torch.prod(1.0 - proposal_occupancy, dim=0)


def _absolute_teacher_coverage(
    teacher_history: Tensor,
    collective_occupancy: Tensor,
) -> tuple[Tensor, Tensor]:
    if teacher_history.ndim != 1 or teacher_history.shape != collective_occupancy.shape:
        raise ValueError("teacher history and collective occupancy must align")
    teacher_history = teacher_history.to(collective_occupancy.dtype)
    covered = torch.sum(teacher_history * collective_occupancy)
    uncovered = torch.sum(teacher_history * (1.0 - collective_occupancy))
    return uncovered, covered


_TRAINING_SCALAR_METRICS = (
    "objective",
    "coverage_loss",
    "soft_next_token_loss",
    "rolling_next_token_loss",
    "hard_selected_next_token_loss",
    "hard_utility",
    "soft_utility",
    "soft_hard_utility_gap",
    "teacher_historical_attention_mass",
    "teacher_historical_attention_covered_mass",
    "teacher_historical_attention_uncovered_mass",
)


def _materialize_training_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return records
    proposal_shape = records[0]["proposal_parameters"].shape
    packed = (
        torch.stack(
            [
                torch.cat(
                    (
                        torch.stack(
                            [record[name].detach().float() for name in _TRAINING_SCALAR_METRICS]
                        ),
                        record["proposal_parameters"].detach().float().reshape(-1),
                    )
                )
                for record in records
            ]
        )
        .cpu()
        .tolist()
    )
    scalar_count = len(_TRAINING_SCALAR_METRICS)
    for record, values in zip(records, packed, strict=True):
        for name, value in zip(_TRAINING_SCALAR_METRICS, values[:scalar_count], strict=True):
            record[name] = value
        proposals = values[scalar_count:]
        record["proposal_parameters"] = [
            proposals[offset : offset + proposal_shape[1]]
            for offset in range(0, len(proposals), proposal_shape[1])
        ]
    return records


def _validate_transition(rolling_tokens: int, activation_tokens: int) -> None:
    if rolling_tokens <= 0:
        raise ValueError("rolling attention is mandatory")
    if activation_tokens < 0:
        raise ValueError("activation threshold cannot be negative")
    if activation_tokens and activation_tokens < rolling_tokens:
        raise ValueError("activation threshold must be at least the working-memory width")


class _ReaderLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.mlp_norm = nn.RMSNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 2, bias=False),
            nn.GELU(),
            nn.Linear(width * 2, width, bias=False),
        )


class FrozenCausalReader(nn.Module):
    """Small frozen transformer used to verify real bounded K/V access on CPU."""

    tokenizer = None

    def __init__(
        self,
        *,
        vocab_size: int,
        width: int,
        layers: int,
        heads: int,
        seed: int,
    ) -> None:
        super().__init__()
        if width <= 0 or layers <= 0 or heads <= 0 or width % heads:
            raise ValueError("reader width must be positive and divisible by its head count")
        self.vocab_size = vocab_size
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, width)
            self.layers = nn.ModuleList(_ReaderLayer(width) for _ in range(layers))
            self.final_norm = nn.RMSNorm(width)
            self.lm_head = nn.Linear(width, vocab_size, bias=False)
            self.apply(self._initialize)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def _heads(self, value: Tensor) -> Tensor:
        return value.view(self.heads, self.head_width)

    def identity(self) -> dict[str, Any]:
        return {
            "backend": "synthetic",
            "architecture": type(self).__name__,
            "vocab_size": self.vocab_size,
            "width": self.width,
            "layers": len(self.layers),
            "heads": self.heads,
            "seed": self.seed,
        }

    def execute(
        self,
        stream: TokenStream,
        policy: SelectionPolicy,
        *,
        rolling_tokens: int,
        historical_token_budget: int,
        window_width: int,
        device: torch.device,
        activation_tokens: int,
        dense_snapshot: DensePrefillSnapshot | None = None,
        snapshot_ownership: str = "evaluation",
        prefill_chunk_tokens: int = 128,
        proposal_trace_interval: int = 1,
    ) -> list[TokenScore]:
        del prefill_chunk_tokens, snapshot_ownership
        if dense_snapshot is not None:
            raise ValueError("synthetic reader does not accept Qwen2 dense prefill snapshots")
        _validate_transition(rolling_tokens, activation_tokens)
        if activation_tokens == 0:
            raise ValueError("bounded execution requires a positive activation threshold")
        if proposal_trace_interval <= 0:
            raise ValueError("proposal trace interval must be positive")
        self.to(device)
        key_cache: list[list[Tensor]] = [[] for _ in self.layers]
        value_cache: list[list[Tensor]] = [[] for _ in self.layers]
        policy_states = [PolicyState() for _ in self.layers]
        scores: list[TokenScore] = []
        with torch.inference_mode():
            for position, token in enumerate(stream.tokens[:-1]):
                hidden = self.embedding(torch.tensor(token, device=device))
                token_accesses: list[AccessRecord] = []
                lower_layer_selection = None
                for layer_index, layer in enumerate(self.layers):
                    normalized = layer.attention_norm(hidden)
                    query = self._heads(layer.query(normalized))
                    key_cache[layer_index].append(self._heads(layer.key(normalized)))
                    value_cache[layer_index].append(self._heads(layer.value(normalized)))

                    if position < activation_tokens:
                        rolling_start = 0
                        rolling = tuple(range(position + 1))
                        historical: tuple[int, ...] = ()
                        centers: tuple[float, ...] = ()
                        spreads: tuple[float, ...] = ()
                        weights: tuple[float, ...] = ()
                        active: tuple[bool, ...] = ()
                        occurrence_limit = None
                        historical_windows: tuple[tuple[int, int], ...] = ()
                        attention_mode = "dense"
                    else:
                        rolling_start = max(0, position - rolling_tokens + 1)
                        rolling = tuple(range(rolling_start, position + 1))
                        if getattr(policy, "full_qkv", False):
                            historical = tuple(range(rolling_start))
                            historical_windows = ((0, rolling_start),) if rolling_start else ()
                            centers = ()
                            spreads = ()
                            weights = ()
                            active = ()
                            occurrence_limit = None
                            attention_mode = "full_qkv"
                        else:
                            proposal_set = policy.propose(
                                SelectionContext(
                                    position=position,
                                    layer=layer_index,
                                    history_end=rolling_start,
                                    hidden=normalized,
                                    lower_layer=lower_layer_selection,
                                ),
                                policy_states[layer_index],
                            )
                            policy_states[layer_index] = proposal_set.state
                            lower_layer_selection = proposal_set.soft_parameters
                            selection = resolve_proposals(
                                proposal_set.proposals,
                                history_end=rolling_start,
                                window_width=window_width,
                                historical_token_budget=historical_token_budget,
                                beta=proposal_set.beta,
                            )
                            historical = selection.indices
                            historical_windows = tuple(
                                (window[0], window[-1] + 1)
                                for window in selection.windows
                                if window
                            )
                            centers = tuple(proposal.center for proposal in proposal_set.proposals)
                            spreads = tuple(proposal.spread for proposal in proposal_set.proposals)
                            weights = tuple(proposal.weight for proposal in proposal_set.proposals)
                            active = tuple(proposal.active for proposal in proposal_set.proposals)
                            occurrence_limit = rolling_tokens + historical_token_budget
                            attention_mode = "bounded"
                    actual_indices = (*historical, *rolling)
                    keys = _gather_cache_occurrences(
                        key_cache[layer_index],
                        actual_indices,
                        occurrence_limit=occurrence_limit,
                    )
                    values = _gather_cache_occurrences(
                        value_cache[layer_index],
                        actual_indices,
                        occurrence_limit=occurrence_limit,
                    )
                    access = _access_record(
                        position=position,
                        layer=layer_index,
                        rolling=rolling,
                        historical=historical,
                        historical_windows=historical_windows,
                        attention_mode=attention_mode,
                        proposal_centers=(
                            centers if position % proposal_trace_interval == 0 else ()
                        ),
                        proposal_spreads=(
                            spreads if position % proposal_trace_interval == 0 else ()
                        ),
                        proposal_weights=(
                            weights if position % proposal_trace_interval == 0 else ()
                        ),
                        proposal_active=(active if position % proposal_trace_interval == 0 else ()),
                        support_position=stream.support_positions[position],
                        history_end=rolling_start,
                        cache_length=len(key_cache[layer_index]),
                        gathered_key_occurrences=keys.shape[1],
                        gathered_value_occurrences=values.shape[1],
                    )
                    logits = torch.einsum("hd,hnd->hn", query, keys) / math.sqrt(self.head_width)
                    weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
                    context = torch.einsum("hn,hnd->hd", weights, values).reshape(self.width)
                    hidden = hidden + layer.output(context)
                    hidden = hidden + layer.mlp(layer.mlp_norm(hidden))
                    token_accesses.append(access)

                output = self.lm_head(self.final_norm(hidden))
                target = stream.tokens[position + 1]
                loss = torch.nn.functional.cross_entropy(
                    output.unsqueeze(0),
                    torch.tensor([target], device=device),
                )
                scores.append(
                    TokenScore(
                        position=position,
                        target=target,
                        loss=float(loss.item()),
                        prediction=int(output.argmax().item()),
                        accesses=tuple(token_accesses),
                        phase=(
                            "dense_bootstrap"
                            if position < min(rolling_tokens, activation_tokens)
                            else "dense_bridge"
                            if position < activation_tokens
                            else "post_transition"
                        ),
                    )
                )
        return scores


class HuggingFaceQwen2Reader:
    """Frozen Qwen2-family reader with actual layer-local bounded K/V gathers."""

    def __init__(self, model: Any, tokenizer: Any = None, *, specification: dict[str, Any]) -> None:
        if model.config.model_type != "qwen2":
            raise ValueError("the pretrained adapter currently supports only Qwen2-family models")
        if any(layer_type != "full_attention" for layer_type in model.config.layer_types):
            raise ValueError("Qwen2 sliding-attention layers are not supported by this adapter")
        if model.config.rope_scaling is not None:
            raise ValueError("Qwen2 nondefault RoPE scaling is not supported by this adapter")
        model.config._attn_implementation = "eager"
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.specification = specification
        self._model_weights_digest: str | None = None
        self._model_weights_signature: tuple[tuple[str, int], ...] | None = None
        self._identity_cache: dict[str, Any] | None = None
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

        self._apply_rotary_pos_emb = apply_rotary_pos_emb
        self._repeat_kv = repeat_kv

    def identity(self) -> dict[str, Any]:
        import transformers

        if self._identity_cache is not None:
            return json.loads(json.dumps(self._identity_cache))

        config = self.model.config
        config_sha256 = hashlib.sha256(
            json.dumps(config.to_dict(), default=str, sort_keys=True).encode()
        ).hexdigest()
        tokenizer_vocabulary_sha256 = None
        tokenizer_class = None
        resolved_tokenizer_revision = None
        if self.tokenizer is not None:
            tokenizer_class = type(self.tokenizer).__name__
            vocabulary = sorted(self.tokenizer.get_vocab().items())
            encoded = json.dumps(vocabulary, separators=(",", ":")).encode()
            tokenizer_vocabulary_sha256 = hashlib.sha256(encoded).hexdigest()
            resolved_tokenizer_revision = self.tokenizer.init_kwargs.get("_commit_hash")
        tokenizer_id = self.specification.get("tokenizer_id", self.specification["id"])
        tokenizer_revision = self.specification.get(
            "tokenizer_revision", self.specification["revision"]
        )
        self._identity_cache = {
            "backend": "huggingface",
            "model_id": self.specification["id"],
            "model_revision": self.specification["revision"],
            "resolved_model_revision": getattr(config, "_commit_hash", None),
            "model_config_sha256": config_sha256,
            "architecture": type(self.model).__name__,
            "model_type": config.model_type,
            "dtype": str(next(self.model.parameters()).dtype).removeprefix("torch."),
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "layers": config.num_hidden_layers,
            "attention_heads": config.num_attention_heads,
            "key_value_heads": config.num_key_value_heads,
            "rope_theta": config.rope_theta,
            "rope_scaling": config.rope_scaling,
            "attention_implementation": "eager",
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
            "resolved_tokenizer_revision": resolved_tokenizer_revision,
            "tokenizer_class": tokenizer_class,
            "tokenizer_vocabulary_sha256": tokenizer_vocabulary_sha256,
            "transformers": transformers.__version__,
        }
        return json.loads(json.dumps(self._identity_cache))

    def _append_layer_kv(
        self,
        layer: Any,
        hidden: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        key_cache: list[Tensor] | _LayerKVCache,
        value_cache: list[Tensor] | _LayerKVCache,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        attention = layer.self_attn
        query = attention.q_proj(normalized).view(
            1, 1, self.model.config.num_attention_heads, attention.head_dim
        )
        key = attention.k_proj(normalized).view(
            1, 1, self.model.config.num_key_value_heads, attention.head_dim
        )
        value = attention.v_proj(normalized).view(
            1, 1, self.model.config.num_key_value_heads, attention.head_dim
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self._apply_rotary_pos_emb(query, key, *position_embeddings)
        key_cache.append(key[0, :, 0, :])
        value_cache.append(value[0, :, 0, :])
        return residual, normalized, query[0, :, 0, :]

    def _finish_layer(
        self,
        layer: Any,
        residual: Tensor,
        query: Tensor,
        key_cache: list[Tensor] | _LayerKVCache,
        value_cache: list[Tensor] | _LayerKVCache,
        actual_indices: tuple[int, ...] | Tensor | None,
        *,
        occurrence_limit: int | None,
    ) -> tuple[Tensor, int, int]:
        attention = layer.self_attn
        keys = _gather_cache_occurrences(
            key_cache,
            actual_indices,
            occurrence_limit=occurrence_limit,
        )
        values = _gather_cache_occurrences(
            value_cache,
            actual_indices,
            occurrence_limit=occurrence_limit,
        )
        gathered_key_occurrences = keys.shape[1]
        gathered_value_occurrences = values.shape[1]
        keys = self._repeat_kv(keys.unsqueeze(0), attention.num_key_value_groups).squeeze(0)
        values = self._repeat_kv(values.unsqueeze(0), attention.num_key_value_groups).squeeze(0)
        attention_logits = torch.einsum("hd,hnd->hn", query, keys) * attention.scaling
        weights = torch.softmax(attention_logits, dim=-1, dtype=torch.float32).to(query.dtype)
        context = torch.einsum("hn,hnd->hd", weights, values)
        hidden = residual + attention.o_proj(context.reshape(1, 1, -1).contiguous())
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        return hidden, gathered_key_occurrences, gathered_value_occurrences

    def _finish_layer_with_soft_history(
        self,
        layer: Any,
        residual: Tensor,
        query: Tensor,
        key_cache: list[Tensor] | _LayerKVCache,
        value_cache: list[Tensor] | _LayerKVCache,
        historical_occupancy: Tensor,
        rolling: tuple[int, ...],
    ) -> Tensor:
        history_end = historical_occupancy.numel()
        cache_length = len(key_cache)
        if (
            len(value_cache) != cache_length
            or len(rolling) != cache_length - history_end
            or not rolling
            or rolling[0] != history_end
            or rolling[-1] != cache_length - 1
        ):
            raise ValueError("soft historical occupancy does not span the causal cache")
        keys = _gather_cache_occurrences(key_cache, None, occurrence_limit=None)
        values = _gather_cache_occurrences(value_cache, None, occurrence_limit=None)
        attention = layer.self_attn
        keys = self._repeat_kv(keys.unsqueeze(0), attention.num_key_value_groups).squeeze(0)
        values = self._repeat_kv(values.unsqueeze(0), attention.num_key_value_groups).squeeze(0)
        attention_logits = torch.einsum("hd,hnd->hn", query, keys) * attention.scaling
        availability = torch.cat(
            (
                historical_occupancy.clamp_min(1e-6),
                torch.ones(len(rolling), device=query.device, dtype=query.dtype),
            )
        )
        attention_logits = attention_logits + availability.log().unsqueeze(0)
        weights = torch.softmax(attention_logits, dim=-1, dtype=torch.float32).to(query.dtype)
        context = torch.einsum("hn,hnd->hd", weights, values)
        hidden = residual + attention.o_proj(context.reshape(1, 1, -1).contiguous())
        residual = hidden
        return residual + layer.mlp(layer.post_attention_layernorm(hidden))

    def _complete_current_token(
        self,
        hidden: Tensor,
        *,
        start_layer: int,
        position_embeddings: tuple[Tensor, Tensor],
        key_cache: list[list[Tensor] | _LayerKVCache],
        value_cache: list[list[Tensor] | _LayerKVCache],
        rolling: tuple[int, ...] | Tensor,
    ) -> Tensor:
        backbone = self.model.model
        for layer_index in range(start_layer, len(backbone.layers)):
            layer = backbone.layers[layer_index]
            branch_keys = (
                key_cache[layer_index].fork()
                if isinstance(key_cache[layer_index], _LayerKVCache)
                else list(key_cache[layer_index])
            )
            branch_values = (
                value_cache[layer_index].fork()
                if isinstance(value_cache[layer_index], _LayerKVCache)
                else list(value_cache[layer_index])
            )
            residual, _, query = self._append_layer_kv(
                layer,
                hidden,
                position_embeddings,
                branch_keys,
                branch_values,
            )
            hidden, _, _ = self._finish_layer(
                layer,
                residual,
                query,
                branch_keys,
                branch_values,
                rolling,
                occurrence_limit=len(rolling),
            )
        return self.model.lm_head(backbone.norm(hidden))[0, 0]

    def _model_weights_sha256(self) -> str:
        state = sorted(self.model.state_dict().items())
        signature = tuple((name, value._version) for name, value in state)
        if self._model_weights_digest is None or signature != self._model_weights_signature:
            digest = hashlib.sha256()
            for name, value in state:
                contiguous = value.detach().cpu().contiguous()
                digest.update(name.encode())
                digest.update(str(tuple(contiguous.shape)).encode())
                digest.update(str(contiguous.dtype).encode())
                digest.update(contiguous.view(torch.uint8).numpy())
            self._model_weights_digest = digest.hexdigest()
            self._model_weights_signature = signature
        return self._model_weights_digest

    @staticmethod
    def _prefix_sha256(tokens: tuple[int, ...]) -> str:
        encoded = json.dumps(tokens, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _dense_prefill_identity(
        self,
        prefix_tokens: tuple[int, ...],
        *,
        rolling_tokens: int,
        activation_tokens: int,
        device: torch.device,
        ownership: str,
        prefill_chunk_tokens: int,
    ) -> dict[str, Any]:
        config = self.model.config
        return {
            "format": "qkvs-dense-prefill-v1",
            "ownership": ownership,
            "prefix_tokens_sha256": self._prefix_sha256(prefix_tokens),
            "prefix_length": len(prefix_tokens),
            "working_memory_tokens": rolling_tokens,
            "activation_tokens": activation_tokens,
            "prefill_chunk_tokens": prefill_chunk_tokens,
            "model": self.identity(),
            "model_weights_sha256": self._model_weights_sha256(),
            "position": {
                "max_position_embeddings": config.max_position_embeddings,
                "rope_theta": config.rope_theta,
                "rope_scaling": config.rope_scaling,
            },
            "numerical": {
                "torch": str(torch.__version__),
                "dtype": str(next(self.model.parameters()).dtype),
                "device": str(device),
                "attention_implementation": "eager",
                "default_dtype": str(torch.get_default_dtype()),
                "fp32_precision": {
                    "global": torch.backends.fp32_precision,
                    "cuda_matmul": torch.backends.cuda.matmul.fp32_precision,
                    "cudnn": torch.backends.cudnn.fp32_precision,
                    "cudnn_conv": torch.backends.cudnn.conv.fp32_precision,
                },
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
            },
        }

    def _validate_dense_prefill(
        self,
        snapshot: DensePrefillSnapshot,
        stream: TokenStream,
        *,
        rolling_tokens: int,
        activation_tokens: int,
        device: torch.device,
        ownership: str,
        prefill_chunk_tokens: int = 128,
    ) -> None:
        _validate_transition(rolling_tokens, activation_tokens)
        if activation_tokens == 0:
            raise ValueError("a dense prefill snapshot requires a positive activation threshold")
        if len(stream.tokens) < activation_tokens:
            raise ValueError("token stream is shorter than the dense prefill")
        prefix = stream.tokens[:activation_tokens]
        expected = self._dense_prefill_identity(
            prefix,
            rolling_tokens=rolling_tokens,
            activation_tokens=activation_tokens,
            device=device,
            ownership=ownership,
            prefill_chunk_tokens=prefill_chunk_tokens,
        )
        if snapshot.identity != expected or snapshot.prefix_tokens != prefix:
            raise ValueError("dense prefill snapshot identity does not match this continuation")
        if snapshot.next_position != activation_tokens:
            raise ValueError("dense prefill continuation position is invalid")
        layer_count = len(self.model.model.layers)
        if len(snapshot.key_cache) != layer_count or len(snapshot.value_cache) != layer_count:
            raise ValueError("dense prefill layer cache count is invalid")
        if any(
            cache.ndim != 3 or cache.shape[1] != activation_tokens for cache in snapshot.key_cache
        ):
            raise ValueError("dense prefill key cache length is invalid")
        if any(
            cache.ndim != 3 or cache.shape[1] != activation_tokens for cache in snapshot.value_cache
        ):
            raise ValueError("dense prefill value cache length is invalid")
        if len(snapshot.dense_losses) != max(0, activation_tokens - 1):
            raise ValueError("dense prefill loss metadata is invalid")
        if len(snapshot.dense_predictions) != activation_tokens:
            raise ValueError("dense prefill prediction metadata is invalid")

    def dense_prefill(
        self,
        stream: TokenStream,
        *,
        rolling_tokens: int,
        activation_tokens: int,
        device: torch.device,
        ownership: str,
        prefill_chunk_tokens: int = 128,
    ) -> DensePrefillSnapshot:
        """Run the S-independent dense prefix once and retain continuation state."""

        _validate_transition(rolling_tokens, activation_tokens)
        if activation_tokens == 0:
            raise ValueError("dense prefill requires a positive activation threshold")
        if len(stream.tokens) < activation_tokens:
            raise ValueError("token stream is shorter than the dense prefill")
        if activation_tokens > self.model.config.max_position_embeddings:
            raise ValueError("dense prefill exceeds the model's supported position range")
        if prefill_chunk_tokens <= 0:
            raise ValueError("dense prefill chunk size must be positive")
        self.model.to(device)
        from transformers import StaticCache

        cache = StaticCache(config=self.model.config, max_cache_len=activation_tokens)
        dense_losses: list[float] = []
        dense_predictions: list[int] = []
        boundary_logits: Tensor | None = None

        with torch.inference_mode():
            for start in range(0, activation_tokens, prefill_chunk_tokens):
                stop = min(activation_tokens, start + prefill_chunk_tokens)
                positions = torch.arange(start, stop, device=device)
                output = self.model(
                    input_ids=torch.tensor([stream.tokens[start:stop]], device=device),
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=positions,
                    position_ids=positions.unsqueeze(0),
                    logits_to_keep=0,
                )
                cache = output.past_key_values
                logits = output.logits[0]
                dense_predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
                scored_stop = min(stop, activation_tokens - 1)
                scored = scored_stop - start
                if scored > 0:
                    targets = torch.tensor(
                        stream.tokens[start + 1 : scored_stop + 1],
                        device=device,
                    )
                    dense_losses.extend(
                        torch.nn.functional.cross_entropy(
                            logits[:scored], targets, reduction="none"
                        )
                        .detach()
                        .cpu()
                        .tolist()
                    )
                boundary_logits = logits[-1].detach().cpu().clone()

        if boundary_logits is None:
            raise RuntimeError("dense prefill did not execute any tokens")
        prefix = stream.tokens[:activation_tokens]
        snapshot = DensePrefillSnapshot(
            identity=self._dense_prefill_identity(
                prefix,
                rolling_tokens=rolling_tokens,
                activation_tokens=activation_tokens,
                device=device,
                ownership=ownership,
                prefill_chunk_tokens=prefill_chunk_tokens,
            ),
            prefix_tokens=prefix,
            next_position=activation_tokens,
            key_cache=tuple(
                layer.keys[0, :, :activation_tokens].detach().cpu().clone()
                for layer in cache.layers
            ),
            value_cache=tuple(
                layer.values[0, :, :activation_tokens].detach().cpu().clone()
                for layer in cache.layers
            ),
            dense_losses=tuple(dense_losses),
            dense_predictions=tuple(dense_predictions),
            boundary_logits=boundary_logits,
        )
        self._validate_dense_prefill(
            snapshot,
            stream,
            rolling_tokens=rolling_tokens,
            activation_tokens=activation_tokens,
            device=device,
            ownership=ownership,
            prefill_chunk_tokens=prefill_chunk_tokens,
        )
        return snapshot

    def save_dense_prefill(self, snapshot: DensePrefillSnapshot, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(
            {
                "identity": snapshot.identity,
                "prefix_tokens": snapshot.prefix_tokens,
                "next_position": snapshot.next_position,
                "key_cache": snapshot.key_cache,
                "value_cache": snapshot.value_cache,
                "dense_losses": snapshot.dense_losses,
                "dense_predictions": snapshot.dense_predictions,
                "boundary_logits": snapshot.boundary_logits,
            },
            temporary,
        )
        temporary.replace(destination)

    def load_dense_prefill(
        self,
        path: str | Path,
        stream: TokenStream,
        *,
        rolling_tokens: int,
        activation_tokens: int,
        device: torch.device,
        ownership: str,
        prefill_chunk_tokens: int = 128,
    ) -> DensePrefillSnapshot:
        saved = torch.load(Path(path), map_location="cpu", weights_only=True)
        snapshot = DensePrefillSnapshot(
            identity=saved["identity"],
            prefix_tokens=tuple(saved["prefix_tokens"]),
            next_position=int(saved["next_position"]),
            key_cache=tuple(saved["key_cache"]),
            value_cache=tuple(saved["value_cache"]),
            dense_losses=tuple(saved["dense_losses"]),
            dense_predictions=tuple(saved["dense_predictions"]),
            boundary_logits=saved["boundary_logits"],
        )
        self._validate_dense_prefill(
            snapshot,
            stream,
            rolling_tokens=rolling_tokens,
            activation_tokens=activation_tokens,
            device=device,
            ownership=ownership,
            prefill_chunk_tokens=prefill_chunk_tokens,
        )
        return snapshot

    def _caches_from_dense_prefill(
        self,
        snapshot: DensePrefillSnapshot,
        device: torch.device,
    ) -> tuple[list[_LayerKVCache], list[_LayerKVCache]]:
        capacity = self.model.config.max_position_embeddings
        return (
            [
                _LayerKVCache.from_prefix(layer, capacity=capacity, device=device)
                for layer in snapshot.key_cache
            ],
            [
                _LayerKVCache.from_prefix(layer, capacity=capacity, device=device)
                for layer in snapshot.value_cache
            ],
        )

    def new_heatfield_rollout_state(
        self,
        stream: TokenStream,
        snapshot: DensePrefillSnapshot,
        *,
        rolling_tokens: int,
        activation_tokens: int,
        device: torch.device,
        ownership: str,
        prefill_chunk_tokens: int,
    ) -> HeatfieldRolloutState:
        self._validate_dense_prefill(
            snapshot,
            stream,
            rolling_tokens=rolling_tokens,
            activation_tokens=activation_tokens,
            device=device,
            ownership=ownership,
            prefill_chunk_tokens=prefill_chunk_tokens,
        )
        key_cache, value_cache = self._caches_from_dense_prefill(snapshot, device)
        return HeatfieldRolloutState(
            stream_id=stream.stream_id,
            next_position=activation_tokens,
            key_cache=key_cache,
            value_cache=value_cache,
            policy_states=[PolicyState() for _ in self.model.model.layers],
        )

    def serialize_heatfield_rollout_state(self, state: HeatfieldRolloutState) -> dict[str, Any]:
        return {
            "stream_id": state.stream_id,
            "next_position": state.next_position,
            "key_cache": tuple(cache.snapshot() for cache in state.key_cache),
            "value_cache": tuple(cache.snapshot() for cache in state.value_cache),
            "policy_states": tuple(
                {
                    "centers": policy_state.centers,
                    "recurrent": (
                        policy_state.recurrent.detach().cpu()
                        if policy_state.recurrent is not None
                        else None
                    ),
                }
                for policy_state in state.policy_states
            ),
        }

    def restore_heatfield_rollout_state(
        self,
        saved: dict[str, Any],
        stream: TokenStream,
        snapshot: DensePrefillSnapshot,
        *,
        rolling_tokens: int,
        activation_tokens: int,
        device: torch.device,
        ownership: str,
        prefill_chunk_tokens: int,
    ) -> HeatfieldRolloutState:
        self._validate_dense_prefill(
            snapshot,
            stream,
            rolling_tokens=rolling_tokens,
            activation_tokens=activation_tokens,
            device=device,
            ownership=ownership,
            prefill_chunk_tokens=prefill_chunk_tokens,
        )
        next_position = int(saved["next_position"])
        if (
            saved["stream_id"] != stream.stream_id
            or next_position < activation_tokens
            or next_position > len(stream.tokens) - 1
        ):
            raise ValueError("streaming training checkpoint continuation is invalid")
        key_tensors = tuple(saved["key_cache"])
        value_tensors = tuple(saved["value_cache"])
        layer_count = len(self.model.model.layers)
        if (
            len(key_tensors) != layer_count
            or len(value_tensors) != layer_count
            or any(tensor.shape[1] != next_position for tensor in key_tensors)
            or any(tensor.shape[1] != next_position for tensor in value_tensors)
        ):
            raise ValueError("streaming training checkpoint K/V length is invalid")
        capacity = self.model.config.max_position_embeddings
        policy_states = [
            PolicyState(
                tuple(record["centers"]),
                record["recurrent"].to(device) if record["recurrent"] is not None else None,
            )
            for record in saved["policy_states"]
        ]
        if len(policy_states) != layer_count:
            raise ValueError("streaming training checkpoint S state is invalid")
        return HeatfieldRolloutState(
            stream_id=stream.stream_id,
            next_position=next_position,
            key_cache=[
                _LayerKVCache.from_prefix(tensor, capacity=capacity, device=device)
                for tensor in key_tensors
            ],
            value_cache=[
                _LayerKVCache.from_prefix(tensor, capacity=capacity, device=device)
                for tensor in value_tensors
            ],
            policy_states=policy_states,
        )

    def _scores_from_dense_prefill(
        self,
        snapshot: DensePrefillSnapshot,
        stream: TokenStream,
        *,
        rolling_tokens: int,
        device: torch.device,
    ) -> list[TokenScore]:
        score_count = min(snapshot.next_position, len(stream.tokens) - 1)
        scores = [
            TokenScore(
                position=position,
                target=stream.tokens[position + 1],
                loss=snapshot.dense_losses[position],
                prediction=snapshot.dense_predictions[position],
                accesses=(),
                phase=("dense_bootstrap" if position < rolling_tokens else "dense_bridge"),
            )
            for position in range(min(score_count, snapshot.next_position - 1))
        ]
        boundary_position = snapshot.next_position - 1
        if boundary_position < score_count:
            target = torch.tensor([stream.tokens[boundary_position + 1]], device=device)
            logits = snapshot.boundary_logits.to(device)
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)
            scores.append(
                TokenScore(
                    position=boundary_position,
                    target=int(target.item()),
                    loss=float(loss),
                    prediction=snapshot.dense_predictions[boundary_position],
                    accesses=(),
                    phase=(
                        "dense_bootstrap" if boundary_position < rolling_tokens else "dense_bridge"
                    ),
                )
            )
        return scores

    def _execute(
        self,
        stream: TokenStream,
        policy: SelectionPolicy,
        *,
        rolling_tokens: int,
        historical_token_budget: int,
        window_width: int,
        device: torch.device,
        diagnostic_full_history: bool,
        activation_tokens: int = 0,
        dense_snapshot: DensePrefillSnapshot | None = None,
        snapshot_ownership: str = "evaluation",
        prefill_chunk_tokens: int = 128,
        proposal_trace_interval: int = 1,
    ) -> tuple[list[TokenScore], Tensor | None]:
        _validate_transition(rolling_tokens, activation_tokens)
        if len(stream.tokens) > self.model.config.max_position_embeddings:
            raise ValueError("token stream exceeds the model's supported position range")
        if diagnostic_full_history and (activation_tokens or dense_snapshot is not None):
            raise ValueError("full-history diagnostics do not use activation snapshots")
        if dense_snapshot is not None and activation_tokens == 0:
            raise ValueError("dense prefill snapshot supplied without an activation threshold")
        if proposal_trace_interval <= 0:
            raise ValueError("proposal trace interval must be positive")
        self.model.to(device)
        backbone = self.model.model
        if activation_tokens:
            if dense_snapshot is None:
                dense_snapshot = self.dense_prefill(
                    stream,
                    rolling_tokens=rolling_tokens,
                    activation_tokens=activation_tokens,
                    device=device,
                    ownership=snapshot_ownership,
                    prefill_chunk_tokens=prefill_chunk_tokens,
                )
            self._validate_dense_prefill(
                dense_snapshot,
                stream,
                rolling_tokens=rolling_tokens,
                activation_tokens=activation_tokens,
                device=device,
                ownership=snapshot_ownership,
                prefill_chunk_tokens=prefill_chunk_tokens,
            )
            key_cache, value_cache = self._caches_from_dense_prefill(dense_snapshot, device)
            scores = self._scores_from_dense_prefill(
                dense_snapshot,
                stream,
                rolling_tokens=rolling_tokens,
                device=device,
            )
            start_position = activation_tokens
        else:
            key_cache = [[] for _ in backbone.layers]
            value_cache = [[] for _ in backbone.layers]
            scores = []
            start_position = 0
        policy_states = [PolicyState() for _ in backbone.layers]
        output_logits: list[Tensor] | None = [] if diagnostic_full_history else None
        pending_scores: list[tuple[int, int, Tensor, Tensor, tuple[AccessRecord, ...]]] = []
        stream_token_ids = torch.tensor(stream.tokens, device=device)
        absolute_positions = torch.arange(len(stream.tokens), device=device)
        layer_invariant_selection = getattr(policy, "layer_invariant", False)

        with torch.inference_mode():
            for position in range(start_position, len(stream.tokens) - 1):
                token_id = stream_token_ids[position].view(1, 1)
                hidden = backbone.embed_tokens(token_id)
                position_ids = absolute_positions[position].view(1, 1)
                position_embeddings = backbone.rotary_emb(hidden, position_ids)
                token_accesses: list[AccessRecord] = []
                lower_layer_selection = None
                rolling_start = max(0, position - rolling_tokens + 1)
                rolling = tuple(range(rolling_start, position + 1))
                shared_selection = None

                for layer_index, layer in enumerate(backbone.layers):
                    residual, normalized, query = self._append_layer_kv(
                        layer,
                        hidden,
                        position_embeddings,
                        key_cache[layer_index],
                        value_cache[layer_index],
                    )

                    if diagnostic_full_history or getattr(policy, "full_qkv", False):
                        historical: tuple[int, ...] = ()
                        historical_windows = ((0, rolling_start),) if rolling_start else ()
                        centers: tuple[float, ...] = ()
                        spreads: tuple[float, ...] = ()
                        weights: tuple[float, ...] = ()
                        active: tuple[bool, ...] = ()
                        actual_indices = None
                        occurrence_limit = None
                        attention_mode = "full_qkv"
                    else:
                        if shared_selection is None:
                            proposal_set = policy.propose(
                                SelectionContext(
                                    position=position,
                                    layer=layer_index,
                                    history_end=rolling_start,
                                    hidden=normalized[0, 0],
                                    lower_layer=lower_layer_selection,
                                ),
                                policy_states[layer_index],
                            )
                            selection = resolve_proposals(
                                proposal_set.proposals,
                                history_end=rolling_start,
                                window_width=window_width,
                                historical_token_budget=historical_token_budget,
                                beta=proposal_set.beta,
                            )
                            historical = selection.indices
                            actual_indices = torch.tensor(
                                (*historical, *rolling),
                                device=device,
                                dtype=torch.long,
                            )
                            if layer_invariant_selection:
                                shared_selection = proposal_set, selection, actual_indices
                        else:
                            proposal_set, selection, actual_indices = shared_selection
                            historical = selection.indices
                        policy_states[layer_index] = proposal_set.state
                        lower_layer_selection = proposal_set.soft_parameters
                        historical_windows = tuple(
                            (window[0], window[-1] + 1) for window in selection.windows if window
                        )
                        centers = tuple(proposal.center for proposal in proposal_set.proposals)
                        spreads = tuple(proposal.spread for proposal in proposal_set.proposals)
                        weights = tuple(proposal.weight for proposal in proposal_set.proposals)
                        active = tuple(proposal.active for proposal in proposal_set.proposals)
                        occurrence_limit = rolling_tokens + historical_token_budget
                        attention_mode = "bounded"
                    hidden, gathered_keys, gathered_values = self._finish_layer(
                        layer,
                        residual,
                        query,
                        key_cache[layer_index],
                        value_cache[layer_index],
                        actual_indices,
                        occurrence_limit=occurrence_limit,
                    )
                    access = _access_record(
                        position=position,
                        layer=layer_index,
                        rolling=rolling,
                        historical=historical,
                        historical_windows=historical_windows,
                        attention_mode=attention_mode,
                        proposal_centers=(
                            centers if position % proposal_trace_interval == 0 else ()
                        ),
                        proposal_spreads=(
                            spreads if position % proposal_trace_interval == 0 else ()
                        ),
                        proposal_weights=(
                            weights if position % proposal_trace_interval == 0 else ()
                        ),
                        proposal_active=(active if position % proposal_trace_interval == 0 else ()),
                        support_position=stream.support_positions[position],
                        history_end=rolling_start,
                        cache_length=len(key_cache[layer_index]),
                        gathered_key_occurrences=gathered_keys,
                        gathered_value_occurrences=gathered_values,
                        historical_occurrence_count=(
                            rolling_start if attention_mode == "full_qkv" else None
                        ),
                        historical_unique_count=(
                            rolling_start if attention_mode == "full_qkv" else None
                        ),
                    )
                    token_accesses.append(access)

                logits = self.model.lm_head(backbone.norm(hidden))[0, 0]
                if output_logits is not None:
                    output_logits.append(logits.detach().cpu())
                target = stream.tokens[position + 1]
                loss = torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), stream_token_ids[position + 1].view(1)
                )
                pending_scores.append(
                    (
                        position,
                        target,
                        loss.detach(),
                        logits.argmax().detach(),
                        tuple(token_accesses),
                    )
                )
        if pending_scores:
            losses = torch.stack([record[2] for record in pending_scores]).cpu().tolist()
            predictions = torch.stack([record[3] for record in pending_scores]).cpu().tolist()
            scores.extend(
                TokenScore(
                    position=record[0],
                    target=record[1],
                    loss=float(loss),
                    prediction=int(prediction),
                    accesses=record[4],
                    phase="post_transition",
                )
                for record, loss, prediction in zip(
                    pending_scores, losses, predictions, strict=True
                )
            )
        return scores, torch.stack(output_logits) if output_logits is not None else None

    def execute(
        self,
        stream: TokenStream,
        policy: SelectionPolicy,
        *,
        rolling_tokens: int,
        historical_token_budget: int,
        window_width: int,
        device: torch.device,
        activation_tokens: int,
        dense_snapshot: DensePrefillSnapshot | None = None,
        snapshot_ownership: str = "evaluation",
        prefill_chunk_tokens: int = 128,
        proposal_trace_interval: int = 1,
    ) -> list[TokenScore]:
        if activation_tokens == 0:
            raise ValueError("bounded execution requires a positive activation threshold")
        scores, _ = self._execute(
            stream,
            policy,
            rolling_tokens=rolling_tokens,
            historical_token_budget=historical_token_budget,
            window_width=window_width,
            device=device,
            diagnostic_full_history=False,
            activation_tokens=activation_tokens,
            dense_snapshot=dense_snapshot,
            snapshot_ownership=snapshot_ownership,
            prefill_chunk_tokens=prefill_chunk_tokens,
            proposal_trace_interval=proposal_trace_interval,
        )
        return scores

    def _heatfield_objective_from_proposal(
        self,
        stream: TokenStream,
        proposal_set: Any,
        *,
        position: int,
        layer_index: int,
        residual: Tensor,
        query: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        key_cache: list[list[Tensor] | _LayerKVCache],
        value_cache: list[list[Tensor] | _LayerKVCache],
        rolling: tuple[int, ...],
        historical_token_budget: int,
        window_width: int,
        coverage_weight: float,
        utility_weight: float,
        device: torch.device,
    ) -> tuple[Tensor, dict[str, Any], tuple[int, ...]]:
        if proposal_set.soft_parameters is None:
            raise RuntimeError("learned policy did not provide differentiable proposals")
        rolling_start = rolling[0]
        selection = resolve_proposals(
            proposal_set.proposals,
            history_end=rolling_start,
            window_width=window_width,
            historical_token_budget=historical_token_budget,
            beta=proposal_set.beta,
        )
        target = torch.tensor([stream.tokens[position + 1]], device=device)
        target_layer = self.model.model.layers[layer_index]
        rolling_index = torch.tensor(rolling, device=device, dtype=torch.long)

        with torch.no_grad():
            rolling_hidden, _, _ = self._finish_layer(
                target_layer,
                residual,
                query,
                key_cache[layer_index],
                value_cache[layer_index],
                rolling_index,
                occurrence_limit=len(rolling),
            )
            rolling_logits = self._complete_current_token(
                rolling_hidden,
                start_layer=layer_index + 1,
                position_embeddings=position_embeddings,
                key_cache=key_cache,
                value_cache=value_cache,
                rolling=rolling_index,
            )
            rolling_loss = torch.nn.functional.cross_entropy(rolling_logits.unsqueeze(0), target)
            selected_indices = (*selection.indices, *rolling)
            selected_index = torch.tensor(selected_indices, device=device, dtype=torch.long)
            selected_hidden, _, _ = self._finish_layer(
                target_layer,
                residual,
                query,
                key_cache[layer_index],
                value_cache[layer_index],
                selected_index,
                occurrence_limit=len(rolling) + historical_token_budget,
            )
            selected_logits = self._complete_current_token(
                selected_hidden,
                start_layer=layer_index + 1,
                position_embeddings=position_embeddings,
                key_cache=key_cache,
                value_cache=value_cache,
                rolling=rolling_index,
            )
            selected_loss = torch.nn.functional.cross_entropy(selected_logits.unsqueeze(0), target)

            teacher_keys = _gather_cache_occurrences(
                key_cache[layer_index], None, occurrence_limit=None
            )
            attention = target_layer.self_attn
            teacher_keys = self._repeat_kv(
                teacher_keys.unsqueeze(0), attention.num_key_value_groups
            ).squeeze(0)
            teacher_logits = torch.einsum("hd,hnd->hn", query, teacher_keys)
            teacher_weights = torch.softmax(
                teacher_logits * attention.scaling,
                dim=-1,
                dtype=torch.float32,
            ).mean(dim=0)
            teacher_history = teacher_weights[:rolling_start]

        soft = proposal_set.soft_parameters
        collective_occupancy = _collective_soft_occupancy(
            soft,
            history_end=rolling_start,
            window_width=window_width,
            beta=proposal_set.beta,
        )
        coverage_loss, coverage_reward = _absolute_teacher_coverage(
            teacher_history,
            collective_occupancy,
        )

        soft_hidden = self._finish_layer_with_soft_history(
            target_layer,
            residual,
            query,
            key_cache[layer_index],
            value_cache[layer_index],
            collective_occupancy,
            rolling,
        )
        soft_logits = self._complete_current_token(
            soft_hidden,
            start_layer=layer_index + 1,
            position_embeddings=position_embeddings,
            key_cache=key_cache,
            value_cache=value_cache,
            rolling=rolling_index,
        )
        soft_loss = torch.nn.functional.cross_entropy(soft_logits.unsqueeze(0), target)
        utility_surrogate = soft_loss - rolling_loss.detach()
        objective = coverage_weight * coverage_loss + utility_weight * utility_surrogate
        windows = tuple((window[0], window[-1] + 1) for window in selection.windows if window)
        metrics = {
            "position": position,
            "layer": layer_index,
            "objective": objective.detach(),
            "coverage_loss": coverage_loss.detach(),
            "soft_next_token_loss": soft_loss.detach(),
            "rolling_next_token_loss": rolling_loss.detach(),
            "hard_selected_next_token_loss": selected_loss.detach(),
            "hard_utility": (rolling_loss - selected_loss).detach(),
            "soft_utility": (rolling_loss - soft_loss.detach()).detach(),
            "soft_hard_utility_gap": (selected_loss - soft_loss.detach()).detach(),
            "teacher_historical_attention_mass": teacher_weights[:rolling_start].sum().detach(),
            "teacher_historical_attention_covered_mass": coverage_reward.detach(),
            "teacher_historical_attention_uncovered_mass": coverage_loss.detach(),
            "historical_windows": windows,
            "historical_occurrences": selection.occurrence_count,
            "historical_unique_tokens": selection.unique_count,
            "overlap_occurrences": selection.occurrence_count - selection.unique_count,
            "proposal_parameters": soft.detach(),
            "coverage_overlap_alpha": 0.0,
            "trajectory": "learned_hard_detached",
            "temporal_credit_assignment": "target_step_only",
        }
        return objective, metrics, selection.indices

    def heatfield_training_objective(
        self,
        stream: TokenStream,
        policy: LearnedHeatFieldPolicy,
        *,
        position: int,
        layer_index: int,
        rolling_tokens: int,
        historical_token_budget: int,
        window_width: int,
        coverage_weight: float,
        utility_weight: float,
        device: torch.device,
        activation_tokens: int,
        dense_snapshot: DensePrefillSnapshot | None = None,
        snapshot_ownership: str = "train",
        prefill_chunk_tokens: int = 128,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Build one truncated causal S objective while the reader remains frozen."""

        _validate_transition(rolling_tokens, activation_tokens)
        if activation_tokens == 0:
            raise ValueError("heat-field training requires a positive activation threshold")
        if position < activation_tokens or position >= len(stream.tokens) - 1:
            raise ValueError("training position needs eligible history and a next-token target")
        if layer_index < 0 or layer_index >= len(self.model.model.layers):
            raise ValueError("training layer is outside the reader")
        if coverage_weight < 0.0 or utility_weight < 0.0:
            raise ValueError("training signal weights must be nonnegative")
        reused_dense_snapshot = dense_snapshot is not None
        self.model.to(device)
        policy.to(device)
        backbone = self.model.model
        if activation_tokens:
            if dense_snapshot is None:
                dense_snapshot = self.dense_prefill(
                    stream,
                    rolling_tokens=rolling_tokens,
                    activation_tokens=activation_tokens,
                    device=device,
                    ownership=snapshot_ownership,
                    prefill_chunk_tokens=prefill_chunk_tokens,
                )
            self._validate_dense_prefill(
                dense_snapshot,
                stream,
                rolling_tokens=rolling_tokens,
                activation_tokens=activation_tokens,
                device=device,
                ownership=snapshot_ownership,
                prefill_chunk_tokens=prefill_chunk_tokens,
            )
            key_cache, value_cache = self._caches_from_dense_prefill(dense_snapshot, device)
            trajectory_start = activation_tokens
        else:
            key_cache = [[] for _ in backbone.layers]
            value_cache = [[] for _ in backbone.layers]
            trajectory_start = 0
        policy_states = [PolicyState() for _ in backbone.layers]
        trajectory_historical_occurrences = 0
        trajectory_unique_tokens = 0
        trajectory_overlap_occurrences = 0

        with torch.no_grad():
            for prior_position in range(trajectory_start, position):
                token_id = torch.tensor([[stream.tokens[prior_position]]], device=device)
                hidden = backbone.embed_tokens(token_id)
                position_ids = torch.tensor([[prior_position]], device=device)
                position_embeddings = backbone.rotary_emb(hidden, position_ids)
                rolling_start = max(0, prior_position - rolling_tokens + 1)
                rolling = tuple(range(rolling_start, prior_position + 1))
                lower_layer_selection = None
                for current_layer, layer in enumerate(backbone.layers):
                    residual, normalized, query = self._append_layer_kv(
                        layer,
                        hidden,
                        position_embeddings,
                        key_cache[current_layer],
                        value_cache[current_layer],
                    )
                    proposal_set = policy.propose(
                        SelectionContext(
                            position=prior_position,
                            layer=current_layer,
                            history_end=rolling_start,
                            hidden=normalized[0, 0],
                            lower_layer=lower_layer_selection,
                        ),
                        policy_states[current_layer],
                    )
                    recurrent = proposal_set.state.recurrent
                    policy_states[current_layer] = PolicyState(
                        proposal_set.state.centers,
                        recurrent.detach() if recurrent is not None else None,
                    )
                    lower_layer_selection = (
                        proposal_set.soft_parameters.detach()
                        if proposal_set.soft_parameters is not None
                        else None
                    )
                    trajectory_selection = resolve_proposals(
                        proposal_set.proposals,
                        history_end=rolling_start,
                        window_width=window_width,
                        historical_token_budget=historical_token_budget,
                        beta=proposal_set.beta,
                    )
                    trajectory_historical_occurrences += trajectory_selection.occurrence_count
                    trajectory_unique_tokens += trajectory_selection.unique_count
                    trajectory_overlap_occurrences += (
                        trajectory_selection.occurrence_count - trajectory_selection.unique_count
                    )
                    hidden, _, _ = self._finish_layer(
                        layer,
                        residual,
                        query,
                        key_cache[current_layer],
                        value_cache[current_layer],
                        torch.tensor(
                            (*trajectory_selection.indices, *rolling),
                            device=device,
                            dtype=torch.long,
                        ),
                        occurrence_limit=rolling_tokens + historical_token_budget,
                    )

            token_id = torch.tensor([[stream.tokens[position]]], device=device)
            hidden = backbone.embed_tokens(token_id)
            position_ids = torch.tensor([[position]], device=device)
            position_embeddings = backbone.rotary_emb(hidden, position_ids)
            rolling_start = max(0, position - rolling_tokens + 1)
            rolling = tuple(range(rolling_start, position + 1))
            lower_layer_selection = None
            for current_layer in range(layer_index):
                layer = backbone.layers[current_layer]
                residual, normalized, query = self._append_layer_kv(
                    layer,
                    hidden,
                    position_embeddings,
                    key_cache[current_layer],
                    value_cache[current_layer],
                )
                proposal_set = policy.propose(
                    SelectionContext(
                        position=position,
                        layer=current_layer,
                        history_end=rolling_start,
                        hidden=normalized[0, 0],
                        lower_layer=lower_layer_selection,
                    ),
                    policy_states[current_layer],
                )
                recurrent = proposal_set.state.recurrent
                policy_states[current_layer] = PolicyState(
                    proposal_set.state.centers,
                    recurrent.detach() if recurrent is not None else None,
                )
                lower_layer_selection = (
                    proposal_set.soft_parameters.detach()
                    if proposal_set.soft_parameters is not None
                    else None
                )
                trajectory_selection = resolve_proposals(
                    proposal_set.proposals,
                    history_end=rolling_start,
                    window_width=window_width,
                    historical_token_budget=historical_token_budget,
                    beta=proposal_set.beta,
                )
                trajectory_historical_occurrences += trajectory_selection.occurrence_count
                trajectory_unique_tokens += trajectory_selection.unique_count
                trajectory_overlap_occurrences += (
                    trajectory_selection.occurrence_count - trajectory_selection.unique_count
                )
                hidden, _, _ = self._finish_layer(
                    layer,
                    residual,
                    query,
                    key_cache[current_layer],
                    value_cache[current_layer],
                    torch.tensor(
                        (*trajectory_selection.indices, *rolling),
                        device=device,
                        dtype=torch.long,
                    ),
                    occurrence_limit=rolling_tokens + historical_token_budget,
                )

            target_layer = backbone.layers[layer_index]
            residual, normalized, query = self._append_layer_kv(
                target_layer,
                hidden,
                position_embeddings,
                key_cache[layer_index],
                value_cache[layer_index],
            )

        proposal_set = policy.propose(
            SelectionContext(
                position=position,
                layer=layer_index,
                history_end=rolling_start,
                hidden=normalized[0, 0].detach(),
                lower_layer=lower_layer_selection,
            ),
            policy_states[layer_index],
        )
        objective, metrics, _ = self._heatfield_objective_from_proposal(
            stream,
            proposal_set,
            position=position,
            layer_index=layer_index,
            residual=residual,
            query=query,
            position_embeddings=position_embeddings,
            key_cache=key_cache,
            value_cache=value_cache,
            rolling=rolling,
            historical_token_budget=historical_token_budget,
            window_width=window_width,
            coverage_weight=coverage_weight,
            utility_weight=utility_weight,
            device=device,
        )
        metrics.update(
            {
                "activation_tokens": activation_tokens,
                "dense_prefix_reused": reused_dense_snapshot,
                "trajectory_historical_occurrences": trajectory_historical_occurrences,
                "trajectory_unique_tokens": trajectory_unique_tokens,
                "trajectory_overlap_occurrences": trajectory_overlap_occurrences,
            }
        )
        metrics = _materialize_training_metrics([metrics])[0]
        return objective, metrics

    def heatfield_training_rollout(
        self,
        stream: TokenStream,
        policy: LearnedHeatFieldPolicy,
        state: HeatfieldRolloutState,
        *,
        stop_position: int,
        target_layers: dict[int, int],
        rolling_tokens: int,
        historical_token_budget: int,
        window_width: int,
        coverage_weight: float,
        utility_weight: float,
        device: torch.device,
    ) -> tuple[list[Tensor], list[dict[str, Any]]]:
        """Advance one causal segment with fixed S weights and target-local gradients."""

        start_position = state.next_position
        if state.stream_id != stream.stream_id:
            raise ValueError("streaming training state belongs to another stream")
        if not start_position < stop_position <= len(stream.tokens) - 1:
            raise ValueError("streaming training rollout bounds are invalid")
        layer_count = len(self.model.model.layers)
        if any(
            position < start_position
            or position >= stop_position
            or layer < 0
            or layer >= layer_count
            for position, layer in target_layers.items()
        ):
            raise ValueError("streaming training target is outside the rollout")
        self.model.to(device)
        policy.to(device)
        backbone = self.model.model
        objectives: list[Tensor] = []
        metrics: list[dict[str, Any]] = []
        historical_occurrences = 0
        unique_tokens = 0
        overlap_occurrences = 0
        rollout_token_ids = torch.tensor(stream.tokens[start_position:stop_position], device=device)
        rollout_positions = torch.arange(start_position, stop_position, device=device)

        for position in range(start_position, stop_position):
            offset = position - start_position
            with torch.no_grad():
                token_id = rollout_token_ids[offset].view(1, 1)
                hidden = backbone.embed_tokens(token_id)
                position_ids = rollout_positions[offset].view(1, 1)
                position_embeddings = backbone.rotary_emb(hidden, position_ids)
            rolling_start = max(0, position - rolling_tokens + 1)
            rolling = tuple(range(rolling_start, position + 1))
            lower_layer_selection = None
            target_layer_index = target_layers.get(position)

            for layer_index, layer in enumerate(backbone.layers):
                with torch.no_grad():
                    residual, normalized, query = self._append_layer_kv(
                        layer,
                        hidden,
                        position_embeddings,
                        state.key_cache[layer_index],
                        state.value_cache[layer_index],
                    )
                context = SelectionContext(
                    position=position,
                    layer=layer_index,
                    history_end=rolling_start,
                    hidden=normalized[0, 0].detach(),
                    lower_layer=lower_layer_selection,
                )
                if layer_index == target_layer_index:
                    proposal_set = policy.propose(context, state.policy_states[layer_index])
                    objective, target_metrics, selected_indices = (
                        self._heatfield_objective_from_proposal(
                            stream,
                            proposal_set,
                            position=position,
                            layer_index=layer_index,
                            residual=residual,
                            query=query,
                            position_embeddings=position_embeddings,
                            key_cache=state.key_cache,
                            value_cache=state.value_cache,
                            rolling=rolling,
                            historical_token_budget=historical_token_budget,
                            window_width=window_width,
                            coverage_weight=coverage_weight,
                            utility_weight=utility_weight,
                            device=device,
                        )
                    )
                    objectives.append(objective)
                    metrics.append(target_metrics)
                else:
                    with torch.no_grad():
                        proposal_set = policy.propose(context, state.policy_states[layer_index])
                    selection = resolve_proposals(
                        proposal_set.proposals,
                        history_end=rolling_start,
                        window_width=window_width,
                        historical_token_budget=historical_token_budget,
                        beta=proposal_set.beta,
                    )
                    selected_indices = selection.indices

                recurrent = proposal_set.state.recurrent
                state.policy_states[layer_index] = PolicyState(
                    proposal_set.state.centers,
                    recurrent.detach() if recurrent is not None else None,
                )
                lower_layer_selection = (
                    proposal_set.soft_parameters.detach()
                    if proposal_set.soft_parameters is not None
                    else None
                )
                occurrence_count = len(selected_indices)
                unique_count = len(set(selected_indices))
                historical_occurrences += occurrence_count
                unique_tokens += unique_count
                overlap_occurrences += occurrence_count - unique_count
                with torch.no_grad():
                    actual_indices = torch.tensor(
                        (*selected_indices, *rolling),
                        device=device,
                        dtype=torch.long,
                    )
                    hidden, _, _ = self._finish_layer(
                        layer,
                        residual,
                        query,
                        state.key_cache[layer_index],
                        state.value_cache[layer_index],
                        actual_indices,
                        occurrence_limit=rolling_tokens + historical_token_budget,
                    )
            state.next_position = position + 1

        metrics = _materialize_training_metrics(metrics)
        for target_metrics in metrics:
            target_metrics.update(
                {
                    "rollout_start_position": start_position,
                    "rollout_stop_position": stop_position,
                    "rollout_historical_occurrences": historical_occurrences,
                    "rollout_unique_tokens": unique_tokens,
                    "rollout_overlap_occurrences": overlap_occurrences,
                    "selector_weights_fixed_within_rollout": True,
                }
            )
        return objectives, metrics

    def full_history_diagnostic(
        self,
        tokens: tuple[int, ...],
        *,
        device: torch.device,
        atol: float,
        rtol: float,
        loss_atol: float | None = None,
        loss_rtol: float | None = None,
    ) -> dict[str, Any]:
        """Compare unbounded incremental execution with the standard reader on a short stream."""

        from qkvs.selection import RollingOnlyPolicy

        stream = TokenStream("full-history-diagnostic", tokens, (None,) * (len(tokens) - 1))
        _, incremental = self._execute(
            stream,
            RollingOnlyPolicy(),
            rolling_tokens=1,
            historical_token_budget=0,
            window_width=1,
            device=device,
            diagnostic_full_history=True,
        )
        if incremental is None:
            raise RuntimeError("full-history diagnostic did not retain incremental logits")
        token_ids = torch.tensor([tokens], device=device)
        with torch.inference_mode():
            reference = (
                self.model(
                    input_ids=token_ids,
                    use_cache=False,
                    logits_to_keep=0,
                )
                .logits[0, :-1]
                .detach()
                .cpu()
            )
        difference = (incremental.float() - reference.float()).abs()
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        incremental_losses = torch.nn.functional.cross_entropy(
            incremental.float(),
            targets,
            reduction="none",
        )
        reference_losses = torch.nn.functional.cross_entropy(
            reference.float(),
            targets,
            reduction="none",
        )
        loss_difference = (incremental_losses - reference_losses).abs()
        effective_loss_atol = atol if loss_atol is None else loss_atol
        effective_loss_rtol = rtol if loss_rtol is None else loss_rtol
        return {
            "diagnostic_only_unbounded": True,
            "positions": len(tokens) - 1,
            "vocabulary": reference.shape[-1],
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "per_position_max_abs_error": difference.amax(dim=-1).tolist(),
            "allclose": bool(
                torch.allclose(incremental.float(), reference.float(), atol=atol, rtol=rtol)
            ),
            "max_abs_loss_error": float(loss_difference.max().item()),
            "mean_abs_loss_error": float(loss_difference.mean().item()),
            "per_position_abs_loss_error": loss_difference.tolist(),
            "loss_atol": effective_loss_atol,
            "loss_rtol": effective_loss_rtol,
            "loss_allclose": bool(
                torch.allclose(
                    incremental_losses,
                    reference_losses,
                    atol=effective_loss_atol,
                    rtol=effective_loss_rtol,
                )
            ),
        }


def load_reader(specification: dict[str, Any], device: torch.device) -> Any:
    if not specification["frozen"]:
        raise ValueError("the research harness requires a frozen reader")
    if specification["backend"] == "synthetic":
        reader = FrozenCausalReader(
            vocab_size=specification["vocab_size"],
            width=specification["width"],
            layers=specification["layers"],
            heads=specification["heads"],
            seed=specification["seed"],
        )
        return reader.to(device)
    if specification["backend"] != "huggingface":
        raise ValueError(f"unsupported reader backend: {specification['backend']}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[specification["dtype"]]
    tokenizer_id = specification.get("tokenizer_id", specification["id"])
    tokenizer_revision = specification.get("tokenizer_revision", specification["revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=tokenizer_revision,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        specification["id"],
        revision=specification["revision"],
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    return HuggingFaceQwen2Reader(model.to(device), tokenizer, specification=specification)
