from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
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
class AccessRecord:
    token_position: int
    layer: int
    rolling_indices: tuple[int, ...]
    historical_indices: tuple[int, ...]
    actual_indices: tuple[int, ...]
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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenScore:
    position: int
    target: int
    loss: float
    prediction: int
    accesses: tuple[AccessRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "target": self.target,
            "loss": self.loss,
            "prediction": self.prediction,
            "accesses": [access.as_dict() for access in self.accesses],
        }


def _access_record(
    *,
    position: int,
    layer: int,
    rolling: tuple[int, ...],
    historical: tuple[int, ...],
    proposal_centers: tuple[float, ...],
    proposal_spreads: tuple[float, ...],
    proposal_weights: tuple[float, ...],
    proposal_active: tuple[bool, ...],
    support_position: int | None,
    history_end: int,
    cache_length: int,
    gathered_key_occurrences: int,
    gathered_value_occurrences: int,
) -> AccessRecord:
    rolling_set = set(rolling)
    unique_historical = len(set(historical))
    return AccessRecord(
        token_position=position,
        layer=layer,
        rolling_indices=rolling,
        historical_indices=historical,
        actual_indices=(*historical, *rolling),
        historical_occurrences=len(historical),
        historical_unique_tokens=unique_historical,
        duplicate_historical_occurrences=len(historical) - unique_historical,
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


def _gather_cache_occurrences(
    cache: list[Tensor],
    indices: tuple[int, ...],
    *,
    occurrence_limit: int | None,
) -> Tensor:
    """Stack only executed cache occurrences, preserving order and duplicates."""

    if occurrence_limit is not None and len(indices) > occurrence_limit:
        raise ValueError("actual K/V gather exceeds the bounded occurrence limit")
    cache_length = len(cache)
    if any(index < 0 or index >= cache_length for index in indices):
        raise ValueError("actual K/V gather index is outside the layer cache")
    selected = [cache[index] for index in indices]
    if not selected:
        raise ValueError("actual K/V gather must include the rolling window")
    return torch.stack(selected, dim=1)


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
    ) -> list[TokenScore]:
        if rolling_tokens <= 0:
            raise ValueError("rolling attention is mandatory")
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

                    rolling_start = max(0, position - rolling_tokens + 1)
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
                    rolling = tuple(range(rolling_start, position + 1))
                    actual_indices = (*selection.indices, *rolling)
                    occurrence_limit = rolling_tokens + historical_token_budget
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
                        historical=selection.indices,
                        proposal_centers=tuple(
                            proposal.center for proposal in proposal_set.proposals
                        ),
                        proposal_spreads=tuple(
                            proposal.spread for proposal in proposal_set.proposals
                        ),
                        proposal_weights=tuple(
                            proposal.weight for proposal in proposal_set.proposals
                        ),
                        proposal_active=tuple(
                            proposal.active for proposal in proposal_set.proposals
                        ),
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
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

        self._apply_rotary_pos_emb = apply_rotary_pos_emb
        self._repeat_kv = repeat_kv

    def identity(self) -> dict[str, Any]:
        import transformers

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
        return {
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

    def _append_layer_kv(
        self,
        layer: Any,
        hidden: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        key_cache: list[Tensor],
        value_cache: list[Tensor],
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
        key_cache: list[Tensor],
        value_cache: list[Tensor],
        actual_indices: tuple[int, ...],
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
        key_cache: list[Tensor],
        value_cache: list[Tensor],
        historical_occupancy: Tensor,
        rolling: tuple[int, ...],
    ) -> Tensor:
        history_end = historical_occupancy.numel()
        actual_indices = (*range(history_end), *rolling)
        keys = _gather_cache_occurrences(key_cache, actual_indices, occurrence_limit=None)
        values = _gather_cache_occurrences(value_cache, actual_indices, occurrence_limit=None)
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
        key_cache: list[list[Tensor]],
        value_cache: list[list[Tensor]],
        rolling: tuple[int, ...],
    ) -> Tensor:
        backbone = self.model.model
        for layer_index in range(start_layer, len(backbone.layers)):
            layer = backbone.layers[layer_index]
            branch_keys = list(key_cache[layer_index])
            branch_values = list(value_cache[layer_index])
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
    ) -> tuple[list[TokenScore], Tensor | None]:
        if rolling_tokens <= 0:
            raise ValueError("rolling attention is mandatory")
        if len(stream.tokens) > self.model.config.max_position_embeddings:
            raise ValueError("token stream exceeds the model's supported position range")
        self.model.to(device)
        backbone = self.model.model
        key_cache: list[list[Tensor]] = [[] for _ in backbone.layers]
        value_cache: list[list[Tensor]] = [[] for _ in backbone.layers]
        policy_states = [PolicyState() for _ in backbone.layers]
        scores: list[TokenScore] = []
        output_logits: list[Tensor] | None = [] if diagnostic_full_history else None

        with torch.inference_mode():
            for position, token in enumerate(stream.tokens[:-1]):
                token_id = torch.tensor([[token]], device=device)
                hidden = backbone.embed_tokens(token_id)
                position_ids = torch.tensor([[position]], device=device)
                position_embeddings = backbone.rotary_emb(hidden, position_ids)
                token_accesses: list[AccessRecord] = []
                lower_layer_selection = None

                for layer_index, layer in enumerate(backbone.layers):
                    residual, normalized, query = self._append_layer_kv(
                        layer,
                        hidden,
                        position_embeddings,
                        key_cache[layer_index],
                        value_cache[layer_index],
                    )

                    rolling_start = max(0, position - rolling_tokens + 1)
                    rolling = tuple(range(rolling_start, position + 1))
                    if diagnostic_full_history:
                        historical = tuple(range(rolling_start))
                        centers: tuple[float, ...] = ()
                        spreads: tuple[float, ...] = ()
                        weights: tuple[float, ...] = ()
                        active: tuple[bool, ...] = ()
                    else:
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
                        centers = tuple(proposal.center for proposal in proposal_set.proposals)
                        spreads = tuple(proposal.spread for proposal in proposal_set.proposals)
                        weights = tuple(proposal.weight for proposal in proposal_set.proposals)
                        active = tuple(proposal.active for proposal in proposal_set.proposals)
                    actual_indices = (*historical, *rolling)
                    occurrence_limit = (
                        None
                        if diagnostic_full_history
                        else rolling_tokens + historical_token_budget
                    )
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
                        proposal_centers=centers,
                        proposal_spreads=spreads,
                        proposal_weights=weights,
                        proposal_active=active,
                        support_position=stream.support_positions[position],
                        history_end=rolling_start,
                        cache_length=len(key_cache[layer_index]),
                        gathered_key_occurrences=gathered_keys,
                        gathered_value_occurrences=gathered_values,
                    )
                    token_accesses.append(access)

                logits = self.model.lm_head(backbone.norm(hidden))[0, 0]
                if output_logits is not None:
                    output_logits.append(logits.detach().cpu())
                target = stream.tokens[position + 1]
                loss = torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), torch.tensor([target], device=device)
                )
                scores.append(
                    TokenScore(
                        position=position,
                        target=target,
                        loss=float(loss.item()),
                        prediction=int(logits.argmax().item()),
                        accesses=tuple(token_accesses),
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
    ) -> list[TokenScore]:
        scores, _ = self._execute(
            stream,
            policy,
            rolling_tokens=rolling_tokens,
            historical_token_budget=historical_token_budget,
            window_width=window_width,
            device=device,
            diagnostic_full_history=False,
        )
        return scores

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
    ) -> tuple[Tensor, dict[str, Any]]:
        """Build one truncated causal S objective while the reader remains frozen."""

        if position < rolling_tokens or position >= len(stream.tokens) - 1:
            raise ValueError("training position needs eligible history and a next-token target")
        if layer_index < 0 or layer_index >= len(self.model.model.layers):
            raise ValueError("training layer is outside the reader")
        if coverage_weight < 0.0 or utility_weight < 0.0:
            raise ValueError("training signal weights must be nonnegative")
        self.model.to(device)
        policy.to(device)
        backbone = self.model.model
        key_cache: list[list[Tensor]] = [[] for _ in backbone.layers]
        value_cache: list[list[Tensor]] = [[] for _ in backbone.layers]
        policy_states = [PolicyState() for _ in backbone.layers]
        trajectory_historical_occurrences = 0
        trajectory_unique_tokens = 0
        trajectory_overlap_occurrences = 0

        with torch.no_grad():
            for prior_position in range(position):
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
                        (*trajectory_selection.indices, *rolling),
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
                    (*trajectory_selection.indices, *rolling),
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
        if proposal_set.soft_parameters is None:
            raise RuntimeError("learned policy did not provide differentiable proposals")
        selection = resolve_proposals(
            proposal_set.proposals,
            history_end=rolling_start,
            window_width=window_width,
            historical_token_budget=historical_token_budget,
            beta=proposal_set.beta,
        )
        target = torch.tensor([stream.tokens[position + 1]], device=device)

        with torch.no_grad():
            rolling_hidden, _, _ = self._finish_layer(
                target_layer,
                residual,
                query,
                key_cache[layer_index],
                value_cache[layer_index],
                rolling,
                occurrence_limit=rolling_tokens,
            )
            rolling_logits = self._complete_current_token(
                rolling_hidden,
                start_layer=layer_index + 1,
                position_embeddings=position_embeddings,
                key_cache=key_cache,
                value_cache=value_cache,
                rolling=rolling,
            )
            rolling_loss = torch.nn.functional.cross_entropy(rolling_logits.unsqueeze(0), target)
            selected_indices = (*selection.indices, *rolling)
            selected_hidden, _, _ = self._finish_layer(
                target_layer,
                residual,
                query,
                key_cache[layer_index],
                value_cache[layer_index],
                selected_indices,
                occurrence_limit=rolling_tokens + historical_token_budget,
            )
            selected_logits = self._complete_current_token(
                selected_hidden,
                start_layer=layer_index + 1,
                position_embeddings=position_embeddings,
                key_cache=key_cache,
                value_cache=value_cache,
                rolling=rolling,
            )
            selected_loss = torch.nn.functional.cross_entropy(selected_logits.unsqueeze(0), target)

            full_indices = tuple(range(position + 1))
            teacher_keys = _gather_cache_occurrences(
                key_cache[layer_index], full_indices, occurrence_limit=None
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
            teacher_history = teacher_history / teacher_history.sum().clamp_min(1e-12)

        soft = proposal_set.soft_parameters
        historical_positions = torch.arange(
            rolling_start,
            device=soft.device,
            dtype=soft.dtype,
        )
        centers = soft[:, 0:1]
        spreads = (window_width * soft[:, 1:2] / proposal_set.beta).clamp_min(0.5)
        importance = soft[:, 2:3]
        gates = soft[:, 3:4]
        distance = (historical_positions.unsqueeze(0) - centers) / spreads
        kernels = torch.exp(-0.5 * distance.square())
        proposal_occupancy = gates * (1.0 - torch.exp(-importance * kernels))
        proposal_occupancy = proposal_occupancy.clamp(0.0, 1.0 - 1e-6)
        collective_occupancy = 1.0 - torch.prod(1.0 - proposal_occupancy, dim=0)
        coverage_reward = torch.sum(teacher_history.to(soft.dtype) * collective_occupancy)
        coverage_loss = 1.0 - coverage_reward

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
            rolling=rolling,
        )
        soft_loss = torch.nn.functional.cross_entropy(soft_logits.unsqueeze(0), target)
        utility_surrogate = soft_loss - rolling_loss.detach()
        objective = coverage_weight * coverage_loss + utility_weight * utility_surrogate
        metrics = {
            "position": position,
            "layer": layer_index,
            "objective": float(objective.detach()),
            "coverage_loss": float(coverage_loss.detach()),
            "soft_next_token_loss": float(soft_loss.detach()),
            "rolling_next_token_loss": float(rolling_loss),
            "hard_selected_next_token_loss": float(selected_loss),
            "hard_utility": float(rolling_loss - selected_loss),
            "soft_utility": float(rolling_loss - soft_loss.detach()),
            "soft_hard_utility_gap": float(selected_loss - soft_loss.detach()),
            "teacher_historical_attention_mass": float(teacher_weights[:rolling_start].sum()),
            "historical_indices": selection.indices,
            "historical_occurrences": selection.occurrence_count,
            "historical_unique_tokens": selection.unique_count,
            "overlap_occurrences": selection.occurrence_count - selection.unique_count,
            "proposal_parameters": soft.detach().cpu().tolist(),
            "coverage_overlap_alpha": 0.0,
            "trajectory": "learned_hard_detached",
            "trajectory_historical_occurrences": trajectory_historical_occurrences,
            "trajectory_unique_tokens": trajectory_unique_tokens,
            "trajectory_overlap_occurrences": trajectory_overlap_occurrences,
            "temporal_credit_assignment": "target_step_only",
        }
        return objective, metrics

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
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    return HuggingFaceQwen2Reader(model.to(device), tokenizer, specification=specification)
