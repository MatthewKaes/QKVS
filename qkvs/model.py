from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from qkvs.selection import (
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
                        ),
                        policy_states[layer_index],
                    )
                    policy_states[layer_index] = proposal_set.state
                    selection = resolve_proposals(
                        proposal_set.proposals,
                        history_end=rolling_start,
                        window_width=window_width,
                        historical_token_budget=historical_token_budget,
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
                            proposal.center
                            for proposal in proposal_set.proposals
                            if proposal.active
                        ),
                        proposal_spreads=tuple(
                            proposal.spread
                            for proposal in proposal_set.proposals
                            if proposal.active
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

                for layer_index, layer in enumerate(backbone.layers):
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
                    key_cache[layer_index].append(key[0, :, 0, :])
                    value_cache[layer_index].append(value[0, :, 0, :])

                    rolling_start = max(0, position - rolling_tokens + 1)
                    rolling = tuple(range(rolling_start, position + 1))
                    if diagnostic_full_history:
                        historical = tuple(range(rolling_start))
                        centers: tuple[float, ...] = ()
                        spreads: tuple[float, ...] = ()
                    else:
                        proposal_set = policy.propose(
                            SelectionContext(
                                position=position,
                                layer=layer_index,
                                history_end=rolling_start,
                                hidden=normalized[0, 0],
                            ),
                            policy_states[layer_index],
                        )
                        policy_states[layer_index] = proposal_set.state
                        selection = resolve_proposals(
                            proposal_set.proposals,
                            history_end=rolling_start,
                            window_width=window_width,
                            historical_token_budget=historical_token_budget,
                        )
                        historical = selection.indices
                        centers = tuple(
                            proposal.center
                            for proposal in proposal_set.proposals
                            if proposal.active
                        )
                        spreads = tuple(
                            proposal.spread
                            for proposal in proposal_set.proposals
                            if proposal.active
                        )
                    actual_indices = (*historical, *rolling)
                    occurrence_limit = (
                        None
                        if diagnostic_full_history
                        else rolling_tokens + historical_token_budget
                    )
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
                        proposal_centers=centers,
                        proposal_spreads=spreads,
                        support_position=stream.support_positions[position],
                        history_end=rolling_start,
                        cache_length=len(key_cache[layer_index]),
                        gathered_key_occurrences=keys.shape[1],
                        gathered_value_occurrences=values.shape[1],
                    )

                    keys = self._repeat_kv(
                        keys.unsqueeze(0), attention.num_key_value_groups
                    ).squeeze(0)
                    values = self._repeat_kv(
                        values.unsqueeze(0), attention.num_key_value_groups
                    ).squeeze(0)
                    query = query[0, :, 0, :]
                    attention_logits = torch.einsum("hd,hnd->hn", query, keys)
                    attention_logits = attention_logits * attention.scaling
                    weights = torch.softmax(attention_logits, dim=-1, dtype=torch.float32).to(
                        query.dtype
                    )
                    context = torch.einsum("hn,hnd->hd", weights, values)
                    attention_output = context.reshape(1, 1, -1).contiguous()
                    hidden = residual + attention.o_proj(attention_output)
                    residual = hidden
                    hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
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
