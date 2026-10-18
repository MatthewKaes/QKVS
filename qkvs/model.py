from __future__ import annotations

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
    proposal_centers: tuple[float, ...]
    proposal_spreads: tuple[float, ...]
    support_position: int | None
    history_end: int

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
                    historical = resolve_proposals(
                        proposal_set.proposals,
                        history_end=rolling_start,
                        window_width=window_width,
                        historical_token_budget=historical_token_budget,
                    )
                    rolling = tuple(range(rolling_start, position + 1))
                    actual = (*historical.indices, *rolling)
                    keys = torch.stack(key_cache[layer_index], dim=1)[:, actual]
                    values = torch.stack(value_cache[layer_index], dim=1)[:, actual]
                    logits = torch.einsum("hd,hnd->hn", query, keys) / math.sqrt(self.head_width)
                    weights = torch.softmax(logits, dim=-1)
                    context = torch.einsum("hn,hnd->hd", weights, values).reshape(self.width)
                    hidden = hidden + layer.output(context)
                    hidden = hidden + layer.mlp(layer.mlp_norm(hidden))
                    support = stream.support_positions[position]
                    token_accesses.append(
                        AccessRecord(
                            token_position=position,
                            layer=layer_index,
                            rolling_indices=rolling,
                            historical_indices=historical.indices,
                            actual_indices=actual,
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
                            support_position=support,
                            history_end=rolling_start,
                        )
                    )

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


def load_reader(specification: dict[str, Any], device: torch.device) -> FrozenCausalReader:
    if specification["backend"] != "synthetic":
        raise ValueError("only the CPU synthetic reader is implemented in this milestone")
    if not specification["frozen"]:
        raise ValueError("the research harness requires a frozen reader")
    reader = FrozenCausalReader(
        vocab_size=specification["vocab_size"],
        width=specification["width"],
        layers=specification["layers"],
        heads=specification["heads"],
        seed=specification["seed"],
    )
    return reader.to(device)
