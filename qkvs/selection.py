from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SelectionContext:
    """Causally available inputs for one token and transformer layer."""

    position: int
    layer: int
    history_end: int
    hidden: Any
    lower_layer: Any = None


@dataclass(frozen=True)
class HeatFieldProposal:
    """One bounded proposal with normalized center and resolution-relative spread."""

    center: float
    spread: float
    weight: float
    active: bool = True


@dataclass(frozen=True)
class PolicyState:
    """Bounded recurrent state retained for one policy layer."""

    centers: tuple[float, ...] = ()
    recurrent: Any = None


@dataclass(frozen=True)
class ProposalSet:
    proposals: tuple[HeatFieldProposal, ...]
    state: PolicyState
    soft_parameters: Any = None
    beta: float = 1.0


@dataclass(frozen=True)
class HistoricalSelection:
    """Historical token occurrences resolved without inspecting K/V contents."""

    indices: tuple[int, ...]
    proposals: tuple[HeatFieldProposal, ...]
    windows: tuple[tuple[int, ...], ...] = ()

    @property
    def occurrence_count(self) -> int:
        return len(self.indices)

    @property
    def unique_count(self) -> int:
        return len(set(self.indices))


class SelectionPolicy(Protocol):
    def propose(self, context: SelectionContext, state: PolicyState) -> ProposalSet: ...


class RollingOnlyPolicy:
    def propose(self, context: SelectionContext, state: PolicyState) -> ProposalSet:
        del context
        return ProposalSet((), state)


class FullQKVPolicy:
    """Explicit unbounded continuation reference; it is not an R-matched selector."""

    full_qkv = True

    def propose(self, context: SelectionContext, state: PolicyState) -> ProposalSet:
        del context
        return ProposalSet((), state)


class FixedOffsetPolicy:
    """A bounded control that proposes fixed distances behind the current token."""

    def __init__(self, offsets: tuple[int, ...]) -> None:
        if not offsets or any(offset <= 0 for offset in offsets):
            raise ValueError("fixed offsets must be positive")
        self.offsets = offsets

    def propose(self, context: SelectionContext, state: PolicyState) -> ProposalSet:
        if context.history_end <= 0:
            proposals = tuple(
                HeatFieldProposal(center=0.0, spread=1.0, weight=1.0, active=False)
                for _ in self.offsets
            )
            return ProposalSet(proposals, PolicyState(tuple(0.0 for _ in self.offsets)))

        centers: list[float] = []
        proposals: list[HeatFieldProposal] = []
        for offset in self.offsets:
            target_position = min(context.history_end - 1, max(0, context.position - offset))
            center = (target_position + 0.5) / context.history_end
            center = min(max(center, 0.0), 1.0)
            centers.append(center)
            proposals.append(
                HeatFieldProposal(
                    center=center,
                    spread=1.0,
                    weight=1.0,
                )
            )
        return ProposalSet(tuple(proposals), PolicyState(tuple(centers)))


class LearnedHeatFieldPolicy(nn.Module):
    """Independent bounded recurrent S pathway with per-layer proposal decoders."""

    def __init__(
        self,
        *,
        hidden_size: int,
        layers: int,
        proposal_count: int,
        latent_size: int,
        gamma: float,
        beta: float,
        minimum_spread: float,
        maximum_spread: float,
        gate_threshold: float,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or layers <= 0 or proposal_count <= 0 or latent_size <= 0:
            raise ValueError("learned heat-field dimensions must be positive")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must lie in [0, 1)")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if minimum_spread <= 0.0 or maximum_spread < minimum_spread:
            raise ValueError("learned spread bounds are invalid")
        if not 0.0 <= gate_threshold <= 1.0:
            raise ValueError("gate threshold must lie in [0, 1]")
        self.hidden_size = hidden_size
        self.layer_count = layers
        self.proposal_count = proposal_count
        self.gamma = gamma
        self.beta = beta
        self.minimum_spread = minimum_spread
        self.maximum_spread = maximum_spread
        self.gate_threshold = gate_threshold
        cross_layer_width = proposal_count * 4
        self.decoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size + cross_layer_width, latent_size),
                nn.Tanh(),
                nn.Linear(latent_size, cross_layer_width),
            )
            for _ in range(layers)
        )

    def _normalized_state(self, state: Tensor, history_end: int) -> Tensor:
        scale = float(max(1, history_end))
        normalized = state.clone()
        normalized[:, 0] = normalized[:, 0] / scale
        normalized[:, 1] = normalized[:, 1] / self.maximum_spread
        normalized[:, 2] = normalized[:, 2] / (1.0 + normalized[:, 2])
        return normalized

    def propose(self, context: SelectionContext, state: PolicyState) -> ProposalSet:
        if context.layer < 0 or context.layer >= self.layer_count:
            raise ValueError("selection layer is outside the learned policy")
        decoder = self.decoders[context.layer]
        parameter = next(decoder.parameters())
        hidden = context.hidden.reshape(-1).to(device=parameter.device, dtype=parameter.dtype)
        if hidden.numel() != self.hidden_size:
            raise ValueError("selection hidden state does not match the learned policy")
        if context.lower_layer is None:
            lower = torch.zeros(
                self.proposal_count,
                4,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        else:
            lower = context.lower_layer.to(device=hidden.device, dtype=hidden.dtype)
            if lower.shape != (self.proposal_count, 4):
                raise ValueError("cross-layer selection state has the wrong shape")
            lower = self._normalized_state(lower, context.history_end)
        raw = decoder(torch.cat((hidden, lower.reshape(-1)))).view(self.proposal_count, 4)
        history_scale = float(max(0, context.history_end - 1))
        candidate = torch.stack(
            (
                torch.sigmoid(raw[:, 0]) * history_scale,
                self.minimum_spread
                + torch.sigmoid(raw[:, 1]) * (self.maximum_spread - self.minimum_spread),
                torch.nn.functional.softplus(raw[:, 2]),
                torch.sigmoid(raw[:, 3]),
            ),
            dim=-1,
        )
        previous = state.recurrent
        if previous is None:
            previous = torch.zeros_like(candidate)
        else:
            previous = previous.to(device=candidate.device, dtype=candidate.dtype)
            if previous.shape != candidate.shape:
                raise ValueError("temporal selection state has the wrong shape")
        current = self.gamma * previous + (1.0 - self.gamma) * candidate
        current = torch.stack(
            (
                current[:, 0].clamp(0.0, history_scale),
                current[:, 1],
                current[:, 2],
                current[:, 3],
            ),
            dim=-1,
        )

        proposals = []
        normalized_centers = []
        for center_value, spread, weight, gate in current.detach().cpu().tolist():
            center = (center_value + 0.5) / context.history_end if context.history_end > 0 else 0.0
            center = min(max(center, 0.0), 1.0)
            normalized_centers.append(center)
            proposals.append(
                HeatFieldProposal(
                    center=center,
                    spread=spread,
                    weight=weight,
                    active=context.history_end > 0 and gate >= self.gate_threshold,
                )
            )
        return ProposalSet(
            tuple(proposals),
            PolicyState(tuple(normalized_centers), current),
            soft_parameters=current,
            beta=self.beta,
        )


def resolve_proposals(
    proposals: tuple[HeatFieldProposal, ...],
    *,
    history_end: int,
    window_width: int,
    historical_token_budget: int,
    beta: float = 1.0,
) -> HistoricalSelection:
    """Resolve J proposals directly in O(J * width), counting overlap occurrences."""

    if history_end < 0:
        raise ValueError("history end cannot be negative")
    if window_width <= 0:
        raise ValueError("window width must be positive")
    if historical_token_budget < 0:
        raise ValueError("historical token budget cannot be negative")
    if beta <= 0.0:
        raise ValueError("spatial beta must be positive")
    ordered = sorted(
        enumerate(proposals),
        key=lambda item: (-item[1].weight, item[0]),
    )
    indices: list[int] = []
    windows: list[tuple[int, ...]] = []
    for _, proposal in ordered:
        if not proposal.active or history_end == 0:
            continue
        if not 0.0 <= proposal.center <= 1.0:
            raise ValueError("proposal centers must lie in [0, 1]")
        if proposal.spread <= 0.0 or proposal.weight < 0.0:
            raise ValueError("proposal spread must be positive and weight nonnegative")
        proposal_width = max(1, round(window_width * proposal.spread / beta))
        proposal_width = min(proposal_width, history_end)
        if len(indices) + proposal_width > historical_token_budget:
            continue
        center_token = min(history_end - 1, int(proposal.center * history_end))
        start = center_token - proposal_width // 2
        start = min(max(0, start), max(0, history_end - proposal_width))
        end = min(history_end, start + proposal_width)
        window = tuple(range(start, end))
        windows.append(window)
        indices.extend(window)
    selection = HistoricalSelection(tuple(indices), proposals, tuple(windows))
    validate_historical_selection(
        selection,
        history_end=history_end,
        historical_token_budget=historical_token_budget,
    )
    return selection


def validate_historical_selection(
    selection: HistoricalSelection,
    *,
    history_end: int,
    historical_token_budget: int,
) -> None:
    if selection.occurrence_count > historical_token_budget:
        raise ValueError("historical selection exceeds the token-occurrence budget")
    if any(index < 0 or index >= history_end for index in selection.indices):
        raise ValueError("historical selection is noncausal or enters the rolling window")


def build_policy(
    condition: dict[str, Any],
    controls: dict[str, Any],
    learned_policy: SelectionPolicy | None = None,
    untrained_policy: SelectionPolicy | None = None,
) -> SelectionPolicy:
    policy = condition["policy"]
    if policy == "rolling":
        return RollingOnlyPolicy()
    if policy == "full_qkv":
        return FullQKVPolicy()
    if policy == "fixed_offsets":
        if len(condition["offsets"]) > controls["proposal_count"]:
            raise ValueError("fixed offsets exceed proposal capacity")
        return FixedOffsetPolicy(tuple(condition["offsets"]))
    if policy == "heatfield":
        if learned_policy is None:
            raise ValueError("learned heat-field policy requires a trained checkpoint")
        return learned_policy
    if policy == "heatfield_untrained":
        if untrained_policy is None:
            raise ValueError("untrained heat-field control requires its initialized policy")
        return untrained_policy
    raise ValueError(f"unsupported selection policy: {policy}")
