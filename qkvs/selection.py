from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SelectionContext:
    """Causally available inputs for one token and transformer layer."""

    position: int
    layer: int
    history_end: int
    hidden: Any


@dataclass(frozen=True)
class HeatFieldProposal:
    """One bounded proposal; spread is reserved and must be 1 in this milestone."""

    center: float
    spread: float
    weight: float
    active: bool = True


@dataclass(frozen=True)
class PolicyState:
    """Bounded recurrent state retained for one policy layer."""

    centers: tuple[float, ...] = ()


@dataclass(frozen=True)
class ProposalSet:
    proposals: tuple[HeatFieldProposal, ...]
    state: PolicyState


@dataclass(frozen=True)
class HistoricalSelection:
    """Historical token occurrences resolved without inspecting K/V contents."""

    indices: tuple[int, ...]
    proposals: tuple[HeatFieldProposal, ...]

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


def resolve_proposals(
    proposals: tuple[HeatFieldProposal, ...],
    *,
    history_end: int,
    window_width: int,
    historical_token_budget: int,
) -> HistoricalSelection:
    """Resolve J proposals directly in O(J * width), counting overlap occurrences."""

    if history_end < 0:
        raise ValueError("history end cannot be negative")
    if window_width <= 0:
        raise ValueError("window width must be positive")
    if historical_token_budget < 0:
        raise ValueError("historical token budget cannot be negative")
    ordered = sorted(
        enumerate(proposals),
        key=lambda item: (-item[1].weight, item[0]),
    )
    indices: list[int] = []
    for _, proposal in ordered:
        if not proposal.active or history_end == 0:
            continue
        if not 0.0 <= proposal.center <= 1.0:
            raise ValueError("proposal centers must lie in [0, 1]")
        if proposal.spread <= 0.0 or proposal.weight < 0.0:
            raise ValueError("proposal spread must be positive and weight nonnegative")
        if proposal.spread != 1.0:
            raise ValueError("proposal spread and spatial beta are not implemented")
        if len(indices) + window_width > historical_token_budget:
            break
        center_token = min(history_end - 1, int(proposal.center * history_end))
        start = center_token - window_width // 2
        start = min(max(0, start), max(0, history_end - window_width))
        end = min(history_end, start + window_width)
        indices.extend(range(start, end))
    selection = HistoricalSelection(tuple(indices), proposals)
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


def build_policy(condition: dict[str, Any], controls: dict[str, Any]) -> SelectionPolicy:
    policy = condition["policy"]
    if policy == "rolling":
        return RollingOnlyPolicy()
    if policy == "fixed_offsets":
        if len(condition["offsets"]) > controls["proposal_count"]:
            raise ValueError("fixed offsets exceed proposal capacity")
        return FixedOffsetPolicy(tuple(condition["offsets"]))
    if policy == "heatfield":
        raise ValueError("learned heat-field policy is not implemented or authorized")
    raise ValueError(f"unsupported selection policy: {policy}")
