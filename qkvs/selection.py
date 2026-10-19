from __future__ import annotations

import hashlib
import json
import random
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


RANDOM_BLOCK_SAMPLER_VERSION = "uniform-aligned-blocks-v1"


@dataclass(frozen=True)
class RandomBlockSelection:
    """One reproducible nonoverlapping block sample over a causal token range."""

    selected_blocks: tuple[int, ...]
    seed: int
    boundary_alignment: int
    available_positions: int
    block_width: int
    aligned_block_count: int
    requested_fraction: float

    @property
    def block_count(self) -> int:
        return len(self.selected_blocks)

    def block_range(self, block: int) -> tuple[int, int]:
        if not 0 <= block < self.aligned_block_count:
            raise IndexError("sampled block is out of range")
        if self.boundary_alignment:
            if block == 0:
                return 0, min(self.boundary_alignment, self.available_positions)
            start = self.boundary_alignment + (block - 1) * self.block_width
        else:
            start = block * self.block_width
        return start, min(self.available_positions, start + self.block_width)

    @property
    def ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.block_range(block) for block in self.selected_blocks)

    @property
    def contiguous_ranges(self) -> tuple[tuple[int, int], ...]:
        if not self.selected_blocks:
            return ()
        merged: list[tuple[int, int]] = []
        run_start = self.selected_blocks[0]
        previous = run_start
        for block in self.selected_blocks[1:]:
            if block != previous + 1:
                merged.append(
                    (self.block_range(run_start)[0], self.block_range(previous)[1])
                )
                run_start = block
            previous = block
        merged.append((self.block_range(run_start)[0], self.block_range(previous)[1]))
        return tuple(merged)

    @property
    def selected_token_count(self) -> int:
        if not self.selected_blocks:
            return 0
        total = len(self.selected_blocks) * self.block_width
        selected_edges = []
        if self.selected_blocks[0] == 0:
            selected_edges.append(0)
        last_block = self.aligned_block_count - 1
        if last_block and self.selected_blocks[-1] == last_block:
            selected_edges.append(last_block)
        for block in selected_edges:
            start, stop = self.block_range(block)
            total += stop - start - self.block_width
        return total

    @property
    def indices(self) -> tuple[int, ...]:
        """Materialize token positions for compatibility outside snapshot execution."""

        return tuple(
            position
            for start, stop in self.contiguous_ranges
            for position in range(start, stop)
        )

    @property
    def realized_fraction(self) -> float:
        return self.selected_token_count / self.available_positions


def derive_random_block_seed(base_seed: int, *coordinates: int) -> int:
    encoded = json.dumps((base_seed, *coordinates), separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def sample_uniform_random_blocks(
    *,
    available_positions: int,
    block_width: int,
    requested_fraction: float,
    seed: int,
) -> RandomBlockSelection:
    """Sample aligned causal blocks uniformly without replacement or overlap."""

    if available_positions <= 0 or block_width <= 0:
        raise ValueError("random block sampling dimensions must be positive")
    if not 0.0 < requested_fraction <= 1.0:
        raise ValueError("random block sampling fraction must lie in (0, 1]")
    if requested_fraction == 1.0:
        alignment = 0
    else:
        alignment = random.Random(seed).randrange(block_width)
    aligned_block_count = (available_positions + block_width - 1) // block_width
    if alignment:
        aligned_block_count = 1 + max(
            0,
            (available_positions - alignment + block_width - 1) // block_width,
        )
    if requested_fraction == 1.0:
        selected_blocks = tuple(range(aligned_block_count))
    else:
        generator = random.Random(seed)
        generator.randrange(block_width)
        block_count = max(
            1,
            min(aligned_block_count, round(requested_fraction * aligned_block_count)),
        )
        selected_blocks = tuple(
            sorted(generator.sample(range(aligned_block_count), block_count))
        )
    return RandomBlockSelection(
        selected_blocks=selected_blocks,
        seed=seed,
        boundary_alignment=alignment,
        available_positions=available_positions,
        block_width=block_width,
        aligned_block_count=aligned_block_count,
        requested_fraction=requested_fraction,
    )


class SelectionPolicy(Protocol):
    def propose(self, context: SelectionContext, state: PolicyState) -> ProposalSet: ...


class RollingOnlyPolicy:
    layer_invariant = True

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

    layer_invariant = True

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
