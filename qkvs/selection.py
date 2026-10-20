from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
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


LEGACY_RANDOM_BLOCK_SAMPLER_VERSION = "uniform-aligned-blocks-v1"
RANDOM_BLOCK_SAMPLER_VERSION = "uniform-aligned-blocks-v2"
COUNTER_RANDOM_BLOCK_SAMPLER_VERSION = "uniform-aligned-blocks-v3"
RANDOM_BLOCK_SAMPLER_VERSIONS = {
    LEGACY_RANDOM_BLOCK_SAMPLER_VERSION,
    RANDOM_BLOCK_SAMPLER_VERSION,
    COUNTER_RANDOM_BLOCK_SAMPLER_VERSION,
}

_SPLITMIX_INCREMENT = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_MULTIPLIER_1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_MULTIPLIER_2 = np.uint64(0x94D049BB133111EB)
_ALIGNMENT_DOMAIN = np.uint64(0xD1B54A32D192ED03)
_BLOCK_DOMAIN = np.uint64(0x8CB92BA72F3D8DD7)


@dataclass(frozen=True, eq=False)
class RandomBlockSelection:
    """One reproducible nonoverlapping block sample over a causal token range."""

    selected_blocks: np.ndarray
    seed: int
    boundary_alignment: int
    available_positions: int
    block_width: int
    aligned_block_count: int
    requested_fraction: float

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RandomBlockSelection)
            and np.array_equal(self.selected_blocks, other.selected_blocks)
            and self.seed == other.seed
            and self.boundary_alignment == other.boundary_alignment
            and self.available_positions == other.available_positions
            and self.block_width == other.block_width
            and self.aligned_block_count == other.aligned_block_count
            and self.requested_fraction == other.requested_fraction
        )

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

    def _block_ranges(self, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.boundary_alignment:
            starts = np.where(
                blocks == 0,
                0,
                self.boundary_alignment + (blocks - 1) * self.block_width,
            )
            stops = np.minimum(self.available_positions, starts + self.block_width)
            stops = np.where(
                blocks == 0,
                min(self.boundary_alignment, self.available_positions),
                stops,
            )
        else:
            starts = blocks * self.block_width
            stops = np.minimum(self.available_positions, starts + self.block_width)
        return starts, stops

    @property
    def ranges(self) -> tuple[tuple[int, int], ...]:
        starts, stops = self._block_ranges(self.selected_blocks)
        return tuple(zip(starts.tolist(), stops.tolist(), strict=True))

    @property
    def contiguous_ranges(self) -> tuple[tuple[int, int], ...]:
        if len(self.selected_blocks) == 0:
            return ()
        boundaries = np.concatenate(
            (
                np.array((0,), dtype=np.int64),
                np.flatnonzero(np.diff(self.selected_blocks) != 1) + 1,
                np.array((len(self.selected_blocks),), dtype=np.int64),
            )
        )
        run_starts = self.selected_blocks[boundaries[:-1]]
        run_stops = self.selected_blocks[boundaries[1:] - 1]
        starts, _ = self._block_ranges(run_starts)
        _, stops = self._block_ranges(run_stops)
        return tuple(zip(starts.tolist(), stops.tolist(), strict=True))

    @property
    def selected_token_count(self) -> int:
        if len(self.selected_blocks) == 0:
            return 0
        total = len(self.selected_blocks) * self.block_width
        selected_edges = []
        if int(self.selected_blocks[0]) == 0:
            selected_edges.append(0)
        last_block = self.aligned_block_count - 1
        if last_block and int(self.selected_blocks[-1]) == last_block:
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


@dataclass(frozen=True)
class RandomBlockSelectionBatch:
    """Native block arrays for a candidate-major, layer-major selection batch."""

    selected_blocks: np.ndarray
    block_offsets: np.ndarray
    boundary_alignments: np.ndarray
    selected_token_counts: np.ndarray
    base_occurrence_counts: np.ndarray
    candidate_count: int
    layer_count: int
    available_positions: int
    block_width: int
    requested_fraction: float

    @property
    def candidate_occurrence_counts(self) -> np.ndarray:
        return self.selected_token_counts.reshape(self.candidate_count, self.layer_count).sum(
            axis=1
        )


def derive_random_block_seed(base_seed: int, *coordinates: int) -> int:
    encoded = json.dumps((base_seed, *coordinates), separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _splitmix64(
    values: np.ndarray | np.uint64,
    *,
    copy: bool = True,
) -> np.ndarray:
    values = np.array(values, dtype=np.uint64, copy=copy)
    shifted = np.empty_like(values)
    np.add(values, _SPLITMIX_INCREMENT, out=values)
    np.right_shift(values, 30, out=shifted)
    np.bitwise_xor(values, shifted, out=values)
    np.multiply(values, _SPLITMIX_MULTIPLIER_1, out=values)
    np.right_shift(values, 27, out=shifted)
    np.bitwise_xor(values, shifted, out=values)
    np.multiply(values, _SPLITMIX_MULTIPLIER_2, out=values)
    np.right_shift(values, 31, out=shifted)
    np.bitwise_xor(values, shifted, out=values)
    return values


def derive_counter_random_block_seeds(
    base_seed: int | np.ndarray,
    *coordinates: int | np.ndarray,
) -> np.ndarray:
    """Derive replay-stable v3 keys with broadcast NumPy operations."""

    state = _splitmix64(np.asarray(base_seed, dtype=np.uint64))
    for coordinate in coordinates:
        state = _splitmix64(
            np.bitwise_xor(state, np.asarray(coordinate, dtype=np.uint64))
        )
    return np.asarray(state, dtype=np.uint64)


def _bounded_counter_draw(
    seeds: np.ndarray,
    bound: int,
    domain: np.uint64,
    coordinate: int,
) -> np.ndarray:
    """Map counter values uniformly into [0, bound) with deterministic rejection."""

    counter = _splitmix64(np.bitwise_xor(domain, np.uint64(coordinate)))
    random_values = _splitmix64(
        np.bitwise_xor(seeds, counter),
        copy=False,
    )
    rejection_threshold = (1 << 64) % bound
    rejected = random_values < rejection_threshold
    attempt = 1
    while np.any(rejected):
        retry_coordinate = np.uint64((attempt << 32) | coordinate)
        retry_counter = _splitmix64(np.bitwise_xor(domain, retry_coordinate))
        random_values[rejected] = _splitmix64(
            np.bitwise_xor(seeds[rejected], retry_counter),
            copy=False,
        )
        rejected = random_values < rejection_threshold
        attempt += 1
    return np.remainder(random_values, np.uint64(bound)).astype(np.int64)


def _sample_counter_subset(
    seeds: np.ndarray,
    population_size: int,
    sample_size: int,
) -> np.ndarray:
    """Run Floyd sampling concurrently over rows using counter-derived draws."""

    selected_mask = np.zeros((len(seeds), population_size), dtype=np.bool_)
    selected = np.empty((len(seeds), sample_size), dtype=np.int64)
    rows = np.arange(len(seeds))
    for output_index, upper_bound in enumerate(
        range(population_size - sample_size, population_size)
    ):
        bound = upper_bound + 1
        choices = _bounded_counter_draw(
            seeds,
            bound,
            _BLOCK_DOMAIN,
            output_index,
        )
        choices = np.where(selected_mask[rows, choices], upper_bound, choices)
        selected_mask[rows, choices] = True
        selected[:, output_index] = choices
    selected.sort(axis=1)
    return selected


def _sample_counter_random_block_batch(
    *,
    available_positions: int,
    block_width: int,
    requested_fraction: float,
    seeds: np.ndarray,
) -> RandomBlockSelectionBatch:
    """Select v3 blocks with batched counter draws and no per-selection generators."""

    flat_seeds = np.asarray(seeds, dtype=np.uint64).reshape(-1)
    if requested_fraction == 1.0:
        alignments = np.zeros(flat_seeds.size, dtype=np.int64)
    else:
        alignments = _bounded_counter_draw(
            flat_seeds,
            block_width,
            _ALIGNMENT_DOMAIN,
            0,
        )

    aligned_block_counts = np.full(
        flat_seeds.size,
        (available_positions + block_width - 1) // block_width,
        dtype=np.int64,
    )
    aligned = alignments != 0
    aligned_block_counts[aligned] = 1 + np.maximum(
        0,
        (
            available_positions
            - alignments[aligned]
            + block_width
            - 1
        )
        // block_width,
    )
    block_counts = np.rint(requested_fraction * aligned_block_counts).astype(np.int64)
    np.clip(block_counts, 1, aligned_block_counts, out=block_counts)
    block_offsets = np.empty(flat_seeds.size + 1, dtype=np.int64)
    block_offsets[0] = 0
    np.cumsum(block_counts, out=block_offsets[1:])
    selected_blocks = np.empty(block_offsets[-1], dtype=np.int64)
    selected_token_counts = np.empty(flat_seeds.size, dtype=np.int64)
    base_occurrence_counts = np.empty(flat_seeds.size, dtype=np.int64)

    for aligned_block_count in np.unique(aligned_block_counts):
        rows = np.flatnonzero(aligned_block_counts == aligned_block_count)
        block_count = int(block_counts[rows[0]])
        if block_count == aligned_block_count:
            selected = np.broadcast_to(
                np.arange(aligned_block_count, dtype=np.int64),
                (len(rows), aligned_block_count),
            )
        else:
            omitted_count = aligned_block_count - block_count
            if block_count <= omitted_count:
                selected = _sample_counter_subset(
                    flat_seeds[rows],
                    aligned_block_count,
                    block_count,
                )
            else:
                omitted = _sample_counter_subset(
                    flat_seeds[rows],
                    aligned_block_count,
                    omitted_count,
                )
                selected_mask = np.ones(
                    (len(rows), aligned_block_count),
                    dtype=np.bool_,
                )
                np.put_along_axis(selected_mask, omitted, False, axis=1)
                selected = np.nonzero(selected_mask)[1].reshape(len(rows), block_count)

        destinations = block_offsets[rows, None] + np.arange(block_count, dtype=np.int64)
        selected_blocks[destinations] = selected
        row_alignments = alignments[rows]
        token_counts = np.full(len(rows), block_count * block_width, dtype=np.int64)
        first_lengths = np.where(
            row_alignments,
            row_alignments,
            min(block_width, available_positions),
        )
        token_counts += np.where(
            selected[:, 0] == 0,
            first_lengths - block_width,
            0,
        )
        last_block = aligned_block_count - 1
        if last_block:
            last_starts = np.where(
                row_alignments,
                row_alignments + (last_block - 1) * block_width,
                last_block * block_width,
            )
            last_lengths = (
                np.minimum(available_positions, last_starts + block_width) - last_starts
            )
            token_counts += np.where(
                selected[:, -1] == last_block,
                last_lengths - block_width,
                0,
            )
        selected_token_counts[rows] = token_counts
        base_occurrence_counts[rows] = token_counts - (selected[:, -1] == last_block)

    for value in (
        selected_blocks,
        block_offsets,
        alignments,
        selected_token_counts,
        base_occurrence_counts,
    ):
        value.setflags(write=False)
    return RandomBlockSelectionBatch(
        selected_blocks=selected_blocks,
        block_offsets=block_offsets,
        boundary_alignments=alignments,
        selected_token_counts=selected_token_counts,
        base_occurrence_counts=base_occurrence_counts,
        candidate_count=seeds.shape[0],
        layer_count=seeds.shape[1],
        available_positions=available_positions,
        block_width=block_width,
        requested_fraction=requested_fraction,
    )


def _sample_uniform_random_block_array(
    *,
    available_positions: int,
    block_width: int,
    requested_fraction: float,
    seed: int,
    algorithm: str,
) -> tuple[np.ndarray, int, int]:
    if algorithm == COUNTER_RANDOM_BLOCK_SAMPLER_VERSION:
        batch = _sample_counter_random_block_batch(
            available_positions=available_positions,
            block_width=block_width,
            requested_fraction=requested_fraction,
            seeds=np.asarray(((seed,),), dtype=np.uint64),
        )
        alignment = int(batch.boundary_alignments[0])
        aligned_block_count = (
            (available_positions + block_width - 1) // block_width
            if not alignment
            else 1
            + max(
                0,
                (available_positions - alignment + block_width - 1) // block_width,
            )
        )
        return batch.selected_blocks, alignment, aligned_block_count
    if requested_fraction == 1.0:
        alignment = 0
    elif algorithm == LEGACY_RANDOM_BLOCK_SAMPLER_VERSION:
        alignment = random.Random(seed).randrange(block_width)
    else:
        generator = np.random.Generator(np.random.PCG64(seed))
        alignment = int(generator.integers(block_width))
    aligned_block_count = (available_positions + block_width - 1) // block_width
    if alignment:
        aligned_block_count = 1 + max(
            0,
            (available_positions - alignment + block_width - 1) // block_width,
        )
    if requested_fraction == 1.0:
        selected_blocks = np.arange(aligned_block_count, dtype=np.int64)
    elif algorithm == LEGACY_RANDOM_BLOCK_SAMPLER_VERSION:
        generator = random.Random(seed)
        generator.randrange(block_width)
        block_count = max(
            1,
            min(aligned_block_count, round(requested_fraction * aligned_block_count)),
        )
        selected_blocks = np.asarray(
            sorted(generator.sample(range(aligned_block_count), block_count)),
            dtype=np.int64,
        )
    else:
        block_count = max(
            1,
            min(aligned_block_count, round(requested_fraction * aligned_block_count)),
        )
        if block_count <= aligned_block_count - block_count:
            selected_blocks = np.sort(
                generator.choice(aligned_block_count, size=block_count, replace=False)
            )
        else:
            omitted = generator.choice(
                aligned_block_count,
                size=aligned_block_count - block_count,
                replace=False,
            )
            selected_mask = np.ones(aligned_block_count, dtype=np.bool_)
            selected_mask[omitted] = False
            selected_blocks = np.flatnonzero(selected_mask)
        selected_blocks = selected_blocks.astype(np.int64, copy=False)
    selected_blocks.setflags(write=False)
    return selected_blocks, alignment, aligned_block_count


def _selected_block_token_count(
    selected_blocks: np.ndarray,
    *,
    boundary_alignment: int,
    aligned_block_count: int,
    available_positions: int,
    block_width: int,
) -> int:
    total = len(selected_blocks) * block_width
    if int(selected_blocks[0]) == 0:
        first_stop = boundary_alignment or min(block_width, available_positions)
        total += first_stop - block_width
    last_block = aligned_block_count - 1
    if last_block and int(selected_blocks[-1]) == last_block:
        start = (
            boundary_alignment + (last_block - 1) * block_width
            if boundary_alignment
            else last_block * block_width
        )
        total += min(available_positions, start + block_width) - start - block_width
    return total


def sample_uniform_random_blocks(
    *,
    available_positions: int,
    block_width: int,
    requested_fraction: float,
    seed: int,
    algorithm: str = RANDOM_BLOCK_SAMPLER_VERSION,
) -> RandomBlockSelection:
    """Sample aligned causal blocks uniformly without replacement or overlap."""

    if available_positions <= 0 or block_width <= 0:
        raise ValueError("random block sampling dimensions must be positive")
    if not 0.0 < requested_fraction <= 1.0:
        raise ValueError("random block sampling fraction must lie in (0, 1]")
    if algorithm not in RANDOM_BLOCK_SAMPLER_VERSIONS:
        raise ValueError("random block sampling algorithm is unsupported")
    selected_blocks, alignment, aligned_block_count = _sample_uniform_random_block_array(
        available_positions=available_positions,
        block_width=block_width,
        requested_fraction=requested_fraction,
        seed=seed,
        algorithm=algorithm,
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


def sample_uniform_random_block_batch(
    *,
    available_positions: int,
    block_width: int,
    requested_fraction: float,
    seeds: np.ndarray,
    algorithm: str = RANDOM_BLOCK_SAMPLER_VERSION,
) -> RandomBlockSelectionBatch:
    """Sample a candidate/layer grid without materializing Python token ranges."""

    if available_positions <= 0 or block_width <= 0:
        raise ValueError("random block sampling dimensions must be positive")
    if not 0.0 < requested_fraction <= 1.0:
        raise ValueError("random block sampling fraction must lie in (0, 1]")
    if algorithm not in RANDOM_BLOCK_SAMPLER_VERSIONS:
        raise ValueError("random block sampling algorithm is unsupported")
    if seeds.ndim != 2 or not seeds.shape[0] or not seeds.shape[1]:
        raise ValueError("random block batch seeds must be a nonempty candidate/layer matrix")
    if algorithm == COUNTER_RANDOM_BLOCK_SAMPLER_VERSION:
        return _sample_counter_random_block_batch(
            available_positions=available_positions,
            block_width=block_width,
            requested_fraction=requested_fraction,
            seeds=seeds,
        )

    selected_arrays: list[np.ndarray] = []
    block_counts = np.empty(seeds.size, dtype=np.int64)
    alignments = np.empty(seeds.size, dtype=np.int64)
    selected_token_counts = np.empty(seeds.size, dtype=np.int64)
    base_occurrence_counts = np.empty(seeds.size, dtype=np.int64)
    for index, raw_seed in enumerate(seeds.flat):
        selected, alignment, aligned_block_count = _sample_uniform_random_block_array(
            available_positions=available_positions,
            block_width=block_width,
            requested_fraction=requested_fraction,
            seed=int(raw_seed),
            algorithm=algorithm,
        )
        selected_arrays.append(selected)
        block_counts[index] = len(selected)
        alignments[index] = alignment
        token_count = _selected_block_token_count(
            selected,
            boundary_alignment=alignment,
            aligned_block_count=aligned_block_count,
            available_positions=available_positions,
            block_width=block_width,
        )
        selected_token_counts[index] = token_count
        base_occurrence_counts[index] = token_count - int(
            int(selected[-1]) == aligned_block_count - 1
        )

    block_offsets = np.empty(seeds.size + 1, dtype=np.int64)
    block_offsets[0] = 0
    np.cumsum(block_counts, out=block_offsets[1:])
    selected_blocks = np.concatenate(selected_arrays)
    for value in (
        selected_blocks,
        block_offsets,
        alignments,
        selected_token_counts,
        base_occurrence_counts,
    ):
        value.setflags(write=False)
    return RandomBlockSelectionBatch(
        selected_blocks=selected_blocks,
        block_offsets=block_offsets,
        boundary_alignments=alignments,
        selected_token_counts=selected_token_counts,
        base_occurrence_counts=base_occurrence_counts,
        candidate_count=seeds.shape[0],
        layer_count=seeds.shape[1],
        available_positions=available_positions,
        block_width=block_width,
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
