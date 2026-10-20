from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from qkvs.selection import (
    COUNTER_RANDOM_BLOCK_SAMPLER_VERSION,
    LEGACY_RANDOM_BLOCK_SAMPLER_VERSION,
    RANDOM_BLOCK_SAMPLER_VERSION,
    FixedOffsetPolicy,
    HeatFieldProposal,
    HistoricalSelection,
    LearnedHeatFieldPolicy,
    PolicyState,
    SelectionContext,
    derive_counter_random_block_seeds,
    derive_random_block_seed,
    resolve_proposals,
    sample_uniform_random_block_batch,
    sample_uniform_random_blocks,
    validate_historical_selection,
)


def test_collective_resolution_counts_overlapping_occurrences() -> None:
    proposals = (
        HeatFieldProposal(center=0.5, spread=1.0, weight=1.0),
        HeatFieldProposal(center=0.5, spread=1.0, weight=0.5),
    )
    selection = resolve_proposals(
        proposals,
        history_end=12,
        window_width=2,
        historical_token_budget=4,
    )
    assert selection.indices == (5, 6, 5, 6)
    assert selection.occurrence_count == 4
    assert selection.unique_count == 2


def test_uniform_random_blocks_are_reproducible_nonoverlapping_and_unanchored() -> None:
    seed = derive_random_block_seed(71, 99, 20, 3, 4)
    first = sample_uniform_random_blocks(
        available_positions=100,
        block_width=8,
        requested_fraction=0.2,
        seed=seed,
    )
    second = sample_uniform_random_blocks(
        available_positions=100,
        block_width=8,
        requested_fraction=0.2,
        seed=seed,
    )

    assert RANDOM_BLOCK_SAMPLER_VERSION == "uniform-aligned-blocks-v2"
    assert first == second
    assert first.block_count == 3
    assert len(first.indices) == len(set(first.indices))
    assert all(
        stop <= next_start
        for (_, stop), (next_start, _) in zip(first.ranges, first.ranges[1:], strict=False)
    )
    assert 0.0 < first.realized_fraction < 1.0

    selections = [
        sample_uniform_random_blocks(
            available_positions=100,
            block_width=8,
            requested_fraction=0.1,
            seed=derive_random_block_seed(71, trial),
        )
        for trial in range(400)
    ]
    assert any(0 not in selection.indices for selection in selections)
    assert any(99 not in selection.indices for selection in selections)
    assert any(99 in selection.indices for selection in selections)
    mean_position = sum(
        sum(selection.indices) / len(selection.indices) for selection in selections
    ) / len(selections)
    assert mean_position == pytest.approx(49.5, abs=2.0)


def test_uniform_random_blocks_keep_one_position_and_dense_control_is_exact() -> None:
    sparse = sample_uniform_random_blocks(
        available_positions=3,
        block_width=8,
        requested_fraction=0.01,
        seed=5,
    )
    dense = sample_uniform_random_blocks(
        available_positions=17,
        block_width=8,
        requested_fraction=1.0,
        seed=999,
    )

    assert sparse.indices
    assert dense.indices == tuple(range(17))
    assert dense.ranges == ((0, 8), (8, 16), (16, 17))
    assert dense.realized_fraction == 1.0


@pytest.mark.parametrize("requested_fraction", (0.1, 0.5, 0.99, 1.0))
def test_native_random_block_batch_preserves_scalar_samples(requested_fraction) -> None:
    seeds = np.asarray(
        [
            [derive_random_block_seed(71, candidate, layer) for layer in range(3)]
            for candidate in range(4)
        ],
        dtype=np.uint64,
    )
    batch = sample_uniform_random_block_batch(
        available_positions=101,
        block_width=8,
        requested_fraction=requested_fraction,
        seeds=seeds,
    )

    expected_occurrences = np.empty_like(batch.selected_token_counts)
    expected_base = np.empty_like(batch.base_occurrence_counts)
    for selection_index, seed in enumerate(seeds.flat):
        scalar = sample_uniform_random_blocks(
            available_positions=101,
            block_width=8,
            requested_fraction=requested_fraction,
            seed=int(seed),
        )
        start = batch.block_offsets[selection_index]
        stop = batch.block_offsets[selection_index + 1]
        assert np.array_equal(batch.selected_blocks[start:stop], scalar.selected_blocks)
        assert batch.boundary_alignments[selection_index] == scalar.boundary_alignment
        expected_occurrences[selection_index] = scalar.selected_token_count
        expected_base[selection_index] = scalar.selected_token_count - int(100 in scalar.indices)

    assert np.array_equal(batch.selected_token_counts, expected_occurrences)
    assert np.array_equal(batch.base_occurrence_counts, expected_base)
    assert np.array_equal(
        batch.candidate_occurrence_counts,
        expected_occurrences.reshape(4, 3).sum(axis=1),
    )


@pytest.mark.parametrize("requested_fraction", (0.1, 0.5, 0.9, 1.0))
def test_counter_random_block_batch_is_replayable_across_batch_boundaries(
    requested_fraction,
    monkeypatch,
) -> None:
    trial_ids = np.arange(400, 407, dtype=np.uint64)
    candidate_seeds = derive_counter_random_block_seeds(
        71,
        22_000,
        90_000,
        trial_ids,
    )
    layer_seeds = derive_counter_random_block_seeds(
        candidate_seeds[:, None],
        np.arange(3, dtype=np.uint64)[None, :],
    )
    expected_candidate_seeds = [
        15321721335827543043,
        12011508823295275547,
        17679342620259676627,
        9428177127233054781,
    ]
    assert candidate_seeds[:4].tolist() == expected_candidate_seeds

    def unexpected_pcg64(*args, **kwargs):
        raise AssertionError("v3 must not construct per-selection PCG64 generators")

    monkeypatch.setattr(np.random, "PCG64", unexpected_pcg64)
    complete = sample_uniform_random_block_batch(
        available_positions=22_001,
        block_width=8,
        requested_fraction=requested_fraction,
        seeds=layer_seeds,
        algorithm=COUNTER_RANDOM_BLOCK_SAMPLER_VERSION,
    )
    chunks = tuple(
        sample_uniform_random_block_batch(
            available_positions=22_001,
            block_width=8,
            requested_fraction=requested_fraction,
            seeds=layer_seeds[start:stop],
            algorithm=COUNTER_RANDOM_BLOCK_SAMPLER_VERSION,
        )
        for start, stop in ((0, 2), (2, 5), (5, 7))
    )

    def selection_records(batch):
        return tuple(
            (
                int(batch.boundary_alignments[index]),
                int(batch.selected_token_counts[index]),
                int(batch.base_occurrence_counts[index]),
                tuple(
                    int(value)
                    for value in batch.selected_blocks[
                        batch.block_offsets[index] : batch.block_offsets[index + 1]
                    ]
                ),
            )
            for index in range(batch.candidate_count * batch.layer_count)
        )

    assert selection_records(complete) == tuple(
        record for chunk in chunks for record in selection_records(chunk)
    )
    for selection_index, seed in enumerate(layer_seeds.flat):
        scalar = sample_uniform_random_blocks(
            available_positions=22_001,
            block_width=8,
            requested_fraction=requested_fraction,
            seed=int(seed),
            algorithm=COUNTER_RANDOM_BLOCK_SAMPLER_VERSION,
        )
        assert selection_records(complete)[selection_index] == (
            scalar.boundary_alignment,
            scalar.selected_token_count,
            scalar.selected_token_count - int(22_000 in scalar.indices),
            tuple(int(value) for value in scalar.selected_blocks),
        )


def test_counter_random_block_scores_distribute_selections_across_blocks() -> None:
    candidate_seeds = derive_counter_random_block_seeds(
        19,
        np.arange(4_096, dtype=np.uint64),
    )
    batch = sample_uniform_random_block_batch(
        available_positions=65,
        block_width=1,
        requested_fraction=0.2,
        seeds=candidate_seeds[:, None],
        algorithm=COUNTER_RANDOM_BLOCK_SAMPLER_VERSION,
    )
    frequencies = np.bincount(batch.selected_blocks, minlength=65)

    assert batch.selected_blocks.size == 4_096 * 13
    assert frequencies.mean() == pytest.approx(4_096 * 0.2)
    assert np.max(np.abs(frequencies - frequencies.mean())) < 100

    alignment_seeds = derive_counter_random_block_seeds(
        23,
        np.arange(7_000, dtype=np.uint64),
    )
    alignment_batch = sample_uniform_random_block_batch(
        available_positions=17,
        block_width=7,
        requested_fraction=0.5,
        seeds=alignment_seeds[:, None],
        algorithm=COUNTER_RANDOM_BLOCK_SAMPLER_VERSION,
    )
    alignment_frequencies = np.bincount(
        alignment_batch.boundary_alignments,
        minlength=7,
    )
    assert np.max(np.abs(alignment_frequencies - 1_000)) < 100


def test_compact_uniform_random_blocks_preserve_v1_samples_exactly() -> None:
    def reference_ranges(
        available_positions: int,
        block_width: int,
        requested_fraction: float,
        seed: int,
    ) -> tuple[tuple[int, int], ...]:
        alignment = (
            0 if requested_fraction == 1.0 else random.Random(seed).randrange(block_width)
        )
        ranges: list[tuple[int, int]] = []
        if alignment:
            ranges.append((0, min(alignment, available_positions)))
        start = alignment
        while start < available_positions:
            stop = min(available_positions, start + block_width)
            ranges.append((start, stop))
            start = stop
        if requested_fraction == 1.0:
            return tuple(ranges)
        generator = random.Random(seed)
        generator.randrange(block_width)
        count = max(1, min(len(ranges), round(requested_fraction * len(ranges))))
        return tuple(
            ranges[index] for index in sorted(generator.sample(range(len(ranges)), count))
        )

    for available_positions, block_width, requested_fraction, seed in (
        (3, 8, 0.01, 5),
        (100, 8, 0.2, 71),
        (32_550, 8, 0.99, 991),
        (17, 8, 1.0, 999),
    ):
        sample = sample_uniform_random_blocks(
            available_positions=available_positions,
            block_width=block_width,
            requested_fraction=requested_fraction,
            seed=seed,
            algorithm=LEGACY_RANDOM_BLOCK_SAMPLER_VERSION,
        )
        expected_ranges = reference_ranges(
            available_positions,
            block_width,
            requested_fraction,
            seed,
        )
        expected_indices = tuple(
            position
            for start, stop in expected_ranges
            for position in range(start, stop)
        )
        assert sample.ranges == expected_ranges
        assert sample.indices == expected_indices
        assert sample.selected_token_count == len(expected_indices)


def test_resolution_is_direct_causal_and_budgeted() -> None:
    proposals = tuple(
        HeatFieldProposal(center=index / 4, spread=1.0, weight=1.0) for index in range(4)
    )
    selection = resolve_proposals(
        proposals,
        history_end=9,
        window_width=2,
        historical_token_budget=4,
    )
    assert len(selection.indices) == 4
    assert all(0 <= index < 9 for index in selection.indices)

    with pytest.raises(ValueError, match="noncausal"):
        validate_historical_selection(
            HistoricalSelection((9,), proposals),
            history_end=9,
            historical_token_budget=4,
        )


def test_fixed_offsets_keep_bounded_layer_state_without_unsupported_controls() -> None:
    policy = FixedOffsetPolicy((8,))
    first = policy.propose(
        SelectionContext(position=20, layer=0, history_end=17, hidden=object()),
        PolicyState(),
    )
    second = policy.propose(
        SelectionContext(position=21, layer=0, history_end=18, hidden=object()),
        first.state,
    )
    assert len(second.state.centers) == 1
    assert second.proposals[0].spread == 1.0
    assert second.state.centers[0] != first.state.centers[0]
    assert 0.0 <= second.state.centers[0] <= 1.0


def test_spread_and_beta_change_direct_spatial_resolution() -> None:
    proposal = (HeatFieldProposal(center=0.5, spread=2.0, weight=1.0),)
    broad = resolve_proposals(
        proposal,
        history_end=20,
        window_width=2,
        historical_token_budget=8,
        beta=1.0,
    )
    concentrated = resolve_proposals(
        proposal,
        history_end=20,
        window_width=2,
        historical_token_budget=8,
        beta=2.0,
    )
    assert broad.occurrence_count == 4
    assert concentrated.occurrence_count == 2
    assert broad.windows != concentrated.windows


def test_proposal_capacity_window_width_and_occurrence_budget_are_independent() -> None:
    proposals = tuple(
        HeatFieldProposal(center=index / 3, spread=1.0, weight=1.0) for index in range(4)
    )
    selection = resolve_proposals(
        proposals,
        history_end=16,
        window_width=3,
        historical_token_budget=7,
    )
    assert selection.occurrence_count == 6
    assert len(selection.proposals) == 4


def _learned_policy(gamma: float) -> LearnedHeatFieldPolicy:
    torch.manual_seed(41)
    return LearnedHeatFieldPolicy(
        hidden_size=8,
        layers=2,
        proposal_count=2,
        latent_size=12,
        gamma=gamma,
        beta=1.5,
        minimum_spread=0.5,
        maximum_spread=2.0,
        gate_threshold=0.5,
    )


def test_gamma_controls_bounded_temporal_persistence_and_absolute_anchoring() -> None:
    responsive = _learned_policy(0.0)
    persistent = _learned_policy(0.9)
    persistent.load_state_dict(responsive.state_dict())
    first_context = SelectionContext(
        position=12,
        layer=0,
        history_end=10,
        hidden=torch.linspace(-1.0, 1.0, 8),
    )
    first = responsive.propose(first_context, PolicyState())
    first_persistent = persistent.propose(first_context, PolicyState())
    second_context = SelectionContext(
        position=13,
        layer=0,
        history_end=11,
        hidden=torch.linspace(1.0, -1.0, 8),
    )
    responsive_second = responsive.propose(second_context, first.state)
    persistent_second = persistent.propose(second_context, first_persistent.state)
    assert torch.allclose(first_persistent.state.recurrent, 0.1 * first.state.recurrent)
    assert torch.allclose(
        persistent_second.state.recurrent,
        0.9 * first_persistent.state.recurrent + 0.1 * responsive_second.state.recurrent,
    )
    assert persistent_second.state.recurrent.shape == (2, 4)


def test_multiple_peaks_gates_and_cross_layer_refinement() -> None:
    policy = _learned_policy(0.0)
    with torch.no_grad():
        output = policy.decoders[0][-1]
        output.weight.zero_()
        output.bias.zero_()
        output.bias[0] = -3.0
        output.bias[3] = 3.0
        output.bias[4] = 3.0
        output.bias[7] = -3.0
    proposals = policy.propose(
        SelectionContext(position=16, layer=0, history_end=12, hidden=torch.ones(8)),
        PolicyState(),
    )
    assert proposals.proposals[0].active
    assert not proposals.proposals[1].active
    assert proposals.proposals[0].center < proposals.proposals[1].center
    resolved = resolve_proposals(
        proposals.proposals,
        history_end=12,
        window_width=2,
        historical_token_budget=8,
        beta=proposals.beta,
    )
    assert len(resolved.windows) == 1

    without_lower = policy.propose(
        SelectionContext(position=16, layer=1, history_end=12, hidden=torch.ones(8)),
        PolicyState(),
    )
    with_lower = policy.propose(
        SelectionContext(
            position=16,
            layer=1,
            history_end=12,
            hidden=torch.ones(8),
            lower_layer=proposals.soft_parameters,
        ),
        PolicyState(),
    )
    assert not torch.equal(without_lower.soft_parameters, with_lower.soft_parameters)
