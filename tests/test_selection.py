from __future__ import annotations

import pytest
import torch

from qkvs.selection import (
    FixedOffsetPolicy,
    HeatFieldProposal,
    HistoricalSelection,
    LearnedHeatFieldPolicy,
    PolicyState,
    SelectionContext,
    resolve_proposals,
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
    prior_centers = first.state.recurrent[:, 0]
    assert torch.linalg.vector_norm(persistent_second.state.recurrent[:, 0] - prior_centers) < (
        torch.linalg.vector_norm(responsive_second.state.recurrent[:, 0] - prior_centers)
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
