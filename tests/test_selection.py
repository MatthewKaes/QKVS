from __future__ import annotations

import pytest

from qkvs.selection import (
    FixedOffsetPolicy,
    HeatFieldProposal,
    HistoricalSelection,
    PolicyState,
    SelectionContext,
    resolve_proposals,
    validate_historical_selection,
)


def test_collective_resolution_counts_overlapping_occurrences() -> None:
    proposals = (
        HeatFieldProposal(center=0.5, spread=0.2, weight=1.0),
        HeatFieldProposal(center=0.5, spread=0.3, weight=0.5),
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


def test_gamma_recurrence_is_bounded_and_beta_is_spatial_only() -> None:
    policy = FixedOffsetPolicy((8,), gamma=0.75, beta=2.0)
    first = policy.propose(
        SelectionContext(position=20, layer=0, history_end=17, hidden=object()),
        PolicyState(),
    )
    second = policy.propose(
        SelectionContext(position=21, layer=0, history_end=18, hidden=object()),
        first.state,
    )
    assert len(second.state.centers) == 1
    assert second.proposals[0].spread == 0.5
    assert second.state.centers[0] != first.state.centers[0]
    assert 0.0 <= second.state.centers[0] <= 1.0
