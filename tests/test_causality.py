from __future__ import annotations

import torch

from qkvs.model import FrozenCausalReader, TokenStream
from qkvs.selection import FixedOffsetPolicy, RollingOnlyPolicy


def _reader() -> FrozenCausalReader:
    return FrozenCausalReader(vocab_size=16, width=24, layers=2, heads=4, seed=17)


def _stream(tokens: tuple[int, ...]) -> TokenStream:
    supports = []
    for position, target in enumerate(tokens[1:]):
        prior = [index for index in range(position + 1) if tokens[index] == target]
        supports.append(prior[-1] if prior else None)
    return TokenStream("test", tokens, tuple(supports))


def test_reader_instruments_actual_causal_kv_accesses() -> None:
    reader = _reader()
    assert all(not parameter.requires_grad for parameter in reader.parameters())
    scores = reader.execute(
        _stream(tuple(index % 8 for index in range(20))),
        FixedOffsetPolicy((8, 12)),
        rolling_tokens=4,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
    )
    assert scores
    for score in scores:
        for access in score.accesses:
            assert access.rolling_indices
            assert access.rolling_indices[-1] == access.token_position
            assert len(access.historical_indices) <= 4
            assert all(index < access.history_end for index in access.historical_indices)
            assert access.actual_indices == (
                *access.historical_indices,
                *access.rolling_indices,
            )


def test_future_suffix_cannot_change_prior_scores_or_accesses() -> None:
    prefix = tuple(index % 8 for index in range(12))
    first = _stream((*prefix, 1, 2, 3, 4))
    second = _stream((*prefix, 9, 10, 11, 12))
    policy = FixedOffsetPolicy((6, 10))
    first_scores = _reader().execute(
        first,
        policy,
        rolling_tokens=4,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
    )
    second_scores = _reader().execute(
        second,
        FixedOffsetPolicy((6, 10)),
        rolling_tokens=4,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
    )
    assert [score.as_dict() for score in first_scores[:11]] == [
        score.as_dict() for score in second_scores[:11]
    ]


def test_same_reader_engine_accepts_rolling_and_bounded_policies() -> None:
    stream = _stream(tuple(index % 8 for index in range(16)))
    rolling = _reader().execute(
        stream,
        RollingOnlyPolicy(),
        rolling_tokens=4,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
    )
    selected = _reader().execute(
        stream,
        FixedOffsetPolicy((6, 10)),
        rolling_tokens=4,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
    )
    assert all(not access.historical_indices for score in rolling for access in score.accesses)
    assert any(access.historical_indices for score in selected for access in score.accesses)
