from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

import qkvs.model as model_module
from qkvs.model import FrozenCausalReader, HuggingFaceQwen2Reader, TokenStream
from qkvs.selection import (
    FixedOffsetPolicy,
    HeatFieldProposal,
    PolicyState,
    ProposalSet,
    RollingOnlyPolicy,
)


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
            assert access.cache_length == access.token_position + 1
            assert access.historical_occurrences == len(access.historical_indices)
            assert access.duplicate_historical_occurrences == (
                len(access.historical_indices) - len(set(access.historical_indices))
            )
            assert access.rolling_overlap_occurrences == 0
            assert access.gathered_key_occurrences == len(access.actual_indices)
            assert access.gathered_value_occurrences == len(access.actual_indices)


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
    assert any(
        rolling_score.loss != selected_score.loss
        for rolling_score, selected_score in zip(rolling, selected, strict=True)
    )


def _qwen_reader() -> HuggingFaceQwen2Reader:
    torch.manual_seed(29)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    model = Qwen2ForCausalLM(config)
    return HuggingFaceQwen2Reader(
        model,
        specification={
            "id": "test-only-random-qwen2",
            "revision": "0" * 40,
            "dtype": "float32",
            "frozen": True,
        },
    )


def test_direct_cache_gather_reads_real_occurrences_in_order() -> None:
    cache = [torch.full((2, 3), float(index)) for index in range(8)]
    gathered = model_module._gather_cache_occurrences(
        cache,
        (5, 2, 5, 7),
        occurrence_limit=4,
    )
    assert gathered.shape == (2, 4, 3)
    assert gathered[0, :, 0].tolist() == [5.0, 2.0, 5.0, 7.0]
    with pytest.raises(ValueError, match="bounded occurrence limit"):
        model_module._gather_cache_occurrences(
            cache,
            (0, 1, 2),
            occurrence_limit=2,
        )


@pytest.mark.parametrize("reader_factory", (_reader, _qwen_reader), ids=("synthetic", "qwen2"))
def test_bounded_execution_never_stacks_the_full_cache(monkeypatch, reader_factory) -> None:
    reader = reader_factory()
    original_stack = torch.stack
    stacked_widths: list[int] = []

    def guarded_stack(tensors, *args, **kwargs):
        dim = kwargs.get("dim", args[0] if args else 0)
        if dim == 1:
            stacked_widths.append(len(tensors))
            assert len(tensors) <= 5
        return original_stack(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "stack", guarded_stack)
    scores = reader.execute(
        _stream(tuple(index % 16 for index in range(48))),
        FixedOffsetPolicy((8, 16, 24)),
        rolling_tokens=2,
        historical_token_budget=3,
        window_width=1,
        device=torch.device("cpu"),
    )
    assert stacked_widths
    assert max(access.cache_length for score in scores for access in score.accesses) == 47
    assert max(stacked_widths) == 5


@pytest.mark.parametrize("reader_factory", (_reader, _qwen_reader), ids=("synthetic", "qwen2"))
def test_direct_gather_matches_legacy_materialized_scores(monkeypatch, reader_factory) -> None:
    stream = _stream(tuple(index % 16 for index in range(24)))
    direct = reader_factory().execute(
        stream,
        FixedOffsetPolicy((6, 10)),
        rolling_tokens=3,
        historical_token_budget=5,
        window_width=2,
        device=torch.device("cpu"),
    )

    def legacy_gather(cache, indices, *, occurrence_limit):
        assert occurrence_limit is None or len(indices) <= occurrence_limit
        return torch.stack(cache, dim=1)[:, indices]

    monkeypatch.setattr(model_module, "_gather_cache_occurrences", legacy_gather)
    materialized = reader_factory().execute(
        stream,
        FixedOffsetPolicy((6, 10)),
        rolling_tokens=3,
        historical_token_budget=5,
        window_width=2,
        device=torch.device("cpu"),
    )
    assert [score.as_dict() for score in direct] == [score.as_dict() for score in materialized]


@pytest.mark.parametrize(
    ("offsets", "rolling_tokens", "historical_budget", "window_width"),
    (
        ((5,), 4, 1, 1),
        ((5, 5), 2, 4, 2),
        ((5, 9, 13), 3, 5, 2),
    ),
)
def test_bounded_gathers_cover_early_overlap_and_independent_limits(
    offsets,
    rolling_tokens,
    historical_budget,
    window_width,
) -> None:
    scores = _reader().execute(
        _stream(tuple(index % 8 for index in range(24))),
        FixedOffsetPolicy(offsets),
        rolling_tokens=rolling_tokens,
        historical_token_budget=historical_budget,
        window_width=window_width,
        device=torch.device("cpu"),
    )
    assert all(access.actual_indices == (0,) for access in scores[0].accesses)
    assert all(
        access.gathered_key_occurrences
        == access.gathered_value_occurrences
        == len(access.actual_indices)
        <= rolling_tokens + historical_budget
        for score in scores
        for access in score.accesses
    )
    if len(set(offsets)) != len(offsets):
        assert any(
            access.duplicate_historical_occurrences for score in scores for access in score.accesses
        )
    if historical_budget % window_width:
        assert (
            max(access.historical_occurrences for score in scores for access in score.accesses)
            == historical_budget - historical_budget % window_width
        )


def test_qwen2_incremental_full_history_matches_standard_cpu_reference() -> None:
    diagnostic = _qwen_reader().full_history_diagnostic(
        (1, 5, 9, 3, 7, 11, 2, 13),
        device=torch.device("cpu"),
        atol=2e-5,
        rtol=2e-5,
    )
    assert diagnostic["diagnostic_only_unbounded"]
    assert diagnostic["allclose"], diagnostic
    assert diagnostic["max_abs_error"] < 2e-5
    assert len(diagnostic["per_position_max_abs_error"]) == 7


def test_qwen2_adapter_rejects_position_variant_outside_validated_support() -> None:
    model = _qwen_reader().model
    model.config.rope_scaling = {"rope_type": "linear", "factor": 2.0}
    with pytest.raises(ValueError, match="nondefault RoPE"):
        HuggingFaceQwen2Reader(
            model,
            specification={
                "id": "test-only-random-qwen2",
                "revision": "0" * 40,
                "dtype": "float32",
                "frozen": True,
            },
        )


def test_qwen2_actual_reads_are_causal_and_layer_specific() -> None:
    class LayerPolicy:
        def propose(self, context, state: PolicyState) -> ProposalSet:
            del state
            center = 0.1 if context.layer == 0 else 0.9
            return ProposalSet(
                (HeatFieldProposal(center=center, spread=1.0, weight=1.0),),
                PolicyState((center,)),
            )

    tokens = tuple(index % 32 for index in range(16))
    scores = _qwen_reader().execute(
        _stream(tokens),
        LayerPolicy(),
        rolling_tokens=2,
        historical_token_budget=1,
        window_width=1,
        device=torch.device("cpu"),
    )
    late = scores[-1].accesses
    assert late[0].historical_indices != late[1].historical_indices
    for score in scores:
        for access in score.accesses:
            assert access.cache_length == score.position + 1
            assert all(index < access.history_end for index in access.historical_indices)
            assert access.rolling_overlap_occurrences == 0


def test_qwen2_future_suffix_cannot_change_prior_scores_or_accesses() -> None:
    prefix = (1, 2, 3, 4, 5, 6, 7)
    first = _stream((*prefix, 8, 9, 10))
    second = _stream((*prefix, 21, 22, 23))
    first_scores = _qwen_reader().execute(
        first,
        FixedOffsetPolicy((4,)),
        rolling_tokens=2,
        historical_token_budget=1,
        window_width=1,
        device=torch.device("cpu"),
    )
    second_scores = _qwen_reader().execute(
        second,
        FixedOffsetPolicy((4,)),
        rolling_tokens=2,
        historical_token_budget=1,
        window_width=1,
        device=torch.device("cpu"),
    )
    assert [score.as_dict() for score in first_scores[:6]] == [
        score.as_dict() for score in second_scores[:6]
    ]


@pytest.mark.skipif(
    "QKVS_PRETRAINED_MODEL" not in os.environ,
    reason="set QKVS_PRETRAINED_MODEL to a complete local Qwen2 checkpoint",
)
def test_local_pretrained_qwen2_full_history_reference() -> None:
    checkpoint = os.environ["QKVS_PRETRAINED_MODEL"]
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    reader = HuggingFaceQwen2Reader(
        model,
        tokenizer,
        specification={
            "id": checkpoint,
            "revision": "0" * 40,
            "dtype": "float32",
            "frozen": True,
        },
    )
    tokens = tuple(tokenizer.encode("A short deterministic reference.", add_special_tokens=False))
    diagnostic = reader.full_history_diagnostic(
        tokens[: min(len(tokens), 12)],
        device=torch.device("cpu"),
        atol=2e-4,
        rtol=2e-4,
    )
    assert diagnostic["allclose"], diagnostic
