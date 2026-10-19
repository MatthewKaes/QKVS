from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

import qkvs.model as model_module
from qkvs.model import FrozenCausalReader, HuggingFaceQwen2Reader, TokenStream
from qkvs.selection import (
    FixedOffsetPolicy,
    FullQKVPolicy,
    HeatFieldProposal,
    LearnedHeatFieldPolicy,
    PolicyState,
    ProposalSet,
    RollingOnlyPolicy,
    SelectionContext,
    resolve_proposals,
)

_PINNED_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_PINNED_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_PINNED_FILE_SHA256 = {
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
}


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
        activation_tokens=4,
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
        activation_tokens=4,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
    )
    second_scores = _reader().execute(
        second,
        FixedOffsetPolicy((6, 10)),
        rolling_tokens=4,
        activation_tokens=4,
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
        activation_tokens=4,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
    )
    selected = _reader().execute(
        stream,
        FixedOffsetPolicy((6, 10)),
        rolling_tokens=4,
        activation_tokens=4,
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


def _qwen_reader(
    dtype: torch.dtype = torch.float32,
    *,
    layers: int = 2,
) -> HuggingFaceQwen2Reader:
    torch.manual_seed(29)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    model = Qwen2ForCausalLM(config).to(dtype=dtype)
    return HuggingFaceQwen2Reader(
        model,
        specification={
            "id": "test-only-random-qwen2",
            "revision": "0" * 40,
            "dtype": str(dtype).removeprefix("torch."),
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


def test_unindexed_cuda_request_accepts_the_concrete_default_device() -> None:
    assert model_module._device_matches_request(
        torch.device("cuda:0"),
        torch.device("cuda"),
    )
    assert model_module._device_matches_request(
        torch.device("cuda:1"),
        torch.device("cuda:1"),
    )
    assert not model_module._device_matches_request(
        torch.device("cuda:1"),
        torch.device("cuda:0"),
    )
    assert not model_module._device_matches_request(
        torch.device("cpu"),
        torch.device("cuda"),
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_dense_snapshot_probe_is_noncommitting_causal_and_full_selection_matches(dtype) -> None:
    reader = _qwen_reader(dtype)
    first = _stream((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    second = _stream((1, 2, 3, 4, 5, 6, 7, 8, 31, 30))
    snapshot = reader.dense_causal_snapshot(
        first,
        position=6,
        device=torch.device("cpu"),
        ownership="snapshot-test",
        prefill_chunk_tokens=2,
    )
    later_snapshot = reader.dense_causal_snapshot(
        first,
        position=8,
        device=torch.device("cpu"),
        ownership="snapshot-test",
        prefill_chunk_tokens=2,
    )
    assert (
        reader.place_dense_causal_snapshot(later_snapshot, device=torch.device("cpu"))
        is later_snapshot
    )
    derived_snapshot = reader.derive_dense_causal_snapshot(
        later_snapshot,
        first,
        position=6,
        device=torch.device("cpu"),
        ownership="snapshot-test",
        prefill_chunk_tokens=2,
    )
    assert derived_snapshot.identity == snapshot.identity
    assert derived_snapshot.prefix_tokens == snapshot.prefix_tokens
    assert all(
        torch.equal(derived, direct)
        for derived, direct in zip(
            derived_snapshot.key_cache,
            snapshot.key_cache,
            strict=True,
        )
    )
    assert all(
        torch.equal(derived, direct)
        for derived, direct in zip(
            derived_snapshot.value_cache,
            snapshot.value_cache,
            strict=True,
        )
    )
    assert all(
        derived.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        for derived, source in zip(
            derived_snapshot.key_cache,
            later_snapshot.key_cache,
            strict=True,
        )
    )
    assert all(
        derived.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        for derived, source in zip(
            derived_snapshot.value_cache,
            later_snapshot.value_cache,
            strict=True,
        )
    )
    input_calls = {"embedding": 0, "rope": 0}
    embedding_hook = reader.model.model.embed_tokens.register_forward_hook(
        lambda _module, _inputs, _output: input_calls.__setitem__(
            "embedding", input_calls["embedding"] + 1
        )
    )
    rope_hook = reader.model.model.rotary_emb.register_forward_hook(
        lambda _module, _inputs, _output: input_calls.__setitem__("rope", input_calls["rope"] + 1)
    )
    state = reader.prepare_snapshot_probe(
        snapshot,
        first,
        position=6,
        device=torch.device("cpu"),
        ownership="snapshot-test",
        prefill_chunk_tokens=2,
    )
    keys_before = tuple(cache.clone() for cache in state.key_cache)
    values_before = tuple(cache.clone() for cache in state.value_cache)
    reader.model.to = Mock(side_effect=AssertionError("snapshot probes must not remap the model"))

    reference = reader.snapshot_probe(
        first,
        state,
        position=6,
        selected_positions=None,
        device=torch.device("cpu"),
    )
    full_selection = tuple(tuple(range(7)) for _ in reader.model.model.layers)
    dense_candidate = reader.snapshot_probe(
        first,
        state,
        position=6,
        selected_positions=full_selection,
        device=torch.device("cpu"),
    )
    compact_ranges = (((0, 2), (4, 7)), ((1, 5),))
    prepared_selection = reader.prepare_snapshot_selection(
        compact_ranges,
        position=6,
        device=torch.device("cpu"),
    )
    assert prepared_selection.base_occurrence_counts == (4, 4)
    compact_candidate = reader.snapshot_probe(
        first,
        state,
        position=6,
        selected_positions=prepared_selection,
        device=torch.device("cpu"),
    )
    explicit_candidate = reader.snapshot_probe(
        first,
        state,
        position=6,
        selected_positions=((0, 1, 4, 5, 6), (1, 2, 3, 4)),
        device=torch.device("cpu"),
    )
    alternate_ranges = (((0, 3),), ((2, 6),))
    alternate_selection = reader.prepare_snapshot_selection(
        alternate_ranges,
        position=6,
        device=torch.device("cpu"),
    )
    alternate_candidate = reader.snapshot_probe(
        first,
        state,
        position=6,
        selected_positions=alternate_selection,
        device=torch.device("cpu"),
    )
    batch_selections = reader.prepare_snapshot_selection_batch(
        (compact_ranges, alternate_ranges),
        position=6,
        device=torch.device("cpu"),
    )
    batch_candidates = reader.snapshot_probe_batch(
        first,
        state,
        position=6,
        selected_positions=batch_selections,
        device=torch.device("cpu"),
    )
    sparse_selection = ((0, 1), (2, 3))
    sparse = reader.snapshot_probe(
        first,
        state,
        position=6,
        selected_positions=sparse_selection,
        device=torch.device("cpu"),
    )
    suffix_changed = reader.snapshot_probe(
        second,
        state,
        position=6,
        selected_positions=sparse_selection,
        device=torch.device("cpu"),
    )
    current_changed = _stream((1, 2, 3, 4, 5, 6, 31, 8, 9, 10))
    with pytest.raises(ValueError, match="requested token and target"):
        reader.snapshot_probe(
            current_changed,
            state,
            position=6,
            selected_positions=sparse_selection,
            device=torch.device("cpu"),
        )

    assert torch.equal(reference.logits, dense_candidate.logits)
    assert torch.equal(reference.loss, dense_candidate.loss)
    reader.model.to.assert_not_called()
    assert torch.equal(compact_candidate.logits, explicit_candidate.logits)
    assert torch.equal(compact_candidate.loss, explicit_candidate.loss)
    for scalar, batched in zip(
        (compact_candidate, alternate_candidate),
        batch_candidates,
        strict=True,
    ):
        torch.testing.assert_close(batched.logits, scalar.logits, rtol=0.0, atol=1e-6)
        torch.testing.assert_close(batched.loss, scalar.loss, rtol=0.0, atol=1e-6)
        assert torch.equal(batched.prediction, scalar.prediction)
        assert batched.gathered_keys_per_layer == scalar.gathered_keys_per_layer
        assert batched.gathered_values_per_layer == scalar.gathered_values_per_layer
    assert compact_candidate.gathered_keys_per_layer == (5, 4)
    assert compact_candidate.gathered_values_per_layer == (5, 4)
    assert dense_candidate.gathered_keys_per_layer == (7, 7)
    assert dense_candidate.gathered_values_per_layer == (7, 7)
    assert sparse.selected_positions_per_layer == (2, 2)
    assert sparse.gathered_keys_per_layer == sparse.gathered_values_per_layer == (2, 2)
    assert torch.equal(sparse.logits, suffix_changed.logits)
    assert torch.equal(sparse.loss, suffix_changed.loss)
    assert input_calls == {"embedding": 1, "rope": 1}
    embedding_hook.remove()
    rope_hook.remove()
    assert all(
        torch.equal(before, after)
        for before, after in zip(keys_before, state.key_cache, strict=True)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(values_before, state.value_cache, strict=True)
    )


def test_snapshot_probe_batch_matches_full_depth_bfloat16_without_repeating_gqa_cache() -> None:
    reader = _qwen_reader(torch.bfloat16, layers=24)
    stream = _stream((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    snapshot = reader.dense_causal_snapshot(
        stream,
        position=6,
        device=torch.device("cpu"),
        ownership="snapshot-batch-test",
        prefill_chunk_tokens=2,
    )
    state = reader.prepare_snapshot_probe(
        snapshot,
        stream,
        position=6,
        device=torch.device("cpu"),
        ownership="snapshot-batch-test",
        prefill_chunk_tokens=2,
    )
    range_batches = tuple(
        tuple(
            (
                ((0, 2), (4, 7))
                if (candidate + layer) % 3 == 0
                else ((0, 3),)
                if (candidate + layer) % 3 == 1
                else ((1, 5),)
            )
            for layer in range(24)
        )
        for candidate in range(4)
    )
    scalar_selections = tuple(
        reader.prepare_snapshot_selection(
            ranges,
            position=6,
            device=torch.device("cpu"),
        )
        for ranges in range_batches
    )
    scalar_results = tuple(
        reader.snapshot_probe(
            stream,
            state,
            position=6,
            selected_positions=selection,
            device=torch.device("cpu"),
        )
        for selection in scalar_selections
    )
    batch_selections = reader.prepare_snapshot_selection_batch(
        range_batches,
        position=6,
        device=torch.device("cpu"),
    )
    reader._repeat_kv = Mock(side_effect=AssertionError("batched GQA must not repeat K/V heads"))
    batch_results = reader.snapshot_probe_batch(
        stream,
        state,
        position=6,
        selected_positions=batch_selections,
        device=torch.device("cpu"),
    )

    reader._repeat_kv.assert_not_called()
    for scalar, batched in zip(scalar_results, batch_results, strict=True):
        assert torch.equal(batched.logits, scalar.logits)
        assert torch.equal(batched.loss, scalar.loss)
        assert torch.equal(batched.prediction, scalar.prediction)
        assert batched.gathered_keys_per_layer == scalar.gathered_keys_per_layer
        assert batched.gathered_values_per_layer == scalar.gathered_values_per_layer


@pytest.mark.parametrize("reader_factory", (_reader, _qwen_reader), ids=("synthetic", "qwen2"))
def test_bounded_execution_never_stacks_the_full_cache(monkeypatch, reader_factory) -> None:
    reader = reader_factory()
    original_gather = model_module._gather_cache_occurrences
    gathered_widths: list[int] = []

    def guarded_gather(cache, indices, *, occurrence_limit, base_occurrence_count=None):
        gathered = original_gather(
            cache,
            indices,
            occurrence_limit=occurrence_limit,
            base_occurrence_count=base_occurrence_count,
        )
        if occurrence_limit is not None:
            gathered_widths.append(gathered.shape[1])
            assert gathered.shape[1] <= 5
        return gathered

    monkeypatch.setattr(model_module, "_gather_cache_occurrences", guarded_gather)
    scores = reader.execute(
        _stream(tuple(index % 16 for index in range(48))),
        FixedOffsetPolicy((8, 16, 24)),
        rolling_tokens=2,
        activation_tokens=2,
        historical_token_budget=3,
        window_width=1,
        device=torch.device("cpu"),
    )
    assert gathered_widths
    assert max(access.cache_length for score in scores for access in score.accesses) == 47
    assert max(gathered_widths) == 5


@pytest.mark.parametrize("reader_factory", (_reader, _qwen_reader), ids=("synthetic", "qwen2"))
def test_direct_gather_matches_legacy_materialized_scores(monkeypatch, reader_factory) -> None:
    stream = _stream(tuple(index % 16 for index in range(24)))
    direct = reader_factory().execute(
        stream,
        FixedOffsetPolicy((6, 10)),
        rolling_tokens=3,
        activation_tokens=3,
        historical_token_budget=5,
        window_width=2,
        device=torch.device("cpu"),
    )

    def legacy_gather(cache, indices, *, occurrence_limit, base_occurrence_count=None):
        del base_occurrence_count
        assert occurrence_limit is None or len(indices) <= occurrence_limit
        full = (
            cache.storage[:, : len(cache)]
            if isinstance(cache, model_module._LayerKVCache)
            else torch.stack(cache, dim=1)
        )
        return full[:, indices]

    monkeypatch.setattr(model_module, "_gather_cache_occurrences", legacy_gather)
    materialized = reader_factory().execute(
        stream,
        FixedOffsetPolicy((6, 10)),
        rolling_tokens=3,
        activation_tokens=3,
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
        activation_tokens=rolling_tokens,
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
    assert diagnostic["loss_allclose"], diagnostic
    assert diagnostic["max_abs_error"] < 2e-5
    assert diagnostic["max_abs_loss_error"] < 2e-5
    assert len(diagnostic["per_position_max_abs_error"]) == 7
    assert len(diagnostic["per_position_abs_loss_error"]) == 7


def test_dense_bootstrap_parity_and_activation_boundary() -> None:
    class RecordingPolicy:
        def __init__(self) -> None:
            self.calls = []

        def propose(self, context, state: PolicyState) -> ProposalSet:
            self.calls.append(
                (
                    context.position,
                    context.layer,
                    context.history_end,
                    state.centers,
                    state.recurrent,
                )
            )
            return ProposalSet((), PolicyState((float(context.position),)))

    reader = _qwen_reader()
    stream = _stream(tuple(index % 32 for index in range(12)))
    device = torch.device("cpu")
    snapshot = reader.dense_prefill(
        stream,
        rolling_tokens=4,
        activation_tokens=6,
        device=device,
        ownership="evaluation",
    )
    with torch.inference_mode():
        reference = reader.model(
            input_ids=torch.tensor([stream.tokens[:6]]),
            use_cache=False,
            logits_to_keep=0,
        ).logits[0]
    targets = torch.tensor(stream.tokens[1:6])
    reference_losses = torch.nn.functional.cross_entropy(reference[:-1], targets, reduction="none")
    assert torch.allclose(
        torch.tensor(snapshot.dense_losses), reference_losses, atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(snapshot.boundary_logits, reference[-1], atol=2e-5, rtol=2e-5)

    policy = RecordingPolicy()
    scores = reader.execute(
        stream,
        policy,
        rolling_tokens=4,
        activation_tokens=6,
        historical_token_budget=2,
        window_width=1,
        device=device,
        dense_snapshot=snapshot,
        snapshot_ownership="evaluation",
    )
    assert [score.phase for score in scores[:6]] == [
        "dense_bootstrap",
        "dense_bootstrap",
        "dense_bootstrap",
        "dense_bootstrap",
        "dense_bridge",
        "dense_bridge",
    ]
    assert all(score.phase == "post_transition" for score in scores[6:])
    assert {(position, layer) for position, layer, *_ in policy.calls} == {
        (position, layer) for position in range(6, 11) for layer in range(2)
    }
    first_calls = [call for call in policy.calls if call[0] == 6]
    assert all(history_end == 3 for _, _, history_end, _, _ in first_calls)
    assert all(centers == () and recurrent is None for _, _, _, centers, recurrent in first_calls)
    assert all(
        access.history_end == max(0, score.position - 4 + 1) and len(access.actual_indices) <= 6
        for score in scores[6:]
        for access in score.accesses
    )


def test_dense_prefill_persistence_and_cached_continuation_equivalence(tmp_path) -> None:
    reader = _qwen_reader()
    stream = _stream(tuple(index % 32 for index in range(14)))
    device = torch.device("cpu")
    snapshot = reader.dense_prefill(
        stream,
        rolling_tokens=4,
        activation_tokens=6,
        device=device,
        ownership="evaluation",
    )
    numerical = snapshot.identity["numerical"]
    assert "float32_matmul_precision" not in numerical
    assert numerical["fp32_precision"] == {
        "global": torch.backends.fp32_precision,
        "cuda_matmul": torch.backends.cuda.matmul.fp32_precision,
        "cudnn": torch.backends.cudnn.fp32_precision,
        "cudnn_conv": torch.backends.cudnn.conv.fp32_precision,
    }
    cache_before = tuple(layer_cache.clone() for layer_cache in snapshot.key_cache)
    path = tmp_path / "dense.pt"
    reader.save_dense_prefill(snapshot, path)
    restored = reader.load_dense_prefill(
        path,
        stream,
        rolling_tokens=4,
        activation_tokens=6,
        device=device,
        ownership="evaluation",
    )
    cached = reader.execute(
        stream,
        FixedOffsetPolicy((6, 9)),
        rolling_tokens=4,
        activation_tokens=6,
        historical_token_budget=2,
        window_width=1,
        device=device,
        dense_snapshot=restored,
        snapshot_ownership="evaluation",
    )
    uncached = reader.execute(
        stream,
        FixedOffsetPolicy((6, 9)),
        rolling_tokens=4,
        activation_tokens=6,
        historical_token_budget=2,
        window_width=1,
        device=device,
        snapshot_ownership="evaluation",
    )
    assert [score.as_dict() for score in cached] == [score.as_dict() for score in uncached]
    branch = _stream((*stream.tokens[:6], 31, 30, 29, 28))
    shared = reader.load_dense_prefill(
        path,
        branch,
        rolling_tokens=4,
        activation_tokens=6,
        device=device,
        ownership="evaluation",
    )
    branch_scores = reader.execute(
        branch,
        FixedOffsetPolicy((6, 9)),
        rolling_tokens=4,
        activation_tokens=6,
        historical_token_budget=2,
        window_width=1,
        device=device,
        dense_snapshot=shared,
        snapshot_ownership="evaluation",
    )
    assert [score.as_dict() for score in branch_scores[:5]] == [
        score.as_dict() for score in cached[:5]
    ]
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            cache_before,
            snapshot.key_cache,
            strict=True,
        )
    )
    changed_prefix = _stream((63, *stream.tokens[1:]))
    with pytest.raises(ValueError, match="identity does not match"):
        reader.load_dense_prefill(
            path,
            changed_prefix,
            rolling_tokens=4,
            activation_tokens=6,
            device=device,
            ownership="evaluation",
        )
    with pytest.raises(ValueError, match="identity does not match"):
        reader.load_dense_prefill(
            path,
            stream,
            rolling_tokens=4,
            activation_tokens=6,
            device=device,
            ownership="train",
        )


def test_full_qkv_continuation_reuses_prefill_and_matches_dense_reference() -> None:
    reader = _qwen_reader()
    stream = _stream(tuple(index % 32 for index in range(14)))
    device = torch.device("cpu")
    snapshot = reader.dense_prefill(
        stream,
        rolling_tokens=4,
        activation_tokens=6,
        device=device,
        ownership="evaluation",
        prefill_chunk_tokens=3,
    )
    scores = reader.execute(
        stream,
        FullQKVPolicy(),
        rolling_tokens=4,
        activation_tokens=6,
        historical_token_budget=2,
        window_width=1,
        device=device,
        dense_snapshot=snapshot,
        snapshot_ownership="evaluation",
        prefill_chunk_tokens=3,
    )
    with torch.inference_mode():
        logits = reader.model(
            input_ids=torch.tensor([stream.tokens]),
            use_cache=False,
            logits_to_keep=0,
        ).logits[0, :-1]
    losses = torch.nn.functional.cross_entropy(
        logits,
        torch.tensor(stream.tokens[1:]),
        reduction="none",
    )
    assert torch.allclose(
        torch.tensor([score.loss for score in scores]), losses, atol=2e-5, rtol=2e-5
    )
    assert all(
        access.attention_mode == "full_qkv"
        and access.gathered_key_occurrences == score.position + 1
        and access.gathered_value_occurrences == score.position + 1
        for score in scores[6:]
        for access in score.accesses
    )


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


def test_huggingface_reader_uses_current_dtype_keyword(monkeypatch) -> None:
    tokenizer = object()
    model = Mock()
    model.to.return_value = model
    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    sentinel = object()
    reader_factory = Mock(return_value=sentinel)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", model_loader)
    monkeypatch.setattr(model_module, "HuggingFaceQwen2Reader", reader_factory)
    specification = {
        "backend": "huggingface",
        "id": "test/qwen2",
        "revision": "1" * 40,
        "dtype": "bfloat16",
        "frozen": True,
    }

    loaded = model_module.load_reader(specification, torch.device("cpu"))

    assert loaded is sentinel
    model_loader.assert_called_once_with(
        "test/qwen2",
        revision="1" * 40,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    reader_factory.assert_called_once_with(model, tokenizer, specification=specification)


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
        activation_tokens=2,
        historical_token_budget=1,
        window_width=1,
        device=torch.device("cpu"),
        proposal_trace_interval=3,
    )
    late = scores[-1].accesses
    assert late[0].historical_indices != late[1].historical_indices
    for score in scores:
        for access in score.accesses:
            assert access.cache_length == score.position + 1
            assert all(index < access.history_end for index in access.historical_indices)
            assert access.rolling_overlap_occurrences == 0
            if score.phase == "post_transition":
                assert bool(access.proposal_weights) == (score.position % 3 == 0)


def test_layer_invariant_reader_reuses_one_index_tensor_per_token(monkeypatch) -> None:
    class CountingFixedOffsetPolicy(FixedOffsetPolicy):
        def __init__(self, offsets):
            super().__init__(offsets)
            self.calls = []

        def propose(self, context, state):
            self.calls.append((context.position, context.layer))
            return super().propose(context, state)

    class LayerwiseFixedOffsetPolicy:
        def __init__(self, offsets):
            self.delegate = FixedOffsetPolicy(offsets)
            self.calls = 0

        def propose(self, context, state):
            self.calls += 1
            return self.delegate.propose(context, state)

    reader = _qwen_reader()
    stream = _stream(tuple(index % 32 for index in range(16)))
    policy = CountingFixedOffsetPolicy((6, 9))
    gathered_indices = []
    original_gather = model_module._gather_cache_occurrences

    def counted_gather(cache, indices, *, occurrence_limit, base_occurrence_count=None):
        if isinstance(cache, model_module._LayerKVCache) and indices is not None:
            gathered_indices.append(indices)
        return original_gather(
            cache,
            indices,
            occurrence_limit=occurrence_limit,
            base_occurrence_count=base_occurrence_count,
        )

    monkeypatch.setattr(model_module, "_gather_cache_occurrences", counted_gather)
    shared = reader.execute(
        stream,
        policy,
        rolling_tokens=4,
        activation_tokens=6,
        historical_token_budget=2,
        window_width=1,
        device=torch.device("cpu"),
    )
    post_transition_tokens = len(stream.tokens) - 1 - 6
    layer_count = reader.model.config.num_hidden_layers
    assert policy.calls == [(position, 0) for position in range(6, len(stream.tokens) - 1)]
    assert len(gathered_indices) == post_transition_tokens * layer_count * 2
    calls_per_token = layer_count * 2
    for start in range(0, len(gathered_indices), calls_per_token):
        token_indices = gathered_indices[start : start + calls_per_token]
        assert all(indices is token_indices[0] for indices in token_indices)

    layerwise_policy = LayerwiseFixedOffsetPolicy((6, 9))
    layerwise = reader.execute(
        stream,
        layerwise_policy,
        rolling_tokens=4,
        activation_tokens=6,
        historical_token_budget=2,
        window_width=1,
        device=torch.device("cpu"),
    )
    assert layerwise_policy.calls == post_transition_tokens * layer_count
    assert [score.as_dict() for score in shared] == [score.as_dict() for score in layerwise]


def test_qwen2_future_suffix_cannot_change_prior_scores_or_accesses() -> None:
    prefix = (1, 2, 3, 4, 5, 6, 7)
    first = _stream((*prefix, 8, 9, 10))
    second = _stream((*prefix, 21, 22, 23))
    first_scores = _qwen_reader().execute(
        first,
        FixedOffsetPolicy((4,)),
        rolling_tokens=2,
        activation_tokens=2,
        historical_token_budget=1,
        window_width=1,
        device=torch.device("cpu"),
    )
    second_scores = _qwen_reader().execute(
        second,
        FixedOffsetPolicy((4,)),
        rolling_tokens=2,
        activation_tokens=2,
        historical_token_budget=1,
        window_width=1,
        device=torch.device("cpu"),
    )
    assert [score.as_dict() for score in first_scores[:6]] == [
        score.as_dict() for score in second_scores[:6]
    ]


def _heatfield_policy(*, layers: int = 2) -> LearnedHeatFieldPolicy:
    torch.manual_seed(53)
    return LearnedHeatFieldPolicy(
        hidden_size=32,
        layers=layers,
        proposal_count=2,
        latent_size=16,
        gamma=0.5,
        beta=1.25,
        minimum_spread=0.5,
        maximum_spread=2.0,
        gate_threshold=0.4,
    )


def test_absolute_teacher_coverage_preserves_mass_and_gate_gradients() -> None:
    policy = _heatfield_policy()
    context = SelectionContext(
        position=10,
        layer=0,
        history_end=8,
        hidden=torch.linspace(-1.0, 1.0, 32),
    )

    def coverage_gradient(teacher_mass_per_position):
        policy.zero_grad(set_to_none=True)
        proposal_set = policy.propose(context, PolicyState())
        occupancy = model_module._collective_soft_occupancy(
            proposal_set.soft_parameters,
            history_end=context.history_end,
            window_width=2,
            beta=proposal_set.beta,
        )
        teacher = torch.full((context.history_end,), teacher_mass_per_position)
        loss, covered = model_module._absolute_teacher_coverage(teacher, occupancy)
        loss.backward()
        gate_gradient = policy.decoders[0][-1].bias.grad.view(2, 4)[:, 3].abs().sum()
        return loss.detach(), covered.detach(), gate_gradient.detach()

    negligible_loss, negligible_covered, negligible_gradient = coverage_gradient(1e-9)
    substantial_loss, substantial_covered, substantial_gradient = coverage_gradient(0.1)

    assert negligible_loss + negligible_covered == pytest.approx(8e-9)
    assert substantial_loss + substantial_covered == pytest.approx(0.8)
    assert substantial_loss > negligible_loss * 1e6
    assert substantial_gradient > negligible_gradient * 1e6
    assert substantial_gradient > 0.0


def test_hard_and_soft_multiplicity_preserve_occurrences_and_gradients() -> None:
    hard = model_module._hard_historical_multiplicity(
        (1, 2, 1, 2, 2),
        history_end=6,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert hard.tolist() == [0.0, 2.0, 3.0, 0.0, 0.0, 0.0]

    soft = torch.tensor(
        ((1.2, 1.1, 0.8, 0.7), (4.5, 1.6, 1.2, 0.6)),
        requires_grad=True,
    )
    multiplicity = model_module._soft_historical_multiplicity(
        soft,
        history_end=6,
        window_width=2,
        historical_token_budget=4,
        beta=1.25,
    )
    assert multiplicity.sum() <= 4.0 + 1e-6
    (multiplicity * torch.arange(6, dtype=torch.float32).square()).sum().backward()
    assert soft.grad is not None
    assert all(torch.count_nonzero(soft.grad[:, field]) for field in (0, 1, 2, 3))


def _configured_heatfield_policy(*, gate_raw: float, layers: int = 2) -> LearnedHeatFieldPolicy:
    policy = _heatfield_policy(layers=layers)
    with torch.no_grad():
        output = policy.decoders[0][-1]
        output.weight.zero_()
        output.bias.copy_(
            torch.tensor(
                (
                    2.0,
                    4.0,
                    2.0,
                    gate_raw,
                    0.0,
                    4.0,
                    1.0,
                    gate_raw,
                )
            )
        )
        for decoder in policy.decoders[1:]:
            decoder[-1].bias.view(2, 4)[:, 3].fill_(gate_raw)
    return policy


def _hard_aligned_objective(
    *,
    gate_raw: float,
    historical_token_budget: int,
    dtype: torch.dtype = torch.float32,
    layers: int = 2,
    predictive_continuation: str = "rolling",
) -> tuple[torch.Tensor, dict[str, object], LearnedHeatFieldPolicy]:
    reader = _qwen_reader(dtype, layers=layers)
    policy = _configured_heatfield_policy(gate_raw=gate_raw, layers=layers)
    stream = _stream(tuple(index % 32 for index in range(10)))
    snapshot = reader.dense_prefill(
        stream,
        rolling_tokens=2,
        activation_tokens=6,
        device=torch.device("cpu"),
        ownership="train",
    )
    objective, metrics = reader.heatfield_training_objective(
        stream,
        policy,
        position=6,
        layer_index=0,
        rolling_tokens=2,
        historical_token_budget=historical_token_budget,
        window_width=2,
        coverage_weight=0.0,
        utility_weight=1.0,
        device=torch.device("cpu"),
        activation_tokens=6,
        dense_snapshot=snapshot,
        snapshot_ownership="train",
        utility_gradient_estimator="hard_multiplicity_ste",
        predictive_continuation=predictive_continuation,
    )
    return objective, metrics, policy


@pytest.mark.parametrize(
    ("dtype", "loss_atol", "layers"),
    ((torch.float32, 1e-5, 2), (torch.bfloat16, 5e-3, 24)),
)
def test_hard_multiplicity_estimator_matches_duplicate_hard_reads_and_has_gradients(
    dtype: torch.dtype,
    loss_atol: float,
    layers: int,
) -> None:
    predictive_continuation = "qkvs" if layers == 24 else "rolling"
    objective, metrics, policy = _hard_aligned_objective(
        gate_raw=4.0,
        historical_token_budget=4,
        dtype=dtype,
        layers=layers,
        predictive_continuation=predictive_continuation,
    )

    assert metrics["utility_gradient_estimator"] == "hard_multiplicity_ste"
    assert metrics["predictive_continuation"] == predictive_continuation
    assert metrics["predictive_utility_scope"] == "target_layer_marginal"
    assert not metrics["full_trajectory_utility_measured"]
    assert metrics["historical_windows"] == ((1, 3), (0, 2))
    assert metrics["historical_occurrences"] == 4
    assert metrics["historical_unique_tokens"] == 3
    assert metrics["overlap_occurrences"] == 1
    assert metrics["hard_forward_loss_atol"] == loss_atol
    assert metrics["hard_forward_logit_atol"] == loss_atol
    assert metrics["utility_forward_logit_max_abs_error"] == 0.0
    assert metrics["utility_forward_loss_abs_error"] == 0.0
    assert metrics["soft_next_token_loss"] == pytest.approx(
        metrics["hard_selected_next_token_loss"],
        abs=loss_atol,
    )

    objective.backward()
    gradient = policy.decoders[0][-1].bias.grad.view(2, 4)
    assert torch.isfinite(gradient).all()
    assert all(torch.count_nonzero(gradient[:, field]) for field in (0, 1, 2, 3))

    _, budgeted, _ = _hard_aligned_objective(
        gate_raw=4.0,
        historical_token_budget=2,
        dtype=dtype,
        layers=layers,
        predictive_continuation=predictive_continuation,
    )
    assert budgeted["historical_windows"] == ((1, 3),)
    assert budgeted["historical_occurrences"] == 2
    assert budgeted["hard_forward_loss_atol"] == loss_atol
    assert budgeted["utility_forward_logit_max_abs_error"] == 0.0
    assert budgeted["utility_forward_loss_abs_error"] == 0.0


def test_hard_multiplicity_estimator_matches_rolling_when_all_gates_are_inactive() -> None:
    objective, metrics, _ = _hard_aligned_objective(
        gate_raw=-4.0,
        historical_token_budget=4,
        dtype=torch.bfloat16,
        layers=24,
        predictive_continuation="qkvs",
    )

    assert torch.isfinite(objective)
    assert metrics["historical_windows"] == ()
    assert metrics["historical_occurrences"] == 0
    assert metrics["hard_selected_next_token_loss"] == pytest.approx(
        metrics["rolling_next_token_loss"],
        abs=1e-7,
    )
    assert metrics["hard_forward_loss_atol"] == 5e-3
    assert metrics["utility_forward_logit_max_abs_error"] == 0.0
    assert metrics["utility_forward_loss_abs_error"] == 0.0


def test_full_depth_bfloat16_training_matches_normal_current_token_qkvs(monkeypatch) -> None:
    reader = _qwen_reader(torch.bfloat16, layers=24)
    policy = _configured_heatfield_policy(gate_raw=4.0, layers=24)
    reference_policy = _configured_heatfield_policy(gate_raw=4.0, layers=24)
    reference_policy.load_state_dict(policy.state_dict())
    stream = _stream(tuple(index % 32 for index in range(10)))
    snapshot = reader.dense_prefill(
        stream,
        rolling_tokens=2,
        activation_tokens=6,
        device=torch.device("cpu"),
        ownership="train",
    )

    normal_logits = []
    hook = reader.model.lm_head.register_forward_hook(
        lambda _module, _inputs, output: normal_logits.append(output[0, 0].detach().clone())
    )
    scores = reader.execute(
        stream,
        reference_policy,
        rolling_tokens=2,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
        activation_tokens=6,
        dense_snapshot=snapshot,
        snapshot_ownership="train",
    )
    hook.remove()

    proposal_contexts = []
    original_propose = policy.propose

    def tracked_propose(context, state):
        proposal_contexts.append(context)
        return original_propose(context, state)

    monkeypatch.setattr(policy, "propose", tracked_propose)
    downstream_indices = []
    original_gather = model_module._gather_cache_occurrences

    def tracked_gather(cache, indices, *, occurrence_limit, base_occurrence_count=None):
        gathered = original_gather(
            cache,
            indices,
            occurrence_limit=occurrence_limit,
            base_occurrence_count=base_occurrence_count,
        )
        if (
            isinstance(cache, model_module._LayerKVCache)
            and cache.overlay
            and isinstance(indices, torch.Tensor)
        ):
            downstream_indices.append(indices.detach().cpu().tolist())
        return gathered

    monkeypatch.setattr(model_module, "_gather_cache_occurrences", tracked_gather)
    training_logits = []
    hook = reader.model.lm_head.register_forward_hook(
        lambda _module, _inputs, output: training_logits.append(output[0, 0].detach().clone())
    )
    objective, metrics = reader.heatfield_training_objective(
        stream,
        policy,
        position=6,
        layer_index=0,
        rolling_tokens=2,
        historical_token_budget=4,
        window_width=2,
        coverage_weight=0.0,
        utility_weight=1.0,
        device=torch.device("cpu"),
        activation_tokens=6,
        dense_snapshot=snapshot,
        snapshot_ownership="train",
        utility_gradient_estimator="hard_multiplicity_ste",
        predictive_continuation="qkvs",
    )
    hook.remove()

    reference_score = next(score for score in scores if score.position == 6)
    assert len(normal_logits) == len(stream.tokens) - 1 - 6
    assert len(training_logits) == 3
    assert torch.equal(training_logits[1], normal_logits[0])
    assert torch.equal(training_logits[2], normal_logits[0])
    assert metrics["hard_selected_next_token_loss"] == reference_score.loss
    assert metrics["utility_forward_logit_max_abs_error"] == 0.0
    assert metrics["utility_forward_loss_abs_error"] == 0.0
    assert metrics["historical_windows"] == ((1, 3), (0, 2))
    assert metrics["overlap_occurrences"] == 1
    assert downstream_indices
    assert any(len(indices) > 2 for indices in downstream_indices)
    assert all(indices[-2:] == [5, 6] for indices in downstream_indices)
    assert all(all(index < 5 for index in indices[:-2]) for indices in downstream_indices)
    assert all(
        context.lower_layer is not None and not context.lower_layer.requires_grad
        for context in proposal_contexts
        if context.layer > 0
    )

    objective.backward()
    target_gradient = policy.decoders[0][-1].bias.grad.view(2, 4)
    assert torch.isfinite(target_gradient).all()
    assert all(torch.count_nonzero(target_gradient[:, field]) for field in (0, 1, 2, 3))
    assert all(
        parameter.grad is None
        for decoder in policy.decoders[1:]
        for parameter in decoder.parameters()
    )
    assert all(parameter.grad is None for parameter in reader.model.parameters())


def test_full_qkvs_training_objective_is_suffix_causal() -> None:
    reader = _qwen_reader()
    first_policy = _configured_heatfield_policy(gate_raw=4.0)
    second_policy = _configured_heatfield_policy(gate_raw=4.0)
    second_policy.load_state_dict(first_policy.state_dict())
    first = _stream((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    second = _stream((1, 2, 3, 4, 5, 6, 7, 8, 31, 30))
    snapshot = reader.dense_prefill(
        first,
        rolling_tokens=2,
        activation_tokens=6,
        device=torch.device("cpu"),
        ownership="train",
    )

    def objective(stream, policy):
        return reader.heatfield_training_objective(
            stream,
            policy,
            position=6,
            layer_index=0,
            rolling_tokens=2,
            historical_token_budget=4,
            window_width=2,
            coverage_weight=1.0,
            utility_weight=1.0,
            device=torch.device("cpu"),
            activation_tokens=6,
            dense_snapshot=snapshot,
            snapshot_ownership="train",
            utility_gradient_estimator="hard_multiplicity_ste",
            predictive_continuation="qkvs",
        )

    first_objective, first_metrics = objective(first, first_policy)
    second_objective, second_metrics = objective(second, second_policy)

    assert torch.equal(first_objective.detach(), second_objective.detach())
    assert first_metrics == second_metrics


def test_learned_gates_can_decline_all_historical_retrieval() -> None:
    policy = _heatfield_policy()
    with torch.no_grad():
        output = policy.decoders[0][-1]
        output.weight.zero_()
        output.bias.view(2, 4)[:, 3].fill_(-100.0)
    proposal_set = policy.propose(
        SelectionContext(
            position=10,
            layer=0,
            history_end=8,
            hidden=torch.zeros(32),
        ),
        PolicyState(),
    )
    selection = resolve_proposals(
        proposal_set.proposals,
        history_end=8,
        window_width=2,
        historical_token_budget=4,
        beta=proposal_set.beta,
    )

    assert not any(proposal.active for proposal in proposal_set.proposals)
    assert selection.occurrence_count == 0


def test_fixed_offsets_duplicate_the_initial_window_at_long_context_activation() -> None:
    policy = FixedOffsetPolicy(tuple(range(4096, 12161, 128)))
    proposal_set = policy.propose(
        SelectionContext(
            position=6144,
            layer=0,
            history_end=2049,
            hidden=None,
        ),
        PolicyState(),
    )
    selection = resolve_proposals(
        proposal_set.proposals,
        history_end=2049,
        window_width=8,
        historical_token_budget=512,
    )

    assert selection.windows.count(tuple(range(8))) == 48
    assert selection.occurrence_count == 512
    assert selection.unique_count == 136


def test_training_rollout_batches_metrics_and_reuses_gather_indices(monkeypatch) -> None:
    reader = _qwen_reader()
    policy = _heatfield_policy()
    stream = _stream(tuple(index % 32 for index in range(14)))
    snapshot = reader.dense_prefill(
        stream,
        rolling_tokens=2,
        activation_tokens=6,
        device=torch.device("cpu"),
        ownership="train",
    )
    state = reader.new_heatfield_rollout_state(
        stream,
        snapshot,
        rolling_tokens=2,
        activation_tokens=6,
        device=torch.device("cpu"),
        ownership="train",
        prefill_chunk_tokens=128,
    )
    gathered_indices = []
    original_gather = model_module._gather_cache_occurrences

    def counted_gather(cache, indices, *, occurrence_limit, base_occurrence_count=None):
        if isinstance(cache, model_module._LayerKVCache):
            gathered_indices.append(indices)
        return original_gather(
            cache,
            indices,
            occurrence_limit=occurrence_limit,
            base_occurrence_count=base_occurrence_count,
        )

    monkeypatch.setattr(model_module, "_gather_cache_occurrences", counted_gather)
    objectives, metrics = reader.heatfield_training_rollout(
        stream,
        policy,
        state,
        stop_position=10,
        target_layers={6: 0, 9: 1},
        rolling_tokens=2,
        historical_token_budget=4,
        window_width=2,
        coverage_weight=1.0,
        utility_weight=0.1,
        device=torch.device("cpu"),
    )

    assert len(objectives) == len(metrics) == 2
    assert all(isinstance(record["objective"], float) for record in metrics)
    assert all(isinstance(record["proposal_parameters"], list) for record in metrics)
    tensor_indices = [indices for indices in gathered_indices if isinstance(indices, torch.Tensor)]
    assert tensor_indices
    assert len(tensor_indices) % 2 == 0
    assert all(
        tensor_indices[index] is tensor_indices[index + 1]
        for index in range(0, len(tensor_indices), 2)
    )
    assert sum(indices is None for indices in gathered_indices) >= 6


def test_qkvs_training_intervention_branches_do_not_mutate_rollout_state() -> None:
    reader = _qwen_reader()
    policy = _configured_heatfield_policy(gate_raw=4.0)
    control_policy = _configured_heatfield_policy(gate_raw=4.0)
    control_policy.load_state_dict(policy.state_dict())
    stream = _stream(tuple(index % 32 for index in range(14)))
    snapshot = reader.dense_prefill(
        stream,
        rolling_tokens=2,
        activation_tokens=6,
        device=torch.device("cpu"),
        ownership="train",
    )
    state = reader.new_heatfield_rollout_state(
        stream,
        snapshot,
        rolling_tokens=2,
        activation_tokens=6,
        device=torch.device("cpu"),
        ownership="train",
        prefill_chunk_tokens=128,
    )
    control_state = reader.new_heatfield_rollout_state(
        stream,
        snapshot,
        rolling_tokens=2,
        activation_tokens=6,
        device=torch.device("cpu"),
        ownership="train",
        prefill_chunk_tokens=128,
    )

    objectives, metrics = reader.heatfield_training_rollout(
        stream,
        policy,
        state,
        stop_position=10,
        target_layers={6: 0, 9: 1},
        rolling_tokens=2,
        historical_token_budget=4,
        window_width=2,
        coverage_weight=1.0,
        utility_weight=1.0,
        device=torch.device("cpu"),
        utility_gradient_estimator="hard_multiplicity_ste",
        predictive_continuation="qkvs",
    )
    control_objectives, control_metrics = reader.heatfield_training_rollout(
        stream,
        control_policy,
        control_state,
        stop_position=10,
        target_layers={},
        rolling_tokens=2,
        historical_token_budget=4,
        window_width=2,
        coverage_weight=1.0,
        utility_weight=1.0,
        device=torch.device("cpu"),
        utility_gradient_estimator="hard_multiplicity_ste",
        predictive_continuation="qkvs",
    )

    assert len(objectives) == len(metrics) == 2
    assert control_objectives == control_metrics == []
    assert state.next_position == control_state.next_position == 10
    assert all(
        torch.equal(cache.snapshot(), control.snapshot())
        for cache, control in zip(state.key_cache, control_state.key_cache, strict=True)
    )
    assert all(
        torch.equal(cache.snapshot(), control.snapshot())
        for cache, control in zip(state.value_cache, control_state.value_cache, strict=True)
    )
    for actual, control in zip(state.policy_states, control_state.policy_states, strict=True):
        assert actual.centers == control.centers
        assert torch.equal(actual.recurrent, control.recurrent)


def test_heatfield_training_updates_only_s_and_checkpoint_executes(tmp_path) -> None:
    reader = _qwen_reader()
    policy = _heatfield_policy()
    stream = _stream(tuple(index % 32 for index in range(14)))
    reader_before = [parameter.detach().clone() for parameter in reader.model.parameters()]
    policy_before = [parameter.detach().clone() for parameter in policy.parameters()]
    dense_snapshot = reader.dense_prefill(
        stream,
        rolling_tokens=2,
        activation_tokens=6,
        device=torch.device("cpu"),
        ownership="train",
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    training_metrics = []
    for position, layer_index in ((8, 0), (9, 1), (10, 0)):
        optimizer.zero_grad(set_to_none=True)
        objective, metrics = reader.heatfield_training_objective(
            stream,
            policy,
            position=position,
            layer_index=layer_index,
            rolling_tokens=2,
            historical_token_budget=4,
            window_width=2,
            coverage_weight=1.0,
            utility_weight=0.1,
            device=torch.device("cpu"),
            activation_tokens=6,
            dense_snapshot=dense_snapshot,
            snapshot_ownership="train",
        )
        assert torch.isfinite(objective)
        objective.backward()
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in policy.parameters()
        )
        assert all(parameter.grad is None for parameter in reader.model.parameters())
        optimizer.step()
        training_metrics.append(metrics)

    assert any(
        not torch.equal(before, after)
        for before, after in zip(policy_before, policy.parameters(), strict=True)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(reader_before, reader.model.parameters(), strict=True)
    )
    assert all(metric["historical_occurrences"] <= 4 for metric in training_metrics)
    assert all(metric["coverage_overlap_alpha"] == 0.0 for metric in training_metrics)
    assert all(
        metric["coverage_loss"]
        == pytest.approx(metric["teacher_historical_attention_uncovered_mass"])
        for metric in training_metrics
    )
    assert all(
        metric["teacher_historical_attention_covered_mass"]
        + metric["teacher_historical_attention_uncovered_mass"]
        == pytest.approx(metric["teacher_historical_attention_mass"])
        for metric in training_metrics
    )
    assert any(metric["historical_occurrences"] > 0 for metric in training_metrics)
    assert any(metric["hard_utility"] != 0.0 for metric in training_metrics)
    assert all(
        metric["utility_gradient_estimator"] == "soft_occupancy" for metric in training_metrics
    )
    assert all(metric["trajectory"] == "learned_hard_detached" for metric in training_metrics)
    assert all(
        metric["temporal_credit_assignment"] == "target_step_only" for metric in training_metrics
    )
    assert any(metric["trajectory_historical_occurrences"] > 0 for metric in training_metrics)
    assert all(metric["dense_prefix_reused"] for metric in training_metrics)

    checkpoint = tmp_path / "selector.pt"
    torch.save(policy.state_dict(), checkpoint)
    restored = _heatfield_policy()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    original_scores = reader.execute(
        stream,
        policy,
        rolling_tokens=2,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
        activation_tokens=6,
        dense_snapshot=dense_snapshot,
        snapshot_ownership="train",
    )
    restored_scores = reader.execute(
        stream,
        restored,
        rolling_tokens=2,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
        activation_tokens=6,
        dense_snapshot=dense_snapshot,
        snapshot_ownership="train",
    )
    assert [score.as_dict() for score in original_scores] == [
        score.as_dict() for score in restored_scores
    ]
    assert all(
        len(access.historical_indices) <= 4
        and all(index < access.history_end for index in access.historical_indices)
        for score in restored_scores
        for access in score.accesses
    )


def test_heatfield_harness_training_resumes_atomically(tmp_path, monkeypatch, capfd) -> None:
    from qkvs.harness import load_run, print_summary, run_experiment, status, validate_config

    class ByteTokenizer:
        init_kwargs = {"_commit_hash": "1" * 40}

        def encode(self, text, *, add_special_tokens, verbose):
            assert not add_special_tokens
            assert not verbose
            return [byte % 64 for byte in text.encode()]

        def get_vocab(self):
            return {str(index): index for index in range(64)}

    reader = _qwen_reader()
    reader.tokenizer = ByteTokenizer()
    source = tmp_path / "source.txt"
    source.write_text("bounded heat field smoke", encoding="utf-8")
    specification = {
        "schema_version": 1,
        "experiment": "learned_heatfield_smoke",
        "status": "harness_verification",
        "run": {"id": "learned-heatfield-smoke", "seed": 71},
        "model": {
            "backend": "huggingface",
            "id": "test/tiny-qwen2",
            "revision": "1" * 40,
            "dtype": "float32",
            "frozen": True,
        },
        "data": {
            "kind": "tokenized_text",
            "source": {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            "boundaries": [
                {"id": "train", "split": "train", "start_token": 0, "stop_token": 12},
                {
                    "id": "evaluation",
                    "split": "evaluation",
                    "start_token": 12,
                    "stop_token": 24,
                },
            ],
        },
        "attention": {
            "rolling_tokens": 2,
            "activation_tokens": 4,
            "historical_token_budget": 4,
        },
        "selection": {
            "proposal_count": 2,
            "window_width": 2,
            "gamma": {"mode": "fixed", "value": 0.5},
            "beta": {"mode": "fixed", "value": 1.25},
            "learned": {
                "latent_size": 16,
                "minimum_spread": 0.5,
                "maximum_spread": 2.0,
                "gate_threshold": 0.4,
            },
            "conditions": [
                {"name": "rolling", "policy": "rolling"},
                {"name": "full_qkv", "policy": "full_qkv"},
                {"name": "untrained_s", "policy": "heatfield_untrained"},
                {"name": "heatfield", "policy": "heatfield"},
            ],
        },
        "training": {
            "enabled": True,
            "steps": 2,
            "learning_rate": 0.01,
            "coverage_weight": 1.0,
            "utility_weight": 0.1,
            "utility_gradient_estimator": "hard_multiplicity_ste",
            "predictive_continuation": "qkvs",
            "checkpoint_interval": 1,
            "rollout_tokens": 4,
            "targets_per_rollout": 3,
            "position_schedule": "streaming_stratified",
            "layer_schedule": "round_robin",
        },
        "evaluation": {
            "metric": "next_token_cross_entropy",
            "paired_conditions": True,
            "proposal_trace_interval": 1,
            "baseline": "rolling",
            "comparisons": [
                {
                    "name": "trained_vs_untrained",
                    "baseline": "untrained_s",
                    "condition": "heatfield",
                }
            ],
            "bootstrap_samples": 20,
        },
        "execution": {
            "allowed_devices": ["cpu"],
            "output_root": str(tmp_path),
            "prefill_chunk_tokens": 2,
        },
    }
    assert validate_config(specification)["runnable"]
    overlapping = json.loads(json.dumps(specification))
    overlapping["data"]["boundaries"][1]["start_token"] = 10
    overlap_report = validate_config(overlapping)
    assert not overlap_report["runnable"]
    assert (
        "learned heat-field train and evaluation boundaries overlap"
        in overlap_report["not_runnable_reasons"]
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps(specification), encoding="utf-8")
    monkeypatch.setattr(model_module, "load_reader", lambda model, device: reader)
    original_dense_prefill = reader.dense_prefill
    prefill_calls = 0

    def counted_dense_prefill(*args, **kwargs):
        nonlocal prefill_calls
        prefill_calls += 1
        return original_dense_prefill(*args, **kwargs)

    monkeypatch.setattr(reader, "dense_prefill", counted_dense_prefill)
    original_rollout = reader.heatfield_training_rollout
    rollout_calls = 0

    def interrupted_rollout(*args, **kwargs):
        nonlocal rollout_calls
        rollout_calls += 1
        if rollout_calls == 2:
            raise RuntimeError("test interruption after an atomic rollout checkpoint")
        return original_rollout(*args, **kwargs)

    monkeypatch.setattr(reader, "heatfield_training_rollout", interrupted_rollout)
    with pytest.raises(RuntimeError, match="test interruption"):
        run_experiment(
            config,
            device="cpu",
            output_root=tmp_path / "results",
            allow_dirty_source=True,
        )
    partial_output = tmp_path / "results" / "learned-heatfield-smoke" / "result.json"
    partial = json.loads(partial_output.read_text())
    assert partial["training"]["completed_steps"] == 1
    assert partial["training"]["streaming"]["next_positions"] == {"train": 8}
    assert partial["training"]["streaming"]["effective_supervised_targets"] == 3
    assert partial["training"]["streaming"]["effective_supervised_targets_per_layer"] == {
        "0": 2,
        "1": 1,
    }
    partial_status = status(partial_output)
    assert partial_status["completed_units"] == 3
    assert partial_status["total_units"] == 8
    assert partial_status["phase_units"] == {
        "dense_prefill": {"completed": 2, "total": 2},
        "training_checkpoints": {"completed": 1, "total": 2},
        "evaluation": {"completed": 0, "total": 4},
    }
    assert partial_status["training_steps"] == {"completed": 1, "total": 2}
    interrupted_progress = capfd.readouterr().out
    assert "dense prefill / train" in interrupted_progress
    assert "train 1/2 0/6 targets" in interrupted_progress
    assert "train 2/2 3/6 targets" in interrupted_progress
    monkeypatch.setattr(reader, "heatfield_training_rollout", original_rollout)
    output, first = run_experiment(
        config,
        device="cpu",
        output_root=tmp_path / "results",
        allow_dirty_source=True,
    )
    assert first["complete"]
    assert prefill_calls == 2
    assert set(first["dense_prefill"]) == {"train", "evaluation"}
    assert all(
        (output.parent / record["path"]).exists() for record in first["dense_prefill"].values()
    )
    assert first["training"]["completed_steps"] == 2
    assert first["training"]["streaming"]["effective_supervised_targets"] == 6
    assert first["training"]["streaming"]["effective_supervised_targets_per_layer"] == {
        "0": 3,
        "1": 3,
    }
    assert first["training"]["streaming"]["utility_gradient_estimator"] == "hard_multiplicity_ste"
    assert first["training"]["streaming"]["predictive_continuation"] == "qkvs"
    assert all(
        record["utility_gradient_estimator"] == "hard_multiplicity_ste"
        and record["predictive_continuation"] == "qkvs"
        and record["predictive_utility_scope"] == "target_layer_marginal"
        and not record["full_trajectory_utility_measured"]
        and record["utility_forward_logit_max_abs_error"] <= record["hard_forward_logit_atol"]
        and record["utility_forward_loss_abs_error"] <= record["hard_forward_loss_atol"]
        for record in first["training"]["history"]
    )
    assert all(record["gradient_norm"] > 0.0 for record in first["training"]["history"])
    assert [record["position"] for record in first["training"]["history"]] == [
        4,
        6,
        7,
        8,
        9,
        10,
    ]
    assert all(
        record["selector_weights_fixed_within_rollout"] for record in first["training"]["history"]
    )
    assert first["training"]["streaming"]["next_positions"] == {"train": 11}
    assert {record["stream_id"] for record in first["training"]["history"]} == {
        "train",
    }
    assert (output.parent / "selector.pt").exists()
    heatfield = next(
        measurement for measurement in first["measurements"] if measurement["policy"] == "heatfield"
    )
    assert any(
        access["proposal_weights"]
        for score in heatfield["token_scores"]
        for access in score["accesses"]
    )
    assert [score["phase"] for score in heatfield["token_scores"][:4]] == [
        "dense_bootstrap",
        "dense_bootstrap",
        "dense_bridge",
        "dense_bridge",
    ]
    assert all(
        not score["accesses"]
        for score in heatfield["token_scores"]
        if score["phase"] != "post_transition"
    )
    assert {measurement["stream_id"] for measurement in first["measurements"]} == {"evaluation"}
    assert {measurement["condition"] for measurement in first["measurements"]} == {
        "rolling",
        "full_qkv",
        "untrained_s",
        "heatfield",
    }
    assert first["training"]["soft_hard_alignment"]["paired_steps"] == 6
    assert first["training"]["soft_hard_alignment"]["soft_hard_utility_correlation"] is not None
    assert 0.0 <= first["training"]["soft_hard_alignment"]["utility_sign_agreement"] <= 1.0
    assert set(first["summary"]["conditions"]) == {
        "rolling",
        "full_qkv",
        "untrained_s",
        "heatfield",
    }
    assert "untrained_s" in first["summary"]["paired_utility"]
    assert "heatfield" in first["summary"]["paired_utility"]
    assert "trained_vs_untrained" in first["summary"]["paired_utility"]
    descriptive = first["summary"]["paired_utility"]["trained_vs_untrained"]
    assert descriptive["paired_streams"] == 1
    assert descriptive["paired_bootstrap_95_interval_nats"] is None
    assert descriptive["statistical_interpretation"] == "descriptive_single_stream"
    assert set(descriptive["per_stream_utility_nats"]) == {"evaluation"}
    assert all(
        first["summary"]["post_transition_region_paired_utility"][region]["trained_vs_untrained"][
            "statistical_interpretation"
        ]
        == "descriptive_single_stream"
        for region in ("early", "middle", "late")
    )
    assert all(
        "historical_budget_utilization" in condition["retrieval"]
        for condition in first["summary"]["conditions"].values()
    )
    assert (
        first["summary"]["conditions"]["full_qkv"]["retrieval"]["historical_budget_utilization"]
        is None
    )
    assert first["summary"]["conditions"]["heatfield"]["post_transition"]["scored_tokens"]
    print_summary(first)
    assert "(descriptive_single_stream)" in capfd.readouterr().out

    _, resumed = run_experiment(
        config,
        device="cpu",
        output_root=tmp_path / "results",
        allow_dirty_source=True,
    )
    assert resumed == first
    assert prefill_calls == 2

    uninterrupted_output, uninterrupted = run_experiment(
        config,
        device="cpu",
        output_root=tmp_path / "uninterrupted-results",
        allow_dirty_source=True,
    )
    assert uninterrupted["training"]["history"] == first["training"]["history"]
    resumed_policy = torch.load(output.parent / "selector.pt", weights_only=True)["policy"]
    uninterrupted_policy = torch.load(
        uninterrupted_output.parent / "selector.pt", weights_only=True
    )["policy"]
    assert all(
        torch.equal(resumed_policy[name], uninterrupted_policy[name]) for name in resumed_policy
    )

    reused_specification = json.loads(json.dumps(specification))
    reused_specification["run"]["id"] = "learned-heatfield-reused-baselines"
    reused_specification["training"]["coverage_weight"] = 0.5
    reused_config = tmp_path / "reused-config.json"
    reused_config.write_text(json.dumps(reused_specification), encoding="utf-8")
    reused_output, reused = run_experiment(
        reused_config,
        device="cpu",
        output_root=tmp_path / "results",
        allow_dirty_source=True,
    )
    raw_reused = json.loads(reused_output.read_text())
    assert {reference["condition"] for reference in raw_reused["measurement_references"]} == {
        "rolling",
        "full_qkv",
        "untrained_s",
    }
    assert {measurement["condition"] for measurement in raw_reused["measurements"]} == {"heatfield"}
    assert all(
        reference["source_run_id"] == "learned-heatfield-smoke"
        and not reference["performance_reused"]
        for reference in raw_reused["measurement_references"]
    )
    assert len(reused["measurements"]) == 4
    reused_baselines = [
        measurement
        for measurement in reused["measurements"]
        if measurement["condition"] != "heatfield"
    ]
    assert all("predictive_reuse" in measurement for measurement in reused_baselines)
    assert all("performance" not in measurement for measurement in reused_baselines)
    assert all(
        reused["summary"]["conditions"][measurement["condition"]]["measured_execution"]["streams"]
        == 0
        for measurement in reused_baselines
    )
    assert load_run(reused_output) == reused
    reused_status = status(reused_output)
    assert reused_status["complete"]
    assert reused_status["phase_units"]["evaluation"] == {"completed": 4, "total": 4}
    raw_reused["measurement_references"][0]["source_artifact_sha256"] = "0" * 64
    reused_output.write_text(json.dumps(raw_reused), encoding="utf-8")
    with pytest.raises(ValueError, match="source artifact identity"):
        load_run(reused_output)


@pytest.mark.skipif(
    "QKVS_PRETRAINED_MODEL" not in os.environ,
    reason="set QKVS_PRETRAINED_MODEL to a complete local Qwen2 checkpoint",
)
def test_local_pretrained_qwen2_full_history_reference(tmp_path, monkeypatch) -> None:
    checkpoint = Path(os.environ["QKVS_PRETRAINED_MODEL"])
    for name, expected in _PINNED_FILE_SHA256.items():
        actual = hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        assert actual == expected, f"unexpected {name} identity: {actual}"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    reader = HuggingFaceQwen2Reader(
        model,
        tokenizer,
        specification={
            "id": _PINNED_MODEL_ID,
            "revision": _PINNED_MODEL_REVISION,
            "dtype": "float32",
            "frozen": True,
        },
    )
    identity = reader.identity()
    assert identity["model_id"] == _PINNED_MODEL_ID
    assert identity["model_revision"] == _PINNED_MODEL_REVISION
    assert identity["architecture"] == "Qwen2ForCausalLM"
    assert identity["attention_heads"] == 14
    assert identity["key_value_heads"] == 2
    assert identity["rope_theta"] == 1_000_000.0
    assert identity["tokenizer_vocabulary_sha256"]

    texts = (
        "The observatory clock marked midnight while every instrument continued recording.",
        "A bounded cache preserves duplicate reads without scanning the complete history.",
        "Numbers two, three, five, eight, and thirteen begin this deterministic sequence.",
    )
    diagnostics = []
    streams = []
    for index, text in enumerate(texts):
        tokens = tuple(tokenizer.encode(text, add_special_tokens=False))
        tokens = tokens[: min(len(tokens), 12)]
        assert len(tokens) >= 6
        diagnostic = reader.full_history_diagnostic(
            tokens,
            device=torch.device("cpu"),
            atol=5e-4,
            rtol=2e-4,
            loss_atol=5e-5,
            loss_rtol=2e-4,
        )
        assert diagnostic["allclose"], diagnostic
        assert diagnostic["loss_allclose"], diagnostic
        diagnostics.append(diagnostic)
        streams.append(TokenStream(f"pretrained-{index}", tokens, (None,) * (len(tokens) - 1)))

    rolling = reader.execute(
        streams[0],
        RollingOnlyPolicy(),
        rolling_tokens=4,
        activation_tokens=4,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
    )
    selected = reader.execute(
        streams[0],
        FixedOffsetPolicy((6, 10)),
        rolling_tokens=4,
        activation_tokens=4,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
    )
    selected_accesses = [access for score in selected for access in score.accesses]
    assert any(access.historical_indices for access in selected_accesses)
    assert all(len(access.actual_indices) <= 8 for access in selected_accesses)
    assert all(
        access.gathered_key_occurrences
        == access.gathered_value_occurrences
        == len(access.actual_indices)
        for access in selected_accesses
    )
    loss_differences = [
        selected_score.loss - rolling_score.loss
        for rolling_score, selected_score in zip(rolling, selected, strict=True)
    ]
    assert any(difference != 0.0 for difference in loss_differences)

    prefix = streams[0].tokens[:6]
    first_suffix = tuple(tokenizer.encode(" first branch", add_special_tokens=False))[:3]
    second_suffix = tuple(tokenizer.encode(" second branch", add_special_tokens=False))[:3]
    assert first_suffix != second_suffix
    first = TokenStream(
        "pretrained-suffix-a",
        (*prefix, *first_suffix),
        (None,) * (len(prefix) + len(first_suffix) - 1),
    )
    second = TokenStream(
        "pretrained-suffix-b",
        (*prefix, *second_suffix),
        (None,) * (len(prefix) + len(second_suffix) - 1),
    )
    first_scores = reader.execute(
        first,
        FixedOffsetPolicy((6, 10)),
        rolling_tokens=4,
        activation_tokens=4,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
    )
    second_scores = reader.execute(
        second,
        FixedOffsetPolicy((6, 10)),
        rolling_tokens=4,
        activation_tokens=4,
        historical_token_budget=4,
        window_width=2,
        device=torch.device("cpu"),
    )
    assert [score.as_dict() for score in first_scores[: len(prefix) - 1]] == [
        score.as_dict() for score in second_scores[: len(prefix) - 1]
    ]

    snapshot_stream = streams[0]
    snapshot = reader.dense_causal_snapshot(
        snapshot_stream,
        position=4,
        device=torch.device("cpu"),
        ownership="pretrained-snapshot-qualification",
        prefill_chunk_tokens=2,
    )
    snapshot_state = reader.prepare_snapshot_probe(
        snapshot,
        snapshot_stream,
        position=4,
        device=torch.device("cpu"),
        ownership="pretrained-snapshot-qualification",
        prefill_chunk_tokens=2,
    )
    snapshot_reference = reader.snapshot_probe(
        snapshot_stream,
        snapshot_state,
        position=4,
        selected_positions=None,
        device=torch.device("cpu"),
    )
    snapshot_dense = reader.snapshot_probe(
        snapshot_stream,
        snapshot_state,
        position=4,
        selected_positions=tuple(tuple(range(5)) for _ in reader.model.model.layers),
        device=torch.device("cpu"),
    )
    assert torch.equal(snapshot_reference.logits, snapshot_dense.logits)
    assert torch.equal(snapshot_reference.loss, snapshot_dense.loss)

    from qkvs.harness import _tokenized_text_streams, run_experiment, validate_config_file

    source_path = tmp_path / "source.txt"
    source_path.write_text(texts[0], encoding="utf-8")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    config_path = tmp_path / "qualification.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "pretrained_reader_qualification",
                "status": "harness_verification",
                "run": {"id": "pretrained-qualification", "seed": 20261018},
                "model": {
                    "backend": "huggingface",
                    "id": _PINNED_MODEL_ID,
                    "revision": _PINNED_MODEL_REVISION,
                    "dtype": "float32",
                    "frozen": True,
                },
                "data": {
                    "kind": "tokenized_text",
                    "source": {"path": str(source_path), "sha256": source_sha256},
                    "boundaries": [{"id": "qualification", "start_token": 0, "stop_token": 8}],
                },
                "attention": {
                    "rolling_tokens": 2,
                    "activation_tokens": 4,
                    "historical_token_budget": 2,
                },
                "selection": {
                    "proposal_count": 2,
                    "window_width": 1,
                    "gamma": {"mode": "disabled"},
                    "beta": {"mode": "disabled"},
                    "conditions": [
                        {"name": "rolling", "policy": "rolling"},
                        {
                            "name": "fixed_offsets",
                            "policy": "fixed_offsets",
                            "offsets": [4, 6],
                        },
                    ],
                },
                "evaluation": {
                    "metric": "next_token_cross_entropy",
                    "paired_conditions": True,
                    "proposal_trace_interval": 1,
                    "baseline": "rolling",
                    "bootstrap_samples": 20,
                },
                "execution": {
                    "allowed_devices": ["cpu"],
                    "output_root": str(tmp_path),
                    "prefill_chunk_tokens": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_module, "load_reader", lambda specification, device: reader)
    output, first_result = run_experiment(
        config_path,
        device="cpu",
        output_root=tmp_path / "results",
        allow_dirty_source=True,
    )
    _, resumed_result = run_experiment(
        config_path,
        device="cpu",
        output_root=tmp_path / "results",
        allow_dirty_source=True,
    )
    assert resumed_result == first_result
    assert first_result["complete"]
    assert first_result["model_identity"] == identity
    assert first_result["data_identity"]["source_sha256"] == source_sha256
    assert first_result["source_identity"]["git_revision"]
    assert not first_result["environment"]["cuda_initialized_before_reader"]
    assert not first_result["environment"]["cuda_initialized_after_reader"]
    assert not output.with_suffix(".json.tmp").exists()
    draft = validate_config_file("configs/heatfield_initial.json")
    assert draft["valid"]
    assert not draft["runnable"]
    study_path = Path("configs/heatfield_learning_study.json")
    study_specification = json.loads(study_path.read_text())
    study_streams, study_identity = _tokenized_text_streams(
        study_specification,
        config_path=study_path,
        tokenizer=tokenizer,
    )
    assert [stream["split"] for stream in study_streams] == ["train", "evaluation"]
    assert all(len(stream["tokens"]) == 16384 for stream in study_streams)
    assert study_identity["source_sha256"] == study_specification["data"]["source"]["sha256"]

    print(
        json.dumps(
            {
                "identity": identity,
                "prefix_diagnostics": diagnostics,
                "bounded_loss_differences": loss_differences,
                "bounded_last_access": selected[-1].accesses[-1].as_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
