from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

import qkvs.model as model_module
from qkvs.model import FrozenCausalReader, HuggingFaceQwen2Reader, TokenStream
from qkvs.selection import (
    FixedOffsetPolicy,
    HeatFieldProposal,
    LearnedHeatFieldPolicy,
    PolicyState,
    ProposalSet,
    RollingOnlyPolicy,
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
        activation_tokens=2,
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
        activation_tokens=3,
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
    assert all(score.phase == "qkvs" for score in scores[6:])
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
    cache_before = tuple(
        value.clone() for layer_cache in snapshot.key_cache for value in layer_cache
    )
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
            (value for layer_cache in snapshot.key_cache for value in layer_cache),
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
        activation_tokens=2,
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


def _heatfield_policy() -> LearnedHeatFieldPolicy:
    torch.manual_seed(53)
    return LearnedHeatFieldPolicy(
        hidden_size=32,
        layers=2,
        proposal_count=2,
        latent_size=16,
        gamma=0.5,
        beta=1.25,
        minimum_spread=0.5,
        maximum_spread=2.0,
        gate_threshold=0.4,
    )


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
    assert any(metric["historical_occurrences"] > 0 for metric in training_metrics)
    assert any(metric["hard_utility"] != 0.0 for metric in training_metrics)
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


def test_heatfield_harness_training_resumes_atomically(tmp_path, monkeypatch) -> None:
    from qkvs.harness import run_experiment, validate_config

    class ByteTokenizer:
        init_kwargs = {"_commit_hash": "1" * 40}

        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
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
            "checkpoint_interval": 1,
            "position_schedule": "cyclic",
            "layer_schedule": "round_robin",
        },
        "evaluation": {
            "metric": "next_token_cross_entropy",
            "paired_conditions": True,
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
        "execution": {"allowed_devices": ["cpu"], "output_root": str(tmp_path)},
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
    assert all(record["gradient_norm"] > 0.0 for record in first["training"]["history"])
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
        not score["accesses"] for score in heatfield["token_scores"] if score["phase"] != "qkvs"
    )
    assert {measurement["stream_id"] for measurement in first["measurements"]} == {"evaluation"}
    assert {measurement["condition"] for measurement in first["measurements"]} == {
        "rolling",
        "untrained_s",
        "heatfield",
    }
    assert first["training"]["soft_hard_alignment"]["paired_steps"] == 2
    assert first["training"]["soft_hard_alignment"]["soft_hard_utility_correlation"] is not None
    assert 0.0 <= first["training"]["soft_hard_alignment"]["utility_sign_agreement"] <= 1.0
    assert set(first["summary"]["conditions"]) == {
        "rolling",
        "untrained_s",
        "heatfield",
    }
    assert "untrained_s" in first["summary"]["paired_utility"]
    assert "heatfield" in first["summary"]["paired_utility"]
    assert "trained_vs_untrained" in first["summary"]["paired_utility"]
    assert all(
        "historical_budget_utilization" in condition["retrieval"]
        for condition in first["summary"]["conditions"].values()
    )

    _, resumed = run_experiment(
        config,
        device="cpu",
        output_root=tmp_path / "results",
        allow_dirty_source=True,
    )
    assert resumed == first
    assert prefill_calls == 2


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
                    "baseline": "rolling",
                    "bootstrap_samples": 20,
                },
                "execution": {"allowed_devices": ["cpu"], "output_root": str(tmp_path)},
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
    assert [stream["split"] for stream in study_streams] == [
        "train",
        "train",
        "evaluation",
        "evaluation",
    ]
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
