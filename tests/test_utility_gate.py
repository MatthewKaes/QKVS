from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from qkvs.utility_gate import (
    _balanced_shuffled_schedule,
    _rolling_receptive_tokens,
    _squad_records,
    _validate_resume,
    build_attention_mask,
    summarize,
)


def _allowed(mask: torch.Tensor) -> torch.Tensor:
    return mask[0, 0] == 0


def test_rolling_mask_is_causal_and_bounded() -> None:
    mask = build_attention_mask(
        8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rolling_tokens=3,
    )
    allowed = _allowed(mask)
    assert allowed[7].nonzero().flatten().tolist() == [5, 6, 7]
    assert allowed[2].nonzero().flatten().tolist() == [0, 1, 2]
    assert not allowed[3, 4]


def test_selected_history_is_visible_only_from_query_start() -> None:
    mask = build_attention_mask(
        10,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rolling_tokens=3,
        query_start=8,
        selected_ranges=[(1, 3)],
    )
    allowed = _allowed(mask)
    assert not allowed[7, 1]
    assert allowed[8, 1]
    assert allowed[9, 2]
    assert not allowed[8, 0]


def test_global_prefix_is_visible_to_every_later_position() -> None:
    mask = build_attention_mask(
        10,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rolling_tokens=3,
        global_ranges=[(0, 2)],
    )
    allowed = _allowed(mask)
    assert allowed[4, 0]
    assert allowed[9, 1]
    assert not allowed[0, 1]
    assert not allowed[9, 2]


def test_decisive_schedules_are_balanced_and_separately_shuffled() -> None:
    targets = _balanced_shuffled_schedule(8, 24, 20262018)
    labels = _balanced_shuffled_schedule(4, 24, 20263018)
    assert [targets.count(position) for position in range(8)] == [3] * 8
    assert [labels.count(label) for label in range(4)] == [6] * 4
    assert targets != [index % 8 for index in range(24)]
    assert labels != [index % 4 for index in range(24)]


def test_multilayer_receptive_field_uses_layerwise_window_overlap() -> None:
    assert _rolling_receptive_tokens(24, 256) == 6121


def test_resume_rejects_a_different_source_revision() -> None:
    expected = {
        "config": {"stage": "A"},
        "dataset_sha256": "dataset",
        "device": "cuda",
        "source_identity": {"git_revision": "new", "files": {"gate.py": "hash"}},
    }
    result = dict(expected)
    _validate_resume(result, expected)
    result["source_identity"] = {"git_revision": "old", "files": {"gate.py": "hash"}}
    with pytest.raises(ValueError, match="source identity"):
        _validate_resume(result, expected)


def test_squad_records_use_the_majority_answer_annotation() -> None:
    payload = {
        "data": [
            {
                "title": "example",
                "paragraphs": [
                    {
                        "context": "The designer was Owen Jones.",
                        "qas": [
                            {
                                "id": "question",
                                "question": "Who was the designer?",
                                "answers": [
                                    {"text": "designer", "answer_start": 4},
                                    {"text": "Owen Jones", "answer_start": 17},
                                    {"text": "Owen Jones", "answer_start": 17},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    assert _squad_records(payload)[0].answer == "Owen Jones"


def test_explicit_dense_mask_matches_builtin_causal_mask() -> None:
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 12))
    explicit = build_attention_mask(
        12,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with torch.inference_mode():
        builtin_logits = model(input_ids, use_cache=False).logits
        explicit_logits = model(input_ids, attention_mask=explicit, use_cache=False).logits
    torch.testing.assert_close(explicit_logits, builtin_logits, atol=1e-6, rtol=1e-6)


def test_incomplete_summary_never_passes_gate() -> None:
    example = {
        "target_block": 1,
        "random_block": 2,
        "recency_block": 3,
        "gold_label": 0,
        "modes": {
            name: {
                "loss": loss,
                "correct": correct,
                "utility_nats": 1.0 - loss,
                **({"selected_block": 1} if name == "utility_oracle" else {}),
            }
            for name, loss, correct in (
                ("full", 0.1, True),
                ("rolling", 1.0, False),
                ("random", 0.9, False),
                ("recency", 0.8, False),
                ("annotated_oracle", 0.2, True),
                ("utility_oracle", 0.2, True),
            )
        },
    }
    summary = summarize(
        [example],
        configured_examples=2,
        seed=1,
        gate={"minimum_full_accuracy": 0.5, "minimum_utility_lower_bound_nats": 0.05},
    )
    assert not summary["complete"]
    assert summary["gate"]["decision"] == "incomplete"
    assert not summary["gate"]["passed"]


def test_complete_summary_applies_predeclared_gate() -> None:
    example = {
        "target_block": 1,
        "random_block": 2,
        "recency_block": 3,
        "gold_label": 0,
        "modes": {
            name: {
                "loss": loss,
                "correct": correct,
                "utility_nats": 1.0 - loss,
                **({"selected_block": 1} if name == "utility_oracle" else {}),
            }
            for name, loss, correct in (
                ("full", 0.1, True),
                ("rolling", 1.0, False),
                ("random", 0.9, False),
                ("recency", 0.8, False),
                ("annotated_oracle", 0.2, True),
                ("utility_oracle", 0.2, True),
            )
        },
    }
    summary = summarize(
        [example] * 4,
        configured_examples=4,
        seed=1,
        gate={"minimum_full_accuracy": 0.5, "minimum_utility_lower_bound_nats": 0.05},
    )
    assert summary["complete"]
    assert summary["gate"]["decision"] == "proceed_to_stage_b"
    assert summary["gate"]["passed"]


def test_selective_utility_gate_does_not_require_full_attention_superiority() -> None:
    example = {
        "target_block": 1,
        "random_block": 2,
        "recency_block": 3,
        "gold_label": 0,
        "modes": {
            name: {
                "loss": loss,
                "correct": correct,
                "utility_nats": 1.0 - loss,
                **({"selected_block": 1} if name == "utility_oracle" else {}),
            }
            for name, loss, correct in (
                ("full", 2.0, False),
                ("rolling", 1.0, False),
                ("random", 0.9, False),
                ("recency", 0.8, False),
                ("annotated_oracle", 0.2, True),
                ("utility_oracle", 0.2, True),
            )
        },
    }
    summary = summarize(
        [example] * 4,
        configured_examples=4,
        seed=1,
        gate={
            "decision_rule": "selective_retrieval_utility",
            "minimum_annotated_accuracy": 0.5,
            "minimum_annotated_vs_rolling_lower_bound_nats": 0.05,
            "diagnostic_full_utility_lower_bound_nats": 0.05,
        },
    )
    assert summary["gate"]["passed"]
    assert summary["gate"]["decision"] == "proceed_to_stage_b"
    assert not summary["gate"]["diagnostics"]["full_beats_rolling_at_previous_threshold"]
