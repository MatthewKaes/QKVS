from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from qkvs.causal_reference import rolling_mask_plan
from qkvs.stage_b_numerical_validation import (
    _cache_structure_passed,
    _execution_status,
    _incremental_hidden_cache,
    _layer_zero_kv_comparison,
    _recover_interrupted_unit,
    _validate_proposal,
)


def _model() -> Qwen2ForCausalLM:
    torch.manual_seed(41)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval()


def test_replacement_proposal_is_authorized_without_authorizing_gate_zero() -> None:
    proposal = json.loads(
        Path("configs/stage_b_numerical_validation_replacement_proposal.json").read_text()
    )
    _validate_proposal(proposal)
    assert proposal["validation_examples"]["indices"] == [1, 2, 4, 6]
    assert proposal["status"] == "approved_for_numerical_validation_only"
    assert not proposal["conditional_gate_0_amendment"]["authorized_by_this_proposal"]


def test_actual_incremental_cache_matches_corrected_layer_zero_reference() -> None:
    model = _model()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 24))
    plan = rolling_mask_plan(
        length=24,
        n_layers=model.config.num_hidden_layers,
        rolling_tokens=8,
        global_prefix_end=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    example = SimpleNamespace(
        block_ranges=((4, 8),),
        target_block=0,
        global_prefix_end=2,
        query_start=20,
        input_ids=tuple(input_ids[0].tolist()),
    )
    with torch.inference_mode():
        _, cache, cache_summary = _incremental_hidden_cache(model, input_ids, plan.masks)
        comparison, _, _ = _layer_zero_kv_comparison(
            model,
            input_ids,
            cache,
            example=example,
        )
    assert _cache_structure_passed(cache_summary)
    assert comparison["structure_passed"]
    assert comparison["key"]["max_absolute_error"] <= 1e-6
    assert comparison["value"]["max_absolute_error"] <= 1e-6
    assert comparison["numerical_acceptance"] == "report_only_no_authorized_tolerance"


def test_cache_structure_rejects_noncontiguous_positions() -> None:
    summary = {
        "first_length_mismatch": None,
        "final_layer_lengths": [24, 24],
        "all_layers_have_full_length": True,
        "supplied_cache_positions": {
            "count": 24,
            "first": 0,
            "last": 23,
            "contiguous_zero_based": False,
        },
    }
    assert not _cache_structure_passed(summary)


def test_interrupted_unit_is_recoverable_from_last_atomic_checkpoint() -> None:
    active = {"example_index": 1, "condition": "bf16_sdpa_default"}
    result = {
        "source_identity": {"git_revision": "revision"},
        "decision": {"outcome": "incomplete"},
        "examples": {
            "1": {"conditions": {}},
            "2": {"conditions": {}},
        },
        "execution_state": {
            "status": "running",
            "active_unit": active,
            "attempts": [{"unit": active, "status": "running"}],
            "recovery_events": [],
        },
    }
    assert _recover_interrupted_unit(result)
    assert result["execution_state"]["active_unit"] is None
    assert result["execution_state"]["attempts"][0]["status"] == ("interrupted_before_checkpoint")
    assert result["execution_state"]["recovery_events"][0]["unit"] == active

    status = _execution_status(
        result,
        examples=[SimpleNamespace(index=1), SimpleNamespace(index=2)],
        conditions=[{"name": "bf16_sdpa_default"}, {"name": "fp32_sdpa_math"}],
    )
    assert status["completed_units"] == 0
    assert status["total_units"] == 4
    assert status["next_unit"] == active
