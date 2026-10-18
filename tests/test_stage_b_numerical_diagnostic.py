from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from qkvs.causal_reference import (
    causal_oracle_mask_plan,
    qwen_forward_with_layer_masks,
    qwen_incremental_with_layer_masks,
    rolling_mask_plan,
)
from qkvs.stage_b_numerical_diagnostic import (
    _forward_trace,
    _incremental_trace,
    _mask_structure,
    _trace_summary,
)
from qkvs.utility_gate import _sha256


def _model() -> Qwen2ForCausalLM:
    torch.manual_seed(37)
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


def test_layer_trace_preserves_reference_execution() -> None:
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
    with torch.inference_mode():
        reference_batch, _ = qwen_forward_with_layer_masks(model, input_ids, plan.masks)
        reference_incremental = qwen_incremental_with_layer_masks(model, input_ids, plan.masks)
        traced_batch, _, batch_trace = _forward_trace(model, input_ids, plan.masks)
        traced_incremental, _, incremental_trace, cache = _incremental_trace(
            model, input_ids, plan.masks
        )
    torch.testing.assert_close(traced_batch, reference_batch)
    torch.testing.assert_close(traced_incremental, reference_incremental)
    torch.testing.assert_close(batch_trace[-1], reference_batch)
    torch.testing.assert_close(incremental_trace[-1], reference_incremental)
    assert batch_trace.shape == (5, 1, 24, 32)
    assert cache["first_length_mismatch"] is None
    assert cache["all_layers_have_full_length"]


def test_trace_summary_locates_earliest_token_then_layer() -> None:
    batched = torch.zeros((3, 1, 4, 2))
    incremental = batched.clone()
    incremental[1, 0, 2, 1] = 0.02
    incremental[2, 0, 1, 0] = 0.2
    summary = _trace_summary(
        batched,
        incremental,
        names=["embedding", "layer_00_output", "final_norm"],
        thresholds=[0.002, 0.1, 1.0],
        probe_positions=[1, 2],
        top_count=2,
    )
    assert summary["earliest_global_above"]["0.002"] == {
        "token": 1,
        "state": "final_norm",
        "feature": 0,
        "max_absolute_error": torch.tensor(0.2).item(),
    }
    assert summary["earliest_global_above"]["0.1"]["token"] == 1
    assert summary["earliest_global_above"]["1.0"] is None
    assert summary["states"]["layer_00_output"]["earliest_token_above"]["0.002"] == 2


def test_mask_structure_proves_only_final_layer_one_retrieval() -> None:
    common = {
        "length": 16,
        "n_layers": 3,
        "rolling_tokens": 4,
        "global_prefix_end": 2,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    rolling = rolling_mask_plan(**common)
    oracle = causal_oracle_mask_plan(
        **common,
        selected_range=(4, 7),
        retrieval_layer=1,
    )
    result = _mask_structure(
        rolling.masks,
        oracle.masks,
        retrieval_layer=1,
        selected_range=(4, 7),
    )
    assert result["passed"]
    assert result["rolling_future_visible_cells"] == 0
    assert result["causal_oracle_future_visible_cells"] == 0
    assert result["oracle_difference_cells"] == 3


def test_diagnostic_preserves_failed_artifact_and_has_no_gate_pass_decision() -> None:
    diagnostic = json.loads(Path("configs/stage_b_numerical_diagnostic.json").read_text())
    original = diagnostic["original_failed_gate"]
    assert _sha256(Path(original["path"])) == original["sha256"]
    failed = json.loads(Path(original["path"]).read_text())
    assert failed["passed"] is False
    assert failed["source_identity"]["git_revision"] == original["source_revision"]
    assert diagnostic["decision"] == "diagnostic_only_stage_b_remains_blocked"
    assert [condition["name"] for condition in diagnostic["conditions"]] == [
        "bf16_sdpa_default",
        "bf16_sdpa_math",
        "fp32_sdpa_math",
        "fp32_eager",
    ]


def test_memory_bounded_retry_preserves_partial_and_restarts_every_condition() -> None:
    diagnostic = json.loads(Path("configs/stage_b_numerical_diagnostic_v2.json").read_text())
    partial = diagnostic["supersedes_partial_diagnostic"]
    assert _sha256(Path(partial["path"])) == partial["sha256"]
    result = json.loads(Path(partial["path"]).read_text())
    assert result["complete"] is False
    assert list(result["conditions"]) == partial["completed_conditions"]
    assert diagnostic["execution"]["trace_storage"] == ("batched_cpu_incremental_gpu_then_cpu")
    assert diagnostic["output"].endswith("stage_b_numerical_diagnostic_v2.json")
    assert [condition["name"] for condition in diagnostic["conditions"]] == [
        "bf16_sdpa_default",
        "bf16_sdpa_math",
        "fp32_sdpa_math",
        "fp32_eager",
    ]
