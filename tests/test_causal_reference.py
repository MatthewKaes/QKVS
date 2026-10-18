from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from qkvs.causal_reference import (
    causal_oracle_mask_plan,
    qwen_forward_with_layer_masks,
    qwen_incremental_with_layer_masks,
    rolling_mask_plan,
    stage_a_oracle_mask_plan,
)


def _model() -> Qwen2ForCausalLM:
    torch.manual_seed(13)
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


def test_causal_oracle_is_available_only_at_the_declared_position_and_layer() -> None:
    plan = causal_oracle_mask_plan(
        length=16,
        n_layers=3,
        rolling_tokens=4,
        global_prefix_end=2,
        selected_range=(4, 7),
        retrieval_layer=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert plan.masks[0][0, 0, 15, 4] < 0
    assert plan.masks[1][0, 0, 15, 4] == 0
    assert plan.masks[2][0, 0, 15, 4] < 0
    assert plan.masks[1][0, 0, 14, 4] < 0


def test_incremental_execution_matches_strict_causal_batch() -> None:
    model = _model()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 24))
    plans = (
        rolling_mask_plan(
            length=24,
            n_layers=3,
            rolling_tokens=8,
            global_prefix_end=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ),
        causal_oracle_mask_plan(
            length=24,
            n_layers=3,
            rolling_tokens=8,
            global_prefix_end=2,
            selected_range=(4, 8),
            retrieval_layer=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ),
        stage_a_oracle_mask_plan(
            length=24,
            n_layers=3,
            rolling_tokens=8,
            global_prefix_end=2,
            query_start=20,
            selected_range=(4, 8),
            dtype=torch.float32,
            device=torch.device("cpu"),
        ),
    )
    for plan in plans:
        with torch.inference_mode():
            batched, _ = qwen_forward_with_layer_masks(model, input_ids, plan.masks)
            incremental = qwen_incremental_with_layer_masks(model, input_ids, plan.masks)
        torch.testing.assert_close(incremental, batched, atol=1e-6, rtol=1e-6)


def test_future_tokens_cannot_change_earlier_hidden_states() -> None:
    model = _model()
    original = torch.randint(0, model.config.vocab_size, (1, 24))
    changed = original.clone()
    changed[:, 18:] = torch.randint(0, model.config.vocab_size, (1, 6))
    plan = stage_a_oracle_mask_plan(
        length=24,
        n_layers=3,
        rolling_tokens=8,
        global_prefix_end=2,
        query_start=20,
        selected_range=(4, 8),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with torch.inference_mode():
        original_hidden, _ = qwen_forward_with_layer_masks(model, original, plan.masks)
        changed_hidden, _ = qwen_forward_with_layer_masks(model, changed, plan.masks)
    torch.testing.assert_close(
        original_hidden[:, :18], changed_hidden[:, :18], atol=1e-6, rtol=1e-6
    )
