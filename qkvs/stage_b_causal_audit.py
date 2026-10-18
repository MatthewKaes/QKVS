from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from qkvs.causal_reference import (
    causal_oracle_mask_plan,
    qwen_forward_with_layer_masks,
    qwen_incremental_with_layer_masks,
    rolling_mask_plan,
    stage_a_oracle_mask_plan,
)
from qkvs.runtime import load_json, write_json


def _audit_mode(
    model: Any, input_ids: torch.Tensor, masks: tuple[torch.Tensor, ...]
) -> dict[str, Any]:
    with torch.inference_mode():
        batched, _ = qwen_forward_with_layer_masks(model, input_ids, masks)
        incremental = qwen_incremental_with_layer_masks(model, input_ids, masks)
        batched_logits = model.lm_head(batched[:, -1]).float()
        incremental_logits = model.lm_head(incremental[:, -1]).float()
    hidden_error = (batched.float() - incremental.float()).abs()
    logit_error = (batched_logits - incremental_logits).abs()
    last_hidden_cosine = torch.nn.functional.cosine_similarity(
        batched[:, -1].float(), incremental[:, -1].float()
    )
    return {
        "hidden_max_absolute_error": float(hidden_error.max().item()),
        "hidden_mean_absolute_error": float(hidden_error.mean().item()),
        "logit_max_absolute_error": float(logit_error.max().item()),
        "logit_mean_absolute_error": float(logit_error.mean().item()),
        "batched_top1_token": int(batched_logits.argmax().item()),
        "incremental_top1_token": int(incremental_logits.argmax().item()),
        "last_hidden_cosine": float(last_hidden_cosine.item()),
    }


def run_causal_audit(
    specification: dict[str, Any],
    *,
    sequence_tokens: int = 96,
) -> dict[str, Any]:
    model_specification = specification["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_specification["id"],
        revision=model_specification["revision"],
        trust_remote_code=False,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_specification["id"],
        revision=model_specification["revision"],
        trust_remote_code=False,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    text = (
        "<|im_start|>system\nStrict causal execution audit.<|im_end|>\n"
        "<|im_start|>user\nHistorical blocks must never receive information from a future "
        "question. The final token may retrieve one committed old block at layer one.\n"
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    while len(token_ids) < sequence_tokens:
        token_ids.extend(token_ids)
    input_ids = torch.tensor(token_ids[:sequence_tokens], dtype=torch.long)[None]
    device = torch.device("cpu")
    common = {
        "length": sequence_tokens,
        "n_layers": model.config.num_hidden_layers,
        "rolling_tokens": 32,
        "global_prefix_end": 8,
        "dtype": torch.float32,
        "device": device,
    }
    plans = {
        "rolling": rolling_mask_plan(**common),
        "causal_oracle": causal_oracle_mask_plan(
            **common,
            selected_range=(16, 24),
            retrieval_layer=1,
        ),
        "stage_a_oracle": stage_a_oracle_mask_plan(
            **common,
            query_start=80,
            selected_range=(16, 24),
        ),
    }
    modes = {name: _audit_mode(model, input_ids, plan.masks) for name, plan in plans.items()}
    thresholds = {
        "hidden_max_absolute_error": 0.002,
        "logit_max_absolute_error": 1e-4,
        "minimum_last_hidden_cosine": 0.999999,
        "require_top1_match": True,
    }
    passed = all(
        mode["hidden_max_absolute_error"] <= thresholds["hidden_max_absolute_error"]
        and mode["logit_max_absolute_error"] <= thresholds["logit_max_absolute_error"]
        and mode["last_hidden_cosine"] >= thresholds["minimum_last_hidden_cosine"]
        and mode["batched_top1_token"] == mode["incremental_top1_token"]
        for mode in modes.values()
    )
    return {
        "schema_version": 1,
        "test": "stage_b_strict_causal_batch_incremental_equivalence",
        "model": model_specification,
        "device": "cpu",
        "dtype": "float32",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sequence_tokens": sequence_tokens,
        "rolling_tokens": common["rolling_tokens"],
        "global_prefix_end": common["global_prefix_end"],
        "selected_range": [16, 24],
        "decision_position": sequence_tokens - 1,
        "retrieval_layer": 1,
        "thresholds": thresholds,
        "modes": modes,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict Stage B causal execution")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-tokens", type=int, default=96)
    args = parser.parse_args()
    result = run_causal_audit(
        load_json(args.config),
        sequence_tokens=args.sequence_tokens,
    )
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
