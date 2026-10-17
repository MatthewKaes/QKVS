from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from qkvs.runtime import load_json, write_json
from qkvs.utility_gate import build_attention_mask


def run_mask_equivalence(
    specification: dict[str, Any],
    *,
    sequence_tokens: int = 128,
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
        "<|im_start|>system\nMask equivalence audit.<|im_end|>\n"
        "<|im_start|>user\nThe frozen checkpoint must produce the same hidden state under "
        "its built-in causal mask and an explicit dense causal mask.\n"
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    while len(token_ids) < sequence_tokens:
        token_ids.extend(token_ids)
    input_ids = torch.tensor(token_ids[:sequence_tokens], dtype=torch.long)[None]
    explicit_mask = build_attention_mask(
        sequence_tokens,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with torch.inference_mode():
        builtin_hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        explicit_hidden = model.model(
            input_ids=input_ids,
            attention_mask=explicit_mask,
            use_cache=False,
        ).last_hidden_state
        builtin_logits = model.lm_head(builtin_hidden[:, -1])
        explicit_logits = model.lm_head(explicit_hidden[:, -1])

    hidden_error = (builtin_hidden - explicit_hidden).abs()
    logit_error = (builtin_logits - explicit_logits).abs()
    thresholds = {
        "hidden_max_absolute_error": 1e-5,
        "logit_max_absolute_error": 1e-4,
    }
    measurements = {
        "hidden_max_absolute_error": float(hidden_error.max().item()),
        "hidden_mean_absolute_error": float(hidden_error.mean().item()),
        "logit_max_absolute_error": float(logit_error.max().item()),
        "logit_mean_absolute_error": float(logit_error.mean().item()),
    }
    passed = (
        measurements["hidden_max_absolute_error"] <= thresholds["hidden_max_absolute_error"]
        and measurements["logit_max_absolute_error"] <= thresholds["logit_max_absolute_error"]
    )
    return {
        "schema_version": 1,
        "test": "actual_checkpoint_explicit_dense_mask_matches_builtin_causal_mask",
        "model": model_specification,
        "device": "cpu",
        "dtype": "float32",
        "sequence_tokens": sequence_tokens,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "thresholds": thresholds,
        "measurements": measurements,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the real checkpoint's explicit causal mask")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-tokens", type=int, default=128)
    args = parser.parse_args()
    result = run_mask_equivalence(
        load_json(args.config),
        sequence_tokens=args.sequence_tokens,
    )
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
