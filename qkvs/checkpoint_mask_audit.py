from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
import transformers
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from qkvs.runtime import load_json, write_json
from qkvs.utility_gate import build_attention_mask


def run_mask_equivalence(
    specification: dict[str, Any],
    *,
    sequence_tokens: int = 128,
    device_name: str = "cpu",
) -> dict[str, Any]:
    device = torch.device(device_name)
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
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
        dtype=model_dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    text = (
        "<|im_start|>system\nMask equivalence audit.<|im_end|>\n"
        "<|im_start|>user\nThe frozen checkpoint must produce the same hidden state under "
        "its built-in causal mask and an explicit dense causal mask.\n"
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    while len(token_ids) < sequence_tokens:
        token_ids.extend(token_ids)
    input_ids = torch.tensor(token_ids[:sequence_tokens], dtype=torch.long, device=device)[None]
    explicit_mask = build_attention_mask(
        sequence_tokens,
        dtype=model_dtype,
        device=device,
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    hidden_error = (builtin_hidden.float() - explicit_hidden.float()).abs()
    logit_error = (builtin_logits.float() - explicit_logits.float()).abs()
    hidden_cosine = F.cosine_similarity(
        builtin_hidden[:, -1].float(), explicit_hidden[:, -1].float()
    )
    thresholds = (
        {
            "hidden_max_absolute_error": 0.125,
            "logit_max_absolute_error": 0.5,
            "minimum_last_hidden_cosine": 0.999,
            "require_top1_match": True,
        }
        if device.type == "cuda"
        else {
            "hidden_max_absolute_error": 1e-5,
            "logit_max_absolute_error": 1e-4,
            "minimum_last_hidden_cosine": 0.999999,
            "require_top1_match": True,
        }
    )
    top1_match = int(builtin_logits.argmax().item()) == int(explicit_logits.argmax().item())
    all_finite = bool(
        torch.isfinite(builtin_hidden).all()
        and torch.isfinite(explicit_hidden).all()
        and torch.isfinite(builtin_logits).all()
        and torch.isfinite(explicit_logits).all()
    )
    measurements = {
        "hidden_max_absolute_error": float(hidden_error.max().item()),
        "hidden_mean_absolute_error": float(hidden_error.mean().item()),
        "logit_max_absolute_error": float(logit_error.max().item()),
        "logit_mean_absolute_error": float(logit_error.mean().item()),
        "last_hidden_cosine": float(hidden_cosine.item()),
        "builtin_top1_token": int(builtin_logits.argmax().item()),
        "explicit_top1_token": int(explicit_logits.argmax().item()),
        "top1_match": top1_match,
        "all_finite": all_finite,
    }
    passed = (
        all_finite
        and measurements["hidden_max_absolute_error"] <= thresholds["hidden_max_absolute_error"]
        and measurements["logit_max_absolute_error"] <= thresholds["logit_max_absolute_error"]
        and measurements["last_hidden_cosine"] >= thresholds["minimum_last_hidden_cosine"]
        and (top1_match or not thresholds["require_top1_match"])
    )
    return {
        "schema_version": 1,
        "test": "actual_checkpoint_explicit_dense_mask_matches_builtin_causal_mask",
        "model": model_specification,
        "device": device_name,
        "dtype": str(model_dtype).removeprefix("torch."),
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = run_mask_equivalence(
        load_json(args.config),
        sequence_tokens=args.sequence_tokens,
        device_name=args.device,
    )
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
