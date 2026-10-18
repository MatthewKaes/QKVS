from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from qkvs.causal_reference import (
    causal_oracle_mask_plan,
    qwen_forward_with_layer_masks,
    qwen_incremental_with_layer_masks,
    rolling_mask_plan,
)
from qkvs.runtime import load_json, write_json
from qkvs.stage_b_oracle import _source_identity
from qkvs.utility_gate import (
    _download,
    _load_model,
    _load_tokenizer,
    _sha256,
    build_decisive_examples,
)


def _compare_mode(
    model: Any,
    input_ids: torch.Tensor,
    masks: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        batched, _ = qwen_forward_with_layer_masks(model, input_ids, masks)
    torch.cuda.synchronize(device)
    batched_seconds = time.perf_counter() - started

    started = time.perf_counter()
    with torch.inference_mode():
        incremental = qwen_incremental_with_layer_masks(model, input_ids, masks)
    torch.cuda.synchronize(device)
    incremental_seconds = time.perf_counter() - started

    with torch.inference_mode():
        batched_logits = model.lm_head(batched[:, -1]).float()
        incremental_logits = model.lm_head(incremental[:, -1]).float()
    hidden_error = (batched.float() - incremental.float()).abs()
    logit_error = (batched_logits - incremental_logits).abs()
    cosine = F.cosine_similarity(batched[:, -1].float(), incremental[:, -1].float())
    return {
        "hidden_max_absolute_error": float(hidden_error.max().item()),
        "hidden_mean_absolute_error": float(hidden_error.mean().item()),
        "logit_max_absolute_error": float(logit_error.max().item()),
        "logit_mean_absolute_error": float(logit_error.mean().item()),
        "last_hidden_cosine": float(cosine.item()),
        "batched_top1_token": int(batched_logits.argmax().item()),
        "incremental_top1_token": int(incremental_logits.argmax().item()),
        "batched_seconds": batched_seconds,
        "incremental_seconds": incremental_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full-length Stage B causal audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()

    specification = load_json(args.config)
    geometry = specification["geometry"]
    oracle = specification["oracle_parity"]
    audit = specification["full_length_causal_audit"]
    dataset = specification["datasets"]["evaluation"]
    dataset_path = Path(dataset["cache_path"])
    _download(dataset["url"], dataset_path, dataset["sha256"])
    tokenizer = _load_tokenizer(specification)
    with dataset_path.open() as source:
        payload = json.load(source)
    examples = build_decisive_examples(
        tokenizer,
        payload,
        count=oracle["examples"],
        seed=oracle["dataset_seed"],
        n_history_blocks=geometry["history_blocks"],
        candidate_history_blocks=len(geometry["candidate_positions"]),
        block_tokens=geometry["block_tokens"],
        rolling_tokens=geometry["rolling_tokens"],
        rolling_layers=geometry["rolling_layers"],
        target_seed=oracle["target_position_seed"],
        label_seed=oracle["answer_label_seed"],
        baseline_seed=oracle["random_baseline_seed"],
    )
    example = examples[audit["example_index"]]
    identity = _source_identity(args.config)
    if not identity["tracked_files_clean"]:
        raise ValueError("full-length audit source files must be committed before measurement")

    device = torch.device(args.device)
    model, model_dtype = _load_model(specification, device)
    input_ids = torch.tensor(example.input_ids, dtype=torch.long, device=device)[None]
    common = {
        "length": len(example.input_ids),
        "n_layers": model.config.num_hidden_layers,
        "rolling_tokens": geometry["rolling_tokens"],
        "global_prefix_end": example.global_prefix_end,
        "dtype": model_dtype,
        "device": device,
    }
    selected_range = example.block_ranges[example.target_block]
    plans = {
        "rolling": rolling_mask_plan(**common),
        "causal_oracle": causal_oracle_mask_plan(
            **common,
            selected_range=selected_range,
            retrieval_layer=geometry["retrieval_layer"],
        ),
    }
    if set(plans) != set(audit["modes"]):
        raise ValueError("configured audit modes do not match the implemented plans")
    modes = {
        name: _compare_mode(model, input_ids, plans[name].masks, device=device)
        for name in audit["modes"]
    }
    passed = all(
        mode["hidden_max_absolute_error"] <= audit["hidden_max_absolute_error"]
        and mode["logit_max_absolute_error"] <= audit["logit_max_absolute_error"]
        and mode["last_hidden_cosine"] >= audit["minimum_last_hidden_cosine"]
        and (
            mode["batched_top1_token"] == mode["incremental_top1_token"]
            or not audit["require_top1_match"]
        )
        for mode in modes.values()
    )
    token_hash = hashlib.sha256(
        b"".join(token.to_bytes(4, "little") for token in example.input_ids)
    ).hexdigest()
    result = {
        "schema_version": 1,
        "test": "stage_b_full_length_batch_incremental_causal_equivalence",
        "config": specification,
        "source_identity": identity,
        "dataset_sha256": _sha256(dataset_path),
        "device": args.device,
        "dtype": str(model_dtype).removeprefix("torch."),
        "example_index": example.index,
        "squad_id": example.squad_id,
        "sequence_tokens": len(example.input_ids),
        "input_token_sha256": token_hash,
        "decision_position": len(example.input_ids) - 1,
        "retrieval_layer": geometry["retrieval_layer"],
        "selected_range": list(selected_range),
        "thresholds": {
            key: audit[key]
            for key in (
                "hidden_max_absolute_error",
                "logit_max_absolute_error",
                "minimum_last_hidden_cosine",
                "require_top1_match",
            )
        },
        "modes": modes,
        "passed": passed,
    }
    output = Path(args.output or audit["output"])
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
