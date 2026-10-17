from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests
import torch
import transformers
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from qkvs.runtime import environment, load_json, write_json


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> str: ...


@dataclass(frozen=True)
class SquadRecord:
    identifier: str
    title: str
    context: str
    question: str
    answer: str
    answer_start: int


@dataclass(frozen=True)
class UtilityExample:
    index: int
    input_ids: tuple[int, ...]
    global_prefix_end: int
    query_start: int
    block_ranges: tuple[tuple[int, int], ...]
    eligible_blocks: tuple[int, ...]
    target_block: int
    random_block: int
    recency_block: int
    label_token_ids: tuple[int, ...]
    gold_label: int
    squad_id: str
    question: str
    options: tuple[str, ...]


SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
SQUAD_SHA256 = "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.exists() and _sha256(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                output.write(chunk)
    actual_sha256 = _sha256(temporary)
    if actual_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"dataset checksum mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    temporary.replace(destination)


def _squad_records(payload: dict[str, Any]) -> list[SquadRecord]:
    records: list[SquadRecord] = []
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            for question in paragraph["qas"]:
                answer_text = Counter(
                    _normalize_answer(answer["text"]) for answer in question["answers"]
                ).most_common(1)[0][0]
                answer = next(
                    answer
                    for answer in question["answers"]
                    if _normalize_answer(answer["text"]) == answer_text
                )
                records.append(
                    SquadRecord(
                        identifier=question["id"],
                        title=article["title"],
                        context=paragraph["context"],
                        question=question["question"],
                        answer=answer["text"].strip(),
                        answer_start=answer["answer_start"],
                    )
                )
    return records


def _normalize_answer(answer: str) -> str:
    return " ".join(answer.casefold().split())


def _fixed_block(
    tokenizer: Tokenizer,
    record: SquadRecord,
    block_tokens: int,
) -> tuple[int, ...] | None:
    prefix = f"\n\nDocument ({record.title}):\n"
    answer_end = record.answer_start + len(record.answer)
    through_answer = tokenizer.encode(
        prefix + record.context[:answer_end], add_special_tokens=False
    )
    if len(through_answer) > block_tokens:
        return None
    token_ids = tokenizer.encode(prefix + record.context + "\n", add_special_tokens=False)
    if len(token_ids) < block_tokens:
        padding = tokenizer.encode(" End of document.\n", add_special_tokens=False)
        while len(token_ids) < block_tokens:
            token_ids.extend(padding)
    return tuple(token_ids[:block_tokens])


def _usable_records(
    tokenizer: Tokenizer,
    records: Iterable[SquadRecord],
    block_tokens: int,
) -> list[tuple[SquadRecord, tuple[int, ...]]]:
    usable: list[tuple[SquadRecord, tuple[int, ...]]] = []
    seen_contexts: set[str] = set()
    for record in records:
        normalized = _normalize_answer(record.answer)
        if not (3 <= len(record.answer) <= 48):
            continue
        if not normalized or normalized in {"yes", "no"} or "\n" in record.answer:
            continue
        if len(record.answer.split()) > 8 or record.context in seen_contexts:
            continue
        block = _fixed_block(tokenizer, record, block_tokens)
        if block is None:
            continue
        seen_contexts.add(record.context)
        usable.append((record, block))
    return usable


def _nonleaking_distractors(
    target: SquadRecord,
    pool: Sequence[tuple[SquadRecord, tuple[int, ...]]],
    start: int,
    count: int,
) -> tuple[list[tuple[SquadRecord, tuple[int, ...]]], int]:
    selected: list[tuple[SquadRecord, tuple[int, ...]]] = []
    forbidden_answers = {_normalize_answer(target.answer)}
    for offset in range(len(pool)):
        record, block = pool[(start + offset) % len(pool)]
        answer = _normalize_answer(record.answer)
        if record.context == target.context or answer in forbidden_answers:
            continue
        if _normalize_answer(target.answer) in record.context.casefold():
            continue
        if any(answer in chosen.context.casefold() for chosen, _ in selected):
            continue
        if any(
            _normalize_answer(chosen.answer) in record.context.casefold() for chosen, _ in selected
        ):
            continue
        selected.append((record, block))
        forbidden_answers.add(answer)
        if len(selected) == count:
            return selected, offset + 1
    raise ValueError("could not construct enough non-leaking distractors")


def _label_token_ids(tokenizer: Tokenizer) -> tuple[int, ...]:
    token_ids: list[int] = []
    for label in "ABCD":
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"expected a one-token answer label for {label!r}, got {encoded}")
        token_ids.append(encoded[0])
    if len(set(token_ids)) != 4:
        raise ValueError("answer labels must map to distinct tokens")
    return tuple(token_ids)


def build_examples(
    tokenizer: Tokenizer,
    squad_payload: dict[str, Any],
    *,
    count: int,
    seed: int,
    n_history_blocks: int,
    block_tokens: int,
    rolling_tokens: int,
    target_position_count: int,
) -> list[UtilityExample]:
    if n_history_blocks < 6:
        raise ValueError("at least six history blocks are required")
    if rolling_tokens < block_tokens:
        raise ValueError("rolling_tokens must cover at least one block")
    label_token_ids = _label_token_ids(tokenizer)
    pool = _usable_records(tokenizer, _squad_records(squad_payload), block_tokens)
    if len(pool) < count * n_history_blocks:
        raise ValueError("not enough usable SQuAD records")

    rng = random.Random(seed)
    rng.shuffle(pool)
    system_ids = tokenizer.encode(
        "<|im_start|>system\nAnswer the user's multiple-choice reading question using the "
        "documents. Reply with only A, B, C, or D.<|im_end|>\n<|im_start|>user\n",
        add_special_tokens=False,
    )
    global_prefix_end = len(system_ids)
    examples: list[UtilityExample] = []
    pool_cursor = 0
    for index in range(count):
        target, target_ids = pool[pool_cursor]
        pool_cursor += 1
        distractors, consumed = _nonleaking_distractors(
            target,
            pool,
            start=pool_cursor,
            count=n_history_blocks - 1,
        )
        pool_cursor += consumed

        target_block = index % target_position_count
        blocks = list(distractors)
        blocks.insert(target_block, (target, target_ids))
        block_ranges: list[tuple[int, int]] = []
        input_ids = list(system_ids)
        for _, block in blocks:
            start = len(input_ids)
            input_ids.extend(block)
            block_ranges.append((start, len(input_ids)))
        query_start = len(input_ids)

        option_records = [target, *(record for record, _ in distractors[:3])]
        gold_label = index % 4
        correct = option_records.pop(0)
        option_records.insert(gold_label, correct)
        options = tuple(record.answer for record in option_records)
        query = (
            "\n\nUse the documents above to answer the question.\n"
            f"Question: {target.question}\n"
            + "\n".join(f"{label}. {answer}" for label, answer in zip("ABCD", options, strict=True))
            + "\nAnswer:<|im_end|>\n<|im_start|>assistant\n"
        )
        input_ids.extend(tokenizer.encode(query, add_special_tokens=False))

        history_cutoff = query_start - rolling_tokens
        eligible_blocks = tuple(
            block_index
            for block_index, (_, end) in enumerate(block_ranges)
            if end <= history_cutoff
        )
        if target_block not in eligible_blocks:
            raise ValueError("target block leaked into the rolling window")
        non_target_blocks = [value for value in eligible_blocks if value != target_block]
        random_block = non_target_blocks[rng.randrange(len(non_target_blocks))]
        recency_block = eligible_blocks[-1]
        examples.append(
            UtilityExample(
                index=index,
                input_ids=tuple(input_ids),
                global_prefix_end=global_prefix_end,
                query_start=query_start,
                block_ranges=tuple(block_ranges),
                eligible_blocks=eligible_blocks,
                target_block=target_block,
                random_block=random_block,
                recency_block=recency_block,
                label_token_ids=label_token_ids,
                gold_label=gold_label,
                squad_id=target.identifier,
                question=target.question,
                options=options,
            )
        )
    return examples


def build_attention_mask(
    length: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    rolling_tokens: int | None = None,
    query_start: int | None = None,
    global_ranges: Sequence[tuple[int, int]] = (),
    selected_ranges: Sequence[tuple[int, int]] = (),
) -> Tensor:
    positions = torch.arange(length, device=device)
    query_positions = positions[:, None]
    key_positions = positions[None, :]
    allowed = key_positions <= query_positions
    if rolling_tokens is not None:
        allowed &= key_positions > query_positions - rolling_tokens
    for start, end in global_ranges:
        allowed |= (
            (key_positions >= start) & (key_positions < end) & (key_positions <= query_positions)
        )
    if selected_ranges:
        if query_start is None:
            raise ValueError("query_start is required for historical selections")
        retrieval_queries = query_positions >= query_start
        for start, end in selected_ranges:
            allowed |= retrieval_queries & (key_positions >= start) & (key_positions < end)
    minimum = torch.finfo(dtype).min
    mask = torch.full((length, length), minimum, dtype=dtype, device=device)
    mask.masked_fill_(allowed, 0.0)
    return mask[None, None]


def _score(
    model: torch.nn.Module,
    example: UtilityExample,
    *,
    device: torch.device,
    model_dtype: torch.dtype,
    rolling_tokens: int | None,
    selected_blocks: Sequence[int] = (),
    global_prefix_attention: bool = False,
) -> dict[str, Any]:
    input_ids = torch.tensor(example.input_ids, dtype=torch.long, device=device)[None]
    attention_mask = None
    if rolling_tokens is not None:
        selected_ranges = [example.block_ranges[index] for index in selected_blocks]
        global_ranges = [(0, example.global_prefix_end)] if global_prefix_attention else []
        attention_mask = build_attention_mask(
            input_ids.shape[1],
            dtype=model_dtype,
            device=device,
            rolling_tokens=rolling_tokens,
            query_start=example.query_start,
            global_ranges=global_ranges,
            selected_ranges=selected_ranges,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    logits = output.logits[0, -1, list(example.label_token_ids)].float()
    log_probabilities = logits.log_softmax(dim=-1)
    loss = -log_probabilities[example.gold_label]
    prediction = int(logits.argmax().item())
    return {
        "loss": float(loss.item()),
        "prediction": prediction,
        "correct": prediction == example.gold_label,
        "label_log_probabilities": [float(value) for value in log_probabilities.tolist()],
        "elapsed_seconds": elapsed,
    }


def evaluate_example(
    model: torch.nn.Module,
    example: UtilityExample,
    *,
    device: torch.device,
    model_dtype: torch.dtype,
    rolling_tokens: int,
    global_prefix_attention: bool,
) -> dict[str, Any]:
    modes = {
        "full": _score(
            model,
            example,
            device=device,
            model_dtype=model_dtype,
            rolling_tokens=None,
        ),
        "rolling": _score(
            model,
            example,
            device=device,
            model_dtype=model_dtype,
            rolling_tokens=rolling_tokens,
            global_prefix_attention=global_prefix_attention,
        ),
        "random": _score(
            model,
            example,
            device=device,
            model_dtype=model_dtype,
            rolling_tokens=rolling_tokens,
            global_prefix_attention=global_prefix_attention,
            selected_blocks=[example.random_block],
        ),
        "recency": _score(
            model,
            example,
            device=device,
            model_dtype=model_dtype,
            rolling_tokens=rolling_tokens,
            global_prefix_attention=global_prefix_attention,
            selected_blocks=[example.recency_block],
        ),
        "annotated_oracle": _score(
            model,
            example,
            device=device,
            model_dtype=model_dtype,
            rolling_tokens=rolling_tokens,
            global_prefix_attention=global_prefix_attention,
            selected_blocks=[example.target_block],
        ),
    }
    measured_blocks = {
        example.target_block: modes["annotated_oracle"],
        example.random_block: modes["random"],
        example.recency_block: modes["recency"],
    }
    block_scores: dict[str, dict[str, Any]] = {}
    for block in example.eligible_blocks:
        if block not in measured_blocks:
            measured_blocks[block] = _score(
                model,
                example,
                device=device,
                model_dtype=model_dtype,
                rolling_tokens=rolling_tokens,
                global_prefix_attention=global_prefix_attention,
                selected_blocks=[block],
            )
        block_scores[str(block)] = measured_blocks[block]
    utility_block = min(block_scores, key=lambda key: block_scores[key]["loss"])
    modes["utility_oracle"] = dict(block_scores[utility_block])
    modes["utility_oracle"]["selected_block"] = int(utility_block)
    rolling_loss = modes["rolling"]["loss"]
    for mode in modes.values():
        mode["utility_nats"] = rolling_loss - mode["loss"]

    token_digest = hashlib.sha256(
        b"".join(token_id.to_bytes(4, "little") for token_id in example.input_ids)
    ).hexdigest()
    return {
        "index": example.index,
        "squad_id": example.squad_id,
        "question": example.question,
        "options": list(example.options),
        "gold_label": example.gold_label,
        "target_block": example.target_block,
        "random_block": example.random_block,
        "recency_block": example.recency_block,
        "eligible_blocks": list(example.eligible_blocks),
        "global_prefix_tokens": example.global_prefix_end,
        "query_start": example.query_start,
        "sequence_tokens": len(example.input_ids),
        "input_token_sha256": token_digest,
        "modes": modes,
        "block_scores": block_scores,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = 2000,
) -> list[float]:
    rng = random.Random(seed)
    means = sorted(
        _mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples)
    )
    return [means[int(0.025 * samples)], means[int(0.975 * samples)]]


def summarize(
    raw_examples: Sequence[dict[str, Any]],
    *,
    configured_examples: int,
    seed: int,
    gate: dict[str, float],
) -> dict[str, Any]:
    mode_names = tuple(raw_examples[0]["modes"]) if raw_examples else ()
    modes: dict[str, Any] = {}
    for mode_name in mode_names:
        mode_values = [example["modes"][mode_name] for example in raw_examples]
        utilities = [float(value["utility_nats"]) for value in mode_values]
        modes[mode_name] = {
            "mean_loss": _mean([float(value["loss"]) for value in mode_values]),
            "accuracy": _mean([float(value["correct"]) for value in mode_values]),
            "mean_utility_nats": _mean(utilities),
            "utility_95pct_bootstrap": _bootstrap_mean_interval(utilities, seed=seed + len(modes)),
        }
    complete = len(raw_examples) == configured_examples
    gate_results: dict[str, bool] = {}
    if complete:
        full_delta = [
            float(example["modes"]["rolling"]["loss"]) - float(example["modes"]["full"]["loss"])
            for example in raw_examples
        ]
        annotated_delta = [
            float(example["modes"]["rolling"]["loss"])
            - float(example["modes"]["annotated_oracle"]["loss"])
            for example in raw_examples
        ]
        annotated_vs_random = [
            float(example["modes"]["random"]["loss"])
            - float(example["modes"]["annotated_oracle"]["loss"])
            for example in raw_examples
        ]
        annotated_vs_recency = [
            float(example["modes"]["recency"]["loss"])
            - float(example["modes"]["annotated_oracle"]["loss"])
            for example in raw_examples
        ]
        comparisons = {
            "full_minus_rolling_utility_nats": {
                "mean": _mean(full_delta),
                "bootstrap_95pct": _bootstrap_mean_interval(full_delta, seed=seed + 101),
            },
            "annotated_minus_rolling_utility_nats": {
                "mean": _mean(annotated_delta),
                "bootstrap_95pct": _bootstrap_mean_interval(annotated_delta, seed=seed + 102),
            },
            "annotated_minus_random_utility_nats": {
                "mean": _mean(annotated_vs_random),
                "bootstrap_95pct": _bootstrap_mean_interval(annotated_vs_random, seed=seed + 103),
            },
            "annotated_minus_recency_utility_nats": {
                "mean": _mean(annotated_vs_recency),
                "bootstrap_95pct": _bootstrap_mean_interval(annotated_vs_recency, seed=seed + 104),
            },
        }
        gate_results = {
            "full_accuracy": modes["full"]["accuracy"] >= gate["minimum_full_accuracy"],
            "full_beats_rolling": comparisons["full_minus_rolling_utility_nats"]["bootstrap_95pct"][
                0
            ]
            >= gate["minimum_utility_lower_bound_nats"],
            "annotated_beats_rolling": comparisons["annotated_minus_rolling_utility_nats"][
                "bootstrap_95pct"
            ][0]
            >= gate["minimum_utility_lower_bound_nats"],
            "annotated_beats_random": comparisons["annotated_minus_random_utility_nats"][
                "bootstrap_95pct"
            ][0]
            > 0.0,
            "annotated_beats_recency": comparisons["annotated_minus_recency_utility_nats"][
                "bootstrap_95pct"
            ][0]
            > 0.0,
        }
    else:
        comparisons = {}
    utility_oracle_match = [
        example["modes"]["utility_oracle"]["selected_block"] == example["target_block"]
        for example in raw_examples
    ]
    return {
        "complete": complete,
        "completed_examples": len(raw_examples),
        "configured_examples": configured_examples,
        "modes": modes,
        "comparisons": comparisons,
        "utility_oracle_target_match_rate": (
            _mean([float(value) for value in utility_oracle_match])
            if utility_oracle_match
            else None
        ),
        "gate": {
            "thresholds": gate,
            "checks": gate_results,
            "passed": complete and all(gate_results.values()),
            "decision": (
                "proceed_to_stage_b"
                if complete and all(gate_results.values())
                else "stop_before_stage_b"
                if complete
                else "incomplete"
            ),
        },
    }


def _load_tokenizer(specification: dict[str, Any]) -> Any:
    model_specification = specification["model"]
    return AutoTokenizer.from_pretrained(
        model_specification["id"],
        revision=model_specification["revision"],
        trust_remote_code=False,
    )


def _load_model(specification: dict[str, Any], device: torch.device) -> tuple[Any, torch.dtype]:
    model_specification = specification["model"]
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_specification["id"],
        revision=model_specification["revision"],
        trust_remote_code=False,
        dtype=model_dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, model_dtype


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage A retrieval-utility gate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--max-examples-this-run", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    specification = load_json(args.config)
    if specification["retrieval_budget_blocks"] != 1:
        raise ValueError("the Stage A implementation requires a one-block retrieval budget")
    dataset_specification = specification["dataset"]
    cache_path = Path(dataset_specification["cache_path"])
    _download(dataset_specification["url"], cache_path, dataset_specification["sha256"])
    tokenizer = _load_tokenizer(specification)
    with cache_path.open() as source:
        squad_payload = json.load(source)
    examples = build_examples(
        tokenizer,
        squad_payload,
        count=specification["examples"],
        seed=specification["seed"],
        n_history_blocks=specification["n_history_blocks"],
        block_tokens=specification["block_tokens"],
        rolling_tokens=specification["rolling_tokens"],
        target_position_count=specification["target_position_count"],
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepared_examples": len(examples),
                    "sequence_token_range": [
                        min(len(example.input_ids) for example in examples),
                        max(len(example.input_ids) for example in examples),
                    ],
                    "model": specification["model"],
                    "dataset_sha256": _sha256(cache_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    model, model_dtype = _load_model(specification, torch.device(args.device))

    output_path = Path(args.output or specification["output"])
    if output_path.exists():
        result = load_json(output_path)
        expected = {
            "config": specification,
            "dataset_sha256": _sha256(cache_path),
            "device": args.device,
        }
        actual = {key: result[key] for key in expected}
        if actual != expected:
            raise ValueError("existing output does not match this experiment configuration")
    else:
        result = {
            "schema_version": 1,
            "stage": "A",
            "hypothesis": specification["hypothesis"],
            "config": specification,
            "dataset_sha256": _sha256(cache_path),
            "device": args.device,
            "environment": environment(),
            "software": {
                "requests": requests.__version__,
                "transformers": transformers.__version__,
            },
            "examples": [],
        }

    completed = {int(example["index"]) for example in result["examples"]}
    remaining = [example for example in examples if example.index not in completed]
    if args.max_examples_this_run is not None:
        remaining = remaining[: args.max_examples_this_run]
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for example in remaining:
        measured = evaluate_example(
            model,
            example,
            device=torch.device(args.device),
            model_dtype=model_dtype,
            rolling_tokens=specification["rolling_tokens"],
            global_prefix_attention=specification.get("global_prefix_attention", False),
        )
        result["examples"].append(measured)
        result["examples"].sort(key=lambda value: value["index"])
        result["summary"] = summarize(
            result["examples"],
            configured_examples=specification["examples"],
            seed=specification["seed"],
            gate=specification["gate"],
        )
        if torch.device(args.device).type == "cuda":
            result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write_json(output_path, result)
        summary = result["summary"]
        print(
            f"example={example.index + 1}/{specification['examples']} "
            f"full={measured['modes']['full']['correct']} "
            f"rolling={measured['modes']['rolling']['correct']} "
            f"annotated={measured['modes']['annotated_oracle']['correct']} "
            f"completed={summary['completed_examples']}"
        )

    if not remaining:
        result["summary"] = summarize(
            result["examples"],
            configured_examples=specification["examples"],
            seed=specification["seed"],
            gate=specification["gate"],
        )
        write_json(output_path, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
