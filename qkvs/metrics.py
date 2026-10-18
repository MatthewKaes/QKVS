from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def paired_bootstrap_interval(
    differences: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if not differences:
        return [0.0, 0.0]
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    generator = random.Random(seed)
    estimates = sorted(
        _mean(differences[generator.randrange(len(differences))] for _ in differences)
        for _ in range(samples)
    )
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return [lower, upper]


def summarize_accesses(
    accesses: list[dict[str, Any]],
    *,
    historical_token_budget: int | None = None,
) -> dict[str, Any]:
    historical_occurrences = sum(len(access["historical_indices"]) for access in accesses)
    historical_unique = sum(len(set(access["historical_indices"])) for access in accesses)
    duplicate_occurrences = historical_occurrences - historical_unique
    rolling_occurrences = sum(len(access["rolling_indices"]) for access in accesses)
    rolling_overlap_occurrences = sum(
        sum(index in set(access["rolling_indices"]) for index in access["historical_indices"])
        for access in accesses
    )
    proposal_centers = [
        center
        for access in accesses
        for center, active in zip(
            access["proposal_centers"],
            access.get("proposal_active", (True,) * len(access["proposal_centers"])),
            strict=True,
        )
        if active
    ]
    proposal_spreads = [
        spread
        for access in accesses
        for spread, active in zip(
            access.get("proposal_spreads", ()),
            access.get("proposal_active", (True,) * len(access.get("proposal_spreads", ()))),
            strict=True,
        )
        if active
    ]
    active_proposals = sum(sum(access.get("proposal_active", ())) for access in accesses)
    inactive_proposals = sum(
        len(access.get("proposal_active", ())) - sum(access.get("proposal_active", ()))
        for access in accesses
    )
    normalized_historical_positions = [
        index / access["history_end"]
        for access in accesses
        if access["history_end"] > 0
        for index in access["historical_indices"]
    ]
    support_trials = [
        access
        for access in accesses
        if access["support_position"] is not None
        and access["support_position"] < access["token_position"] + 1
    ]
    support_hits = sum(
        access["support_position"] in access["actual_indices"] for access in support_trials
    )

    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for access in accesses:
        by_layer[access["layer"]].append(access)
    stability: list[float] = []
    responsiveness: list[float] = []
    for layer_accesses in by_layer.values():
        ordered = sorted(layer_accesses, key=lambda access: access["token_position"])
        for previous, current in zip(ordered, ordered[1:], strict=False):
            previous_indices = set(previous["historical_indices"])
            current_indices = set(current["historical_indices"])
            union = previous_indices | current_indices
            stability.append(len(previous_indices & current_indices) / len(union) if union else 1.0)
            if previous["support_position"] != current["support_position"]:
                responsiveness.append(float(previous_indices != current_indices))

    per_layer = {
        str(layer): {
            "attention_steps": len(layer_accesses),
            "historical_token_occurrences": sum(
                len(access["historical_indices"]) for access in layer_accesses
            ),
            "duplicate_historical_occurrences": sum(
                len(access["historical_indices"]) - len(set(access["historical_indices"]))
                for access in layer_accesses
            ),
            "rolling_token_occurrences": sum(
                len(access["rolling_indices"]) for access in layer_accesses
            ),
        }
        for layer, layer_accesses in sorted(by_layer.items())
    }

    summary = {
        "historical_token_occurrences": historical_occurrences,
        "historical_unique_tokens": historical_unique,
        "duplicate_historical_occurrences": duplicate_occurrences,
        "overlap_occurrences": duplicate_occurrences,
        "rolling_overlap_occurrences": rolling_overlap_occurrences,
        "rolling_token_occurrences": rolling_occurrences,
        "actual_token_occurrences": historical_occurrences + rolling_occurrences,
        "per_layer": per_layer,
        "mean_proposal_center": _mean(proposal_centers),
        "mean_proposal_spread": _mean(proposal_spreads),
        "active_proposals": active_proposals,
        "inactive_proposals": inactive_proposals,
        "mean_normalized_historical_position": _mean(normalized_historical_positions),
        "support_coverage": support_hits / len(support_trials) if support_trials else None,
        "support_trials": len(support_trials),
        "temporal_stability": _mean(stability),
        "temporal_responsiveness": _mean(responsiveness),
        "responsiveness_trials": len(responsiveness),
    }
    if historical_token_budget is not None:
        maximum = len(accesses) * historical_token_budget
        summary["historical_budget_utilization"] = (
            historical_occurrences / maximum if maximum else 0.0
        )
    return summary


def summarize_training_alignment(history: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (float(record["soft_utility"]), float(record["hard_utility"]))
        for record in history
        if "soft_utility" in record and "hard_utility" in record
    ]
    if not pairs:
        return {
            "paired_steps": 0,
            "soft_hard_utility_correlation": None,
            "utility_sign_agreement": None,
            "mean_absolute_utility_gap": None,
        }
    soft = [pair[0] for pair in pairs]
    hard = [pair[1] for pair in pairs]
    soft_mean = _mean(soft)
    hard_mean = _mean(hard)
    covariance = sum(
        (soft_value - soft_mean) * (hard_value - hard_mean) for soft_value, hard_value in pairs
    )
    soft_variance = sum((value - soft_mean) ** 2 for value in soft)
    hard_variance = sum((value - hard_mean) ** 2 for value in hard)
    denominator = (soft_variance * hard_variance) ** 0.5
    nonzero_pairs = [pair for pair in pairs if pair[0] != 0.0 and pair[1] != 0.0]
    return {
        "paired_steps": len(pairs),
        "soft_hard_utility_correlation": covariance / denominator if denominator else None,
        "utility_sign_agreement": (
            _mean(float(soft_value * hard_value > 0.0) for soft_value, hard_value in nonzero_pairs)
            if nonzero_pairs
            else None
        ),
        "mean_absolute_utility_gap": _mean(
            abs(soft_value - hard_value) for soft_value, hard_value in pairs
        ),
    }


def summarize_measurements(
    measurements: list[dict[str, Any]],
    *,
    baseline: str,
    bootstrap_samples: int,
    seed: int,
    historical_token_budget: int | None = None,
    paired_comparisons: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for measurement in measurements:
        by_condition[measurement["condition"]].append(measurement)
    conditions: dict[str, Any] = {}
    sequence_losses: dict[str, dict[str, float]] = {}
    for condition, records in sorted(by_condition.items()):
        token_losses = [score["loss"] for record in records for score in record["token_scores"]]
        accesses = [
            access
            for record in records
            for score in record["token_scores"]
            for access in score["accesses"]
        ]
        per_sequence = {
            record["stream_id"]: _mean(score["loss"] for score in record["token_scores"])
            for record in records
        }
        sequence_losses[condition] = per_sequence
        conditions[condition] = {
            "streams": len(records),
            "scored_tokens": len(token_losses),
            "mean_next_token_cross_entropy": _mean(token_losses),
            "accuracy": _mean(
                float(score["prediction"] == score["target"])
                for record in records
                for score in record["token_scores"]
            ),
            "retrieval": summarize_accesses(
                accesses,
                historical_token_budget=historical_token_budget,
            ),
        }

    if baseline not in sequence_losses:
        raise ValueError("configured baseline is absent from completed measurements")
    paired: dict[str, Any] = {}
    baseline_losses = sequence_losses[baseline]
    for condition, losses in sorted(sequence_losses.items()):
        if condition == baseline:
            continue
        common = sorted(set(baseline_losses) & set(losses))
        differences = [baseline_losses[key] - losses[key] for key in common]
        paired[condition] = {
            "paired_streams": len(common),
            "mean_utility_nats": _mean(differences),
            "paired_bootstrap_95_interval_nats": paired_bootstrap_interval(
                differences,
                samples=bootstrap_samples,
                seed=seed,
            ),
            "per_stream_utility_nats": dict(zip(common, differences, strict=True)),
        }
    for comparison in paired_comparisons or []:
        if (
            comparison["baseline"] not in sequence_losses
            or comparison["condition"] not in sequence_losses
        ):
            continue
        comparison_baseline = sequence_losses[comparison["baseline"]]
        comparison_condition = sequence_losses[comparison["condition"]]
        common = sorted(set(comparison_baseline) & set(comparison_condition))
        differences = [comparison_baseline[key] - comparison_condition[key] for key in common]
        paired[comparison["name"]] = {
            "baseline": comparison["baseline"],
            "condition": comparison["condition"],
            "paired_streams": len(common),
            "mean_utility_nats": _mean(differences),
            "paired_bootstrap_95_interval_nats": paired_bootstrap_interval(
                differences,
                samples=bootstrap_samples,
                seed=seed,
            ),
            "per_stream_utility_nats": dict(zip(common, differences, strict=True)),
        }
    return {"conditions": conditions, "paired_utility": paired}
