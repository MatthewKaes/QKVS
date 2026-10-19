from __future__ import annotations

import base64
import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

_SNAPSHOT_OVERLAP_THRESHOLDS = (0.9999, 0.9995, 0.995, 0.95)


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def paired_bootstrap_interval(
    differences: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if len(differences) < 2:
        raise ValueError("paired stream bootstrap requires at least two independent streams")
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


def _paired_comparison_summary(
    baseline_losses: dict[str, float],
    condition_losses: dict[str, float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    common = sorted(set(baseline_losses) & set(condition_losses))
    differences = [baseline_losses[key] - condition_losses[key] for key in common]
    paired_streams = len(common)
    if paired_streams >= 2:
        interval = paired_bootstrap_interval(
            differences,
            samples=bootstrap_samples,
            seed=seed,
        )
        interpretation = "paired_stream_bootstrap"
    else:
        interval = None
        interpretation = "descriptive_single_stream" if paired_streams == 1 else "no_paired_streams"
    return {
        "paired_streams": paired_streams,
        "mean_utility_nats": _mean(differences),
        "paired_bootstrap_95_interval_nats": interval,
        "statistical_interpretation": interpretation,
        "per_stream_utility_nats": dict(zip(common, differences, strict=True)),
    }


def _windows(access: dict[str, Any]) -> list[tuple[int, int]]:
    if "historical_windows_varint_base64" in access:
        encoded = base64.b64decode(access["historical_windows_varint_base64"], validate=True)
        windows = []
        offset = 0

        def decode_varint() -> int:
            nonlocal offset
            value = 0
            shift = 0
            while offset < len(encoded):
                byte = encoded[offset]
                offset += 1
                value |= (byte & 0x7F) << shift
                if not byte & 0x80:
                    return value
                shift += 7
                if shift > 63:
                    break
            raise ValueError("invalid packed historical window")

        while offset < len(encoded):
            start = decode_varint()
            length = decode_varint()
            if length <= 0:
                raise ValueError("packed historical windows must be nonempty")
            windows.append((start, start + length))
        return windows
    if "historical_windows" in access:
        return [tuple(window) for window in access["historical_windows"]]
    indices = access.get("historical_indices", [])
    windows: list[tuple[int, int]] = []
    for index in indices:
        if windows and index == windows[-1][1]:
            windows[-1] = (windows[-1][0], index + 1)
        else:
            windows.append((index, index + 1))
    return windows


def _rolling_range(access: dict[str, Any]) -> tuple[int, int]:
    if "rolling_range" in access:
        return tuple(access["rolling_range"])
    indices = access.get("rolling_indices", [])
    return (indices[0], indices[-1] + 1) if indices else (0, 0)


def _merged(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def _interval_length(windows: list[tuple[int, int]]) -> int:
    return sum(stop - start for start, stop in _merged(windows))


def _intersection_length(first: list[tuple[int, int]], second: list[tuple[int, int]]) -> int:
    left = _merged(first)
    right = _merged(second)
    total = 0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        stop = min(left[left_index][1], right[right_index][1])
        total += max(0, stop - start)
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def _contains(windows: list[tuple[int, int]], position: int) -> bool:
    return any(start <= position < stop for start, stop in windows)


def summarize_accesses(
    accesses: list[dict[str, Any]],
    *,
    historical_token_budget: int | None = None,
) -> dict[str, Any]:
    historical_occurrences = sum(
        access.get(
            "historical_occurrences",
            sum(stop - start for start, stop in _windows(access)),
        )
        for access in accesses
    )
    historical_unique = sum(
        access.get("historical_unique_tokens", _interval_length(_windows(access)))
        for access in accesses
    )
    duplicate_occurrences = historical_occurrences - historical_unique
    rolling_occurrences = sum(stop - start for start, stop in map(_rolling_range, accesses))
    rolling_overlap_occurrences = sum(
        access.get("rolling_overlap_occurrences", 0) for access in accesses
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
    normalized_historical_position_sum = sum(
        (start + stop - 1) * (stop - start) / 2 / access["history_end"]
        for access in accesses
        if access["history_end"] > 0
        for start, stop in _windows(access)
    )
    support_trials = [
        access
        for access in accesses
        if access["support_position"] is not None
        and access["support_position"] < access["token_position"] + 1
    ]
    support_hits = 0
    for access in support_trials:
        support = access["support_position"]
        rolling_start, rolling_stop = _rolling_range(access)
        support_hits += bool(
            rolling_start <= support < rolling_stop or _contains(_windows(access), support)
        )

    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for access in accesses:
        by_layer[access["layer"]].append(access)
    stability: list[float] = []
    responsiveness: list[float] = []
    for layer_accesses in by_layer.values():
        ordered = sorted(layer_accesses, key=lambda access: access["token_position"])
        for previous, current in zip(ordered, ordered[1:], strict=False):
            previous_windows = _windows(previous)
            current_windows = _windows(current)
            intersection = _intersection_length(previous_windows, current_windows)
            union = (
                _interval_length(previous_windows)
                + _interval_length(current_windows)
                - intersection
            )
            stability.append(intersection / union if union else 1.0)
            if previous["support_position"] != current["support_position"]:
                responsiveness.append(float(previous_windows != current_windows))

    per_layer = {
        str(layer): {
            "attention_steps": len(layer_accesses),
            "historical_token_occurrences": sum(
                access.get(
                    "historical_occurrences",
                    sum(stop - start for start, stop in _windows(access)),
                )
                for access in layer_accesses
            ),
            "duplicate_historical_occurrences": sum(
                access.get("duplicate_historical_occurrences", 0) for access in layer_accesses
            ),
            "rolling_token_occurrences": sum(
                stop - start for start, stop in map(_rolling_range, layer_accesses)
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
        "proposal_trace_records": sum(bool(access["proposal_centers"]) for access in accesses),
        "mean_normalized_historical_position": (
            normalized_historical_position_sum / historical_occurrences
            if historical_occurrences
            else 0.0
        ),
        "support_coverage": support_hits / len(support_trials) if support_trials else None,
        "support_trials": len(support_trials),
        "temporal_stability": _mean(stability),
        "temporal_responsiveness": _mean(responsiveness),
        "responsiveness_trials": len(responsiveness),
    }
    unbounded = any(access.get("attention_mode") == "full_qkv" for access in accesses)
    summary["bounded_historical_attention"] = not unbounded
    if historical_token_budget is not None:
        maximum = len(accesses) * historical_token_budget
        summary["historical_budget_utilization"] = (
            None if unbounded else historical_occurrences / maximum if maximum else 0.0
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


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def _tolerance_key(value: float) -> str:
    return format(value, ".12g")


def summarize_snapshot_trial_set(
    candidates: list[dict[str, Any]],
    *,
    fidelity_tolerances: list[float],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("snapshot trial summary requires candidate observations")
    divergences = [float(candidate["kl_divergence_nats"]) for candidate in candidates]
    overlaps = [float(candidate["probability_distribution_overlap"]) for candidate in candidates]
    loss_deltas = [float(candidate["next_token_ce_delta_nats"]) for candidate in candidates]
    realized = [float(candidate["realized_sampled_fraction"]) for candidate in candidates]
    best = min(
        candidates,
        key=lambda candidate: (
            float(candidate["kl_divergence_nats"]),
            float(candidate["realized_sampled_fraction"]),
            int(candidate["trial_id"]),
        ),
    )
    histogram_width = len(candidates[0]["sampled_position_histogram"])
    if any(
        len(candidate["sampled_position_histogram"]) != histogram_width for candidate in candidates
    ):
        raise ValueError("snapshot candidate histograms do not align")
    return {
        "candidate_count": len(candidates),
        "kl_divergence_nats": {
            "minimum": min(divergences),
            "median": _quantile(divergences, 0.5),
            "mean": _mean(divergences),
            "maximum": max(divergences),
        },
        "probability_distribution_overlap": {
            "minimum": min(overlaps),
            "median": _quantile(overlaps, 0.5),
            "mean": _mean(overlaps),
            "maximum": max(overlaps),
        },
        "next_token_ce_delta_nats": {
            "minimum": min(loss_deltas),
            "median": _quantile(loss_deltas, 0.5),
            "mean": _mean(loss_deltas),
            "maximum": max(loss_deltas),
        },
        "realized_sampled_fraction": {
            "minimum": min(realized),
            "mean": _mean(realized),
            "maximum": max(realized),
        },
        "top_one_agreement_fraction": _mean(
            float(candidate["top_one_agreement"]) for candidate in candidates
        ),
        "successful_candidate_fraction": {
            _tolerance_key(tolerance): _mean(
                float(candidate["kl_divergence_nats"] <= tolerance) for candidate in candidates
            )
            for tolerance in fidelity_tolerances
        },
        "overlap_successful_candidate_fraction": {
            _tolerance_key(threshold): _mean(
                float(candidate["probability_distribution_overlap"] >= threshold)
                for candidate in candidates
            )
            for threshold in _SNAPSHOT_OVERLAP_THRESHOLDS
        },
        "mean_selected_token_occurrences": _mean(
            candidate["selected_token_occurrences"] for candidate in candidates
        ),
        "mean_aggregate_block_count": _mean(
            candidate["aggregate_block_count"] for candidate in candidates
        ),
        "sampled_position_histogram": [
            sum(candidate["sampled_position_histogram"][index] for candidate in candidates)
            for index in range(histogram_width)
        ],
        "best_candidate": {
            key: best[key]
            for key in (
                "trial_id",
                "candidate_seed",
                "requested_sampled_fraction",
                "realized_sampled_fraction",
                "kl_divergence_nats",
                "probability_distribution_overlap",
                "next_token_ce_delta_nats",
                "top_one_agreement",
            )
        },
    }


def summarize_snapshot_frontiers(
    trial_sets: list[dict[str, Any]],
    *,
    fidelity_tolerances: list[float],
) -> dict[str, Any]:
    positions = sorted({int(trial_set["position"]) for trial_set in trial_sets})
    frontiers: dict[str, Any] = {}
    for position in positions:
        candidates = [
            candidate
            for trial_set in trial_sets
            if int(trial_set["position"]) == position
            for candidate in trial_set["candidates"]
        ]
        position_frontier: dict[str, Any] = {}
        for tolerance in fidelity_tolerances:
            successful = [
                candidate
                for candidate in candidates
                if float(candidate["kl_divergence_nats"]) <= tolerance
            ]
            key = _tolerance_key(tolerance)
            if not successful:
                position_frontier[key] = {
                    "status": "not_found",
                    "smallest_discovered_fraction": None,
                }
                continue
            discovered = min(
                successful,
                key=lambda candidate: (
                    float(candidate["realized_sampled_fraction"]),
                    float(candidate["kl_divergence_nats"]),
                    int(candidate["trial_id"]),
                ),
            )
            position_frontier[key] = {
                "status": "found",
                "smallest_discovered_fraction": discovered["realized_sampled_fraction"],
                "requested_sampled_fraction": discovered["requested_sampled_fraction"],
                "trial_id": discovered["trial_id"],
                "candidate_seed": discovered["candidate_seed"],
                "kl_divergence_nats": discovered["kl_divergence_nats"],
            }
        frontiers[str(position)] = position_frontier
    return {
        "interpretation": "minimum_discovered_random_sample_not_theoretical_minimum",
        "frontiers_by_position": frontiers,
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
    post_sequence_losses: dict[str, dict[str, float]] = {}
    region_sequence_losses: dict[str, dict[str, dict[str, float]]] = {}
    for condition, records in sorted(by_condition.items()):
        token_losses = [score["loss"] for record in records for score in record["token_scores"]]
        post_scores = [
            score
            for record in records
            for score in record["token_scores"]
            if score.get("phase") in {"post_transition", "qkvs"}
        ]
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
        post_sequence_losses[condition] = {
            record["stream_id"]: _mean(
                score["loss"]
                for score in record["token_scores"]
                if score.get("phase") in {"post_transition", "qkvs"}
            )
            for record in records
        }
        region_losses: dict[str, list[float]] = {"early": [], "middle": [], "late": []}
        region_per_stream: dict[str, dict[str, float]] = {
            "early": {},
            "middle": {},
            "late": {},
        }
        region_names = ("early", "middle", "late")
        for record in records:
            continuation = [
                score
                for score in record["token_scores"]
                if score.get("phase") in {"post_transition", "qkvs"}
            ]
            for index, score in enumerate(continuation):
                region = region_names[min(2, 3 * index // max(1, len(continuation)))]
                region_losses[region].append(score["loss"])
            for region in region_names:
                region_per_stream[region][record["stream_id"]] = _mean(
                    score["loss"]
                    for index, score in enumerate(continuation)
                    if region_names[min(2, 3 * index // max(1, len(continuation)))] == region
                )
        region_sequence_losses[condition] = region_per_stream
        performance = [record["performance"] for record in records if "performance" in record]
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
            "post_transition": {
                "scored_tokens": len(post_scores),
                "mean_next_token_cross_entropy": _mean(score["loss"] for score in post_scores),
                "accuracy": _mean(
                    float(score["prediction"] == score["target"]) for score in post_scores
                ),
                "loss_by_context_region": {
                    name: _mean(region_losses[name]) for name in region_names
                },
            },
            "measured_execution": {
                "streams": len(performance),
                "mean_wall_seconds_including_snapshot_restore": _mean(
                    item["wall_seconds_including_snapshot_restore"] for item in performance
                ),
                "mean_post_transition_tokens_per_second": _mean(
                    item["post_transition_tokens_per_second"] for item in performance
                ),
                "max_peak_accelerator_bytes": max(
                    (item.get("peak_accelerator_bytes") or 0 for item in performance),
                    default=0,
                )
                or None,
            },
        }

    if baseline not in sequence_losses:
        raise ValueError("configured baseline is absent from completed measurements")

    def comparisons(losses_by_condition: dict[str, dict[str, float]]) -> dict[str, Any]:
        paired: dict[str, Any] = {}
        baseline_losses = losses_by_condition[baseline]
        for condition, losses in sorted(losses_by_condition.items()):
            if condition == baseline:
                continue
            paired[condition] = _paired_comparison_summary(
                baseline_losses,
                losses,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        return paired

    paired = comparisons(sequence_losses)
    post_paired = comparisons(post_sequence_losses)
    region_paired = {
        region: comparisons(
            {condition: losses[region] for condition, losses in region_sequence_losses.items()}
        )
        for region in ("early", "middle", "late")
    }
    for comparison in paired_comparisons or []:
        if (
            comparison["baseline"] not in sequence_losses
            or comparison["condition"] not in sequence_losses
        ):
            continue

        def add_comparison(
            destination: dict[str, Any],
            losses_by_condition: dict[str, dict[str, float]],
            comparison: dict[str, str] = comparison,
        ) -> None:
            comparison_baseline = losses_by_condition[comparison["baseline"]]
            comparison_condition = losses_by_condition[comparison["condition"]]
            destination[comparison["name"]] = {
                "baseline": comparison["baseline"],
                "condition": comparison["condition"],
                **_paired_comparison_summary(
                    comparison_baseline,
                    comparison_condition,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                ),
            }

        add_comparison(paired, sequence_losses)
        add_comparison(post_paired, post_sequence_losses)
        for region in ("early", "middle", "late"):
            add_comparison(
                region_paired[region],
                {condition: losses[region] for condition, losses in region_sequence_losses.items()},
            )
    return {
        "conditions": conditions,
        "paired_utility": paired,
        "post_transition_paired_utility": post_paired,
        "post_transition_region_paired_utility": region_paired,
    }
