from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_stage_b_excludes_exactly_the_stage_a_evaluation_questions() -> None:
    exclusion_path = Path("configs/stage_b_excluded_question_ids.json")
    exclusion = json.loads(exclusion_path.read_text())
    stage_a_path = Path(exclusion["source_artifact"])
    stage_a = json.loads(stage_a_path.read_text())
    stage_a_ids = [example["squad_id"] for example in stage_a["examples"]]
    assert exclusion["count"] == 24
    assert len(set(exclusion["question_ids"])) == 24
    assert set(exclusion["question_ids"]) == set(stage_a_ids)
    assert (
        hashlib.sha256(stage_a_path.read_bytes()).hexdigest() == exclusion["source_artifact_sha256"]
    )


def test_stage_b_primary_aggregate_is_over_examples_after_seed_mean() -> None:
    specification = json.loads(Path("configs/stage_b_semantic_directory.json").read_text())
    aggregation = specification["seed_aggregation"]
    assert aggregation["primary"] == "per_example_mean_across_three_seeds"
    assert aggregation["uncertainty"] == "paired_bootstrap_over_examples"
    assert aggregation["bootstrap_samples"] == 2000
    assert aggregation["preserve_individual_seed_results"]
    assert not aggregation["treat_seeds_as_independent_examples"]
    assert not aggregation["select_best_seed"]
    assert (
        specification["datasets"]["evaluation"]["excluded_question_ids_path"]
        == "configs/stage_b_excluded_question_ids.json"
    )
