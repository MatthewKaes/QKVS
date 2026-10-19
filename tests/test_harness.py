from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qkvs.harness import (
    _predictive_execution_version,
    _predictive_measurement_key,
    _Progress,
    _progress_bar,
    _progress_counts,
    _progress_line,
    _tokenized_text_streams,
    export_snapshot_data,
    load_run,
    print_summary,
    run_experiment,
    status,
    validate_config,
    validate_config_file,
)
from qkvs.metrics import summarize_snapshot_frontiers, summarize_snapshot_trial_set
from qkvs.selection import derive_random_block_seed


def test_run_wrapper_delegates_without_printing_a_validation_report(tmp_path, monkeypatch) -> None:
    executable = tmp_path / "qkvs"
    executable.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*"\n', encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    completed = subprocess.run(
        [Path("run.sh").resolve(), "configs/harness_smoke.json"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == "run --config configs/harness_smoke.json --device cpu\n"
    assert completed.stderr == ""


def test_progress_line_is_short_enough_to_never_wrap() -> None:
    line = _progress_line(3, 384, "heatfield / stream-002", 65, spinner="/", columns=80)
    assert line.startswith("/ [   3 / 384 ] [▒")
    bar = line.split("] [", maxsplit=1)[1].split("]", maxsplit=1)[0]
    assert len(bar) == 24
    assert set(bar) <= {"█", "▓", "▒", " "}
    assert "00:01:05" in line
    assert line.endswith("heatfield / stream-002")
    assert len(line) <= 79

    narrow = _progress_line(3, 384, "heatfield / stream-002", 65, columns=60)
    assert "00:01:05" in narrow
    assert len(narrow) <= 59


def test_progress_bar_uses_third_cell_resolution() -> None:
    assert _progress_bar(0, 9, width=3) == "   "
    assert _progress_bar(1, 9, width=3) == "▒  "
    assert _progress_bar(2, 9, width=3) == "▓  "
    assert _progress_bar(3, 9, width=3) == "█  "
    assert _progress_bar(4, 9, width=3) == "█▒ "
    assert _progress_bar(5, 9, width=3) == "█▓ "
    assert _progress_bar(9, 9, width=3) == "███"
    assert _progress_bar(4.5, 9, width=3) == "█▓ "


def test_training_progress_updates_fraction_without_advancing_durable_count() -> None:
    class InteractiveOutput(io.StringIO):
        def isatty(self) -> bool:
            return True

    output = InteractiveOutput()
    progress = _Progress(completed=4, total=17, stream=output, interval_seconds=60.0)
    progress.start_job("train 9/40 192/960 targets", fraction=0.0)
    progress.update_job("train 9/40 216/960 targets 12.5s/step", fraction=0.25)

    assert progress.completed == 4
    assert progress._current() == 5
    assert progress._bar_current() == pytest.approx(4.25)
    rendered = output.getvalue()
    assert "train 9/40" in rendered
    assert rendered.count("\n") == 0

    progress.complete_job()
    progress.close()
    assert progress.completed == 5
    assert output.getvalue().count("\n") == 1


def test_redirected_training_progress_does_not_emit_intermediate_lines() -> None:
    output = io.StringIO()
    progress = _Progress(completed=4, total=17, stream=output)
    progress.start_job("train 9/40 192/960 targets", fraction=0.0)
    progress.update_job("train 9/40 216/960 targets 12.5s/step", fraction=0.25)
    progress.complete_job()
    progress.close()

    assert output.getvalue().count("\n") == 1
    assert "192/960" in output.getvalue()
    assert "216/960" not in output.getvalue()


def test_interactive_progress_spins_on_one_erased_terminal_row() -> None:
    class InteractiveOutput(io.StringIO):
        def isatty(self) -> bool:
            return True

    output = InteractiveOutput()
    progress = _Progress(completed=2, total=12, stream=output, interval_seconds=60.0)
    progress.start_job("fixed_offsets / synthetic-002")
    progress.render()
    progress.complete_job()
    progress.close()

    rendered = output.getvalue()
    erase_line = "\r\x1b[2K"
    frames = [frame.rstrip("\n") for frame in rendered.split(erase_line)[1:]]
    assert frames[0].startswith("| ")
    assert frames[1].startswith("/ ")
    assert all(len(frame) <= 79 for frame in frames)
    assert rendered.count("\n") == 1


def test_progress_reports_each_job_once_without_carriage_returns() -> None:
    output = io.StringIO()
    progress = _Progress(completed=2, total=12, stream=output)
    progress.start_job("fixed_offsets / synthetic-002")
    progress.complete_job()
    progress.close()

    rendered = output.getvalue()
    assert rendered.count("fixed_offsets / synthetic-002") == 1
    assert rendered.count("\n") == 1
    assert "[  3 / 12 ]" in rendered
    assert "\r" not in rendered


def test_progress_restore_does_not_advance_durable_count() -> None:
    output = io.StringIO()
    progress = _Progress(completed=2, total=12, stream=output)
    progress.start_job("restore dense prefill", advances=False)
    progress.complete_job()
    progress.close()

    rendered = output.getvalue()
    assert "[  2 / 12 ]" in rendered
    assert "[  3 / 12 ]" not in rendered
    assert rendered.count("\n") == 1
    assert "\r" not in rendered


def test_long_context_progress_counts_prefill_training_and_evaluation() -> None:
    specification = json.loads(Path("configs/heatfield_learning_study.json").read_text())
    result = {
        "config": specification,
        "streams": [
            {"stream_id": boundary["id"], "split": boundary["split"]}
            for boundary in specification["data"]["boundaries"]
        ],
        "dense_prefill": {},
        "training": {},
        "measurements": [],
    }
    completed, total, phases = _progress_counts(result)

    assert completed == 0
    assert total == 17
    assert phases == {
        "dense_prefill": (0, 2),
        "training_checkpoints": (0, 10),
        "evaluation": (0, 5),
    }


def test_snapshot_summaries_report_discovered_and_not_found_without_impossibility_claims() -> None:
    candidates = [
        {
            "trial_id": 0,
            "candidate_seed": 11,
            "requested_sampled_fraction": 0.25,
            "realized_sampled_fraction": 0.25,
            "kl_divergence_nats": 0.2,
            "probability_distribution_overlap": 0.99,
            "next_token_ce_delta_nats": 0.1,
            "top_one_agreement": False,
            "selected_token_occurrences": 4,
            "aggregate_block_count": 2,
            "sampled_position_histogram": [2, 2],
        },
        {
            "trial_id": 1,
            "candidate_seed": 12,
            "requested_sampled_fraction": 0.25,
            "realized_sampled_fraction": 0.20,
            "kl_divergence_nats": 0.3,
            "probability_distribution_overlap": 0.9996,
            "next_token_ce_delta_nats": -0.1,
            "top_one_agreement": True,
            "selected_token_occurrences": 3,
            "aggregate_block_count": 2,
            "sampled_position_histogram": [1, 2],
        },
    ]
    summary = summarize_snapshot_trial_set(candidates, fidelity_tolerances=[0.1, 0.25])
    frontiers = summarize_snapshot_frontiers(
        [{"position": 7, "candidates": candidates}],
        fidelity_tolerances=[0.1, 0.25],
    )

    assert summary["successful_candidate_fraction"] == {"0.1": 0.0, "0.25": 0.5}
    assert summary["overlap_successful_candidate_fraction"] == {
        "0.9999": 0.0,
        "0.9995": 0.5,
        "0.995": 0.5,
        "0.95": 1.0,
    }
    assert summary["probability_distribution_overlap"] == {
        "minimum": 0.99,
        "median": 0.99,
        "mean": 0.9948,
        "maximum": 0.9996,
    }
    assert frontiers["frontiers_by_position"]["7"]["0.1"] == {
        "status": "not_found",
        "smallest_discovered_fraction": None,
    }
    assert frontiers["frontiers_by_position"]["7"]["0.25"]["status"] == "found"


def test_checked_in_configurations_preserve_prior_runs_and_draft_full_qkvs_training() -> None:
    smoke = validate_config_file("configs/harness_smoke.json")
    heatfield = validate_config_file("configs/heatfield_initial.json")
    pilot_path = Path("configs/heatfield_learning_study.json")
    followup_path = Path("configs/heatfield_learning_followup.json")
    hard_aligned_path = Path("configs/heatfield_hard_aligned_study.json")
    corrected_path = Path("configs/heatfield_hard_aligned_corrected_study.json")
    full_qkvs_path = Path("configs/heatfield_full_qkvs_learning_study.json")
    h1a_path = Path("configs/h1a_snapshot_compressibility.json")
    h1a_expansion_path = Path("configs/h1a_context_expansion.json")
    h1a_replication_path = Path("configs/h1a_independent_document_replication.json")
    h1a_cell_extension_path = Path("configs/h1a_kl_0_01_existing_cell_extension.json")
    h1a_position_extension_path = Path("configs/h1a_kl_0_01_position_extension.json")
    learning_study = validate_config_file(pilot_path)
    followup = validate_config_file(followup_path)
    hard_aligned = validate_config_file(hard_aligned_path)
    corrected = validate_config_file(corrected_path)
    full_qkvs = validate_config_file(full_qkvs_path)
    h1a = validate_config_file(h1a_path)
    h1a_expansion = validate_config_file(h1a_expansion_path)
    h1a_replication = validate_config_file(h1a_replication_path)
    h1a_cell_extension = validate_config_file(h1a_cell_extension_path)
    h1a_position_extension = validate_config_file(h1a_position_extension_path)
    assert smoke == {
        "valid": True,
        "runnable": True,
        "errors": [],
        "not_runnable_reasons": [],
        "experiment": "harness_synthetic_verification",
        "status": "harness_verification",
    }
    assert heatfield["valid"]
    assert not heatfield["runnable"]
    assert (
        "configuration is not authorized for harness execution" in heatfield["not_runnable_reasons"]
    )
    assert learning_study == {
        "valid": True,
        "runnable": True,
        "errors": [],
        "not_runnable_reasons": [],
        "experiment": "heatfield_long_context_learning_study",
        "status": "approved",
    }
    assert hashlib.sha256(pilot_path.read_bytes()).hexdigest() == (
        "32e35fc0a06907e39c2c917ea1fdedb54f52ff7c4773df5d350fd0d01c02bc66"
    )
    assert followup == {
        "valid": True,
        "runnable": True,
        "errors": [],
        "not_runnable_reasons": [],
        "experiment": "heatfield_absolute_mass_learning_followup",
        "status": "approved",
    }
    assert hashlib.sha256(followup_path.read_bytes()).hexdigest() == (
        "ca7258c563dfdc778d3d561839b12c53b7a03e86019aa0893b69abe716165030"
    )
    assert hard_aligned == {
        "valid": True,
        "runnable": True,
        "errors": [],
        "not_runnable_reasons": [],
        "experiment": "heatfield_hard_aligned_learning_study",
        "status": "approved",
    }
    assert hashlib.sha256(hard_aligned_path.read_bytes()).hexdigest() == (
        "badc55865c1ffab1c526222ded6b054a8d3c94b3b94e0f578c07b1f31876c96c"
    )
    assert corrected == {
        "valid": True,
        "runnable": True,
        "errors": [],
        "not_runnable_reasons": [],
        "experiment": "heatfield_corrected_hard_aligned_learning_study",
        "status": "approved",
    }
    assert hashlib.sha256(corrected_path.read_bytes()).hexdigest() == (
        "b36379abda410a6e4003cc2c16b6deaa46bb6abf327f4e047e93a8fc65f9252b"
    )
    assert full_qkvs == {
        "valid": True,
        "runnable": False,
        "errors": [],
        "not_runnable_reasons": ["configuration is not authorized for harness execution"],
        "experiment": "heatfield_full_qkvs_predictive_learning_study",
        "status": "draft_not_authorized",
    }
    assert h1a == {
        "valid": True,
        "runnable": True,
        "errors": [],
        "not_runnable_reasons": [],
        "experiment": "h1a_unconstrained_snapshot_compressibility",
        "status": "approved",
    }
    assert h1a_expansion == {
        "valid": True,
        "runnable": True,
        "errors": [],
        "not_runnable_reasons": [],
        "experiment": "h1a_context_density_expansion",
        "status": "approved",
    }
    assert h1a_replication == {
        "valid": True,
        "runnable": False,
        "errors": [],
        "not_runnable_reasons": ["configuration is not authorized for harness execution"],
        "experiment": "h1a_independent_document_replication",
        "status": "draft_not_authorized",
    }
    assert h1a_cell_extension == {
        "valid": True,
        "runnable": True,
        "errors": [],
        "not_runnable_reasons": [],
        "experiment": "h1a_kl_0_01_existing_cell_extension",
        "status": "approved",
    }
    assert h1a_position_extension == {
        "valid": True,
        "runnable": True,
        "errors": [],
        "not_runnable_reasons": [],
        "experiment": "h1a_kl_0_01_position_extension",
        "status": "approved",
    }
    assert hashlib.sha256(h1a_path.read_bytes()).hexdigest() == (
        "dfaea9ba01928ac8e1bc041a3d3ae94f1e0f126ae87623b6d7892454ce671425"
    )
    assert hashlib.sha256(Path("docs/data/H1a.json").read_bytes()).hexdigest() == (
        "c060125fc921642f3486bba111804ae5c41f2a24be02f5f84440ac834e0291ba"
    )
    h1a_specification = json.loads(h1a_path.read_text())
    assert all(
        field not in h1a_specification
        for field in ("attention", "selection", "training", "evaluation")
    )
    assert h1a_specification["study"] == {
        "kind": "unconstrained_snapshot_compressibility",
        "positions": [99, 999, 4095, 6143, 8191, 10239, 16382],
        "block_width": 8,
        "sampling_percentages": [1, 5, 10, 20, 40, 60, 80, 95, 99, 100],
        "trial_start": 0,
        "trials_per_fraction": 200,
        "sampling_algorithm": "uniform-aligned-blocks-v1",
        "fidelity_tolerances_nats": [0.001, 0.01, 0.05, 0.1],
        "dense_parity_tolerance": {"logit_atol": 0.005, "loss_atol": 0.005},
    }
    invalid_h1a = json.loads(json.dumps(h1a_specification))
    invalid_h1a["attention"] = {
        "rolling_tokens": 4096,
        "activation_tokens": 6144,
        "historical_token_budget": 512,
    }
    assert (
        "snapshot studies must not define QKVS attention or selector controls"
        in validate_config(invalid_h1a)["errors"]
    )

    expansion = json.loads(h1a_expansion_path.read_text())
    replication = json.loads(h1a_replication_path.read_text())
    cell_extension = json.loads(h1a_cell_extension_path.read_text())
    position_extension = json.loads(h1a_position_extension_path.read_text())
    original_positions = set(h1a_specification["study"]["positions"])
    expansion_positions = expansion["study"]["positions"]
    expansion_cohorts = expansion["study"]["position_cohorts"]
    assert len(expansion_positions) == len(set(expansion_positions)) == 60
    assert len(expansion_cohorts["stratified_512_to_32k"]) == 48
    assert len(expansion_cohorts["targeted_15_5k_to_16k"]) == 12
    assert set(expansion_positions).isdisjoint(original_positions)
    assert set(expansion_positions) == {
        position for cohort in expansion_cohorts.values() for position in cohort
    }
    assert all(15872 <= position < 16383 for position in expansion_cohorts["targeted_15_5k_to_16k"])
    assert min(expansion_cohorts["stratified_512_to_32k"]) >= 512
    assert max(expansion_cohorts["stratified_512_to_32k"]) <= 32766
    assert expansion["study"]["sampling_percentages"] == [
        1,
        5,
        10,
        20,
        40,
        60,
        70,
        80,
        90,
        95,
        99,
        100,
    ]
    assert expansion["study"]["trials_per_fraction"] == 400
    assert expansion["study"]["expected_candidate_passes"] == 264060
    assert expansion["study"]["expected_candidate_passes"] == 60 * (11 * 400 + 1)
    position_selection = expansion["study"]["position_selection"]
    assert position_selection["validated_max_position_embeddings"] == 32768

    def regenerate_positions(start, stop, count, cohort, excluded, selection_seed):
        regenerated = []
        for index in range(count):
            lower = start + index * (stop - start) // count
            upper = start + (index + 1) * (stop - start) // count
            position = lower + derive_random_block_seed(selection_seed, cohort, index) % (
                upper - lower
            )
            while position in excluded or position in regenerated:
                position = lower + (position - lower + 1) % (upper - lower)
            regenerated.append(position)
        return regenerated

    regenerated_stratified = regenerate_positions(
        *position_selection["stratified_range"],
        position_selection["stratified_count"],
        1,
        original_positions,
        position_selection["seed"],
    )
    regenerated_targeted = regenerate_positions(
        *position_selection["targeted_range"],
        position_selection["targeted_count"],
        2,
        original_positions | set(regenerated_stratified),
        position_selection["seed"],
    )
    assert regenerated_stratified == expansion_cohorts["stratified_512_to_32k"]
    assert regenerated_targeted == expansion_cohorts["targeted_15_5k_to_16k"]
    assert expansion["model"] == h1a_specification["model"]
    assert expansion["data"]["source"] == h1a_specification["data"]["source"]
    for field in (
        "block_width",
        "trial_start",
        "sampling_algorithm",
        "fidelity_tolerances_nats",
        "dense_parity_tolerance",
    ):
        assert expansion["study"][field] == h1a_specification["study"][field]
        assert replication["study"][field] == expansion["study"][field]
    assert (
        replication["study"]["sampling_percentages"]
        == h1a_specification["study"]["sampling_percentages"]
    )
    assert replication["study"]["trials_per_fraction"] == 200
    assert replication["study"]["expected_candidate_passes"] == 108060
    assert replication["study"]["positions"] == expansion_positions
    assert replication["study"]["position_cohorts"] == expansion_cohorts
    assert replication["model"] == expansion["model"]
    assert replication["data"]["source"] == {
        "path": "../data/project_gutenberg_2701.txt",
        "url": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
        "bytes": 1276315,
        "sha256": "b5e88b71a1cda068e9ab1d94344c9d932338ba6da1a66def97a1aecd9cfcc87d",
    }
    assert replication["data"]["source"] != expansion["data"]["source"]

    assert cell_extension["model"] == expansion["model"]
    assert cell_extension["data"] == expansion["data"]
    assert cell_extension["execution"] == expansion["execution"]
    assert cell_extension["study"]["positions"] == expansion_positions
    assert cell_extension["study"]["position_cohorts"] == expansion_cohorts
    assert (
        cell_extension["study"]["sampling_percentages"]
        == expansion["study"]["sampling_percentages"]
    )
    assert cell_extension["study"]["trial_start"] == 400
    assert cell_extension["study"]["trials_per_fraction"] == 100
    assert cell_extension["study"]["expected_candidate_passes"] == 60 * (11 * 100 + 1)
    assert cell_extension["study"]["fidelity_tolerances_nats"] == [0.01]
    assert cell_extension["study"]["reuse_dense_snapshots_from"] == (
        "../artifacts/results/h1a-context-density-expansion"
    )

    extension_positions = position_extension["study"]["positions"]
    assert len(extension_positions) == len(set(extension_positions)) == 50
    assert set(extension_positions).isdisjoint(original_positions | set(expansion_positions))
    assert min(extension_positions) >= 512
    assert max(extension_positions) <= 32766
    assert position_extension["model"] == expansion["model"]
    assert position_extension["data"] == expansion["data"]
    assert position_extension["execution"] == expansion["execution"]
    assert (
        position_extension["study"]["sampling_percentages"]
        == expansion["study"]["sampling_percentages"]
    )
    assert position_extension["study"]["trial_start"] == 0
    assert position_extension["study"]["trials_per_fraction"] == 500
    assert position_extension["study"]["expected_candidate_passes"] == 50 * (11 * 500 + 1)
    assert position_extension["study"]["fidelity_tolerances_nats"] == [0.01]
    extension_selection = position_extension["study"]["position_selection"]
    regenerated_extension = regenerate_positions(
        *extension_selection["stratified_range"],
        extension_selection["stratified_count"],
        3,
        original_positions | set(expansion_positions),
        extension_selection["seed"],
    )
    assert regenerated_extension == extension_positions

    pilot = json.loads(pilot_path.read_text())
    denser = json.loads(followup_path.read_text())
    hard = json.loads(hard_aligned_path.read_text())
    fixed = json.loads(corrected_path.read_text())
    full = json.loads(full_qkvs_path.read_text())
    assert denser["run"]["id"] != pilot["run"]["id"]
    assert denser["model"] == pilot["model"]
    assert denser["data"] == pilot["data"]
    assert denser["attention"] == pilot["attention"]
    assert denser["selection"] == pilot["selection"]
    assert denser["evaluation"] == pilot["evaluation"]
    assert denser["execution"] == pilot["execution"]
    assert denser["training"] == {
        **pilot["training"],
        "targets_per_rollout": 24,
    }
    assert [
        boundary["stop_token"] - boundary["start_token"]
        for boundary in denser["data"]["boundaries"]
    ] == [16384, 16384]
    assert len(denser["selection"]["conditions"]) == 5
    assert denser["training"]["steps"] == 40
    assert denser["training"]["steps"] * denser["training"]["targets_per_rollout"] == 960
    assert 960 // 24 == 40
    assert hard["run"] == {
        **denser["run"],
        "id": "heatfield-hard-aligned-learning-study",
    }
    assert hard["model"] == denser["model"]
    assert hard["data"] == denser["data"]
    assert hard["attention"] == denser["attention"]
    assert hard["selection"] == denser["selection"]
    assert hard["evaluation"] == denser["evaluation"]
    assert hard["execution"] == denser["execution"]
    assert hard["training"] == {
        **denser["training"],
        "utility_gradient_estimator": "hard_multiplicity_ste",
    }
    assert fixed["run"] == {
        **hard["run"],
        "id": "heatfield-corrected-hard-aligned-learning-study",
    }
    assert fixed["model"] == hard["model"]
    assert fixed["data"] == hard["data"]
    assert fixed["attention"] == hard["attention"]
    assert fixed["selection"] == hard["selection"]
    assert fixed["training"] == hard["training"]
    assert fixed["evaluation"] == hard["evaluation"]
    assert fixed["execution"] == hard["execution"]
    assert full["run"] == {
        **fixed["run"],
        "id": "heatfield-full-qkvs-predictive-learning-study",
    }
    assert full["model"] == fixed["model"]
    assert full["data"] == fixed["data"]
    assert full["attention"] == fixed["attention"]
    assert full["selection"] == fixed["selection"]
    assert full["training"] == {
        **fixed["training"],
        "predictive_continuation": "qkvs",
    }
    assert full["evaluation"] == fixed["evaluation"]
    assert full["execution"] == fixed["execution"]

    invalid = json.loads(json.dumps(hard))
    invalid["training"]["utility_gradient_estimator"] = "unsupported"
    assert (
        "learned heat-field training configuration is incomplete"
        in validate_config(invalid)["not_runnable_reasons"]
    )
    invalid = json.loads(json.dumps(full))
    invalid["training"]["predictive_continuation"] = "unsupported"
    assert (
        "learned heat-field training configuration is incomplete"
        in validate_config(invalid)["not_runnable_reasons"]
    )
    invalid = json.loads(json.dumps(full))
    invalid["training"]["utility_gradient_estimator"] = "soft_occupancy"
    assert (
        "learned heat-field training configuration is incomplete"
        in validate_config(invalid)["not_runnable_reasons"]
    )


def test_checked_in_h1a_csv_export_is_complete_and_self_consistent() -> None:
    directory = Path("docs/data/h1a-unconstrained-snapshot-compressibility")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["format"] == "qkvs-h1a-csv-v1"
    assert manifest["run_id"] == "h1a-unconstrained-snapshot-compressibility"
    assert manifest["source_identity"]["git_revision"] == (
        "f53dba9e8c46ea2c95e80fa996c3d99a2ad0f9a1"
    )
    assert manifest["sampled_location_histogram_bins"] == 20
    for name, record in manifest["csv"].items():
        path = directory / name
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]

    with (directory / "H1a_trials.csv").open(newline="") as source:
        trials = list(csv.DictReader(source))
    with (directory / "H1a_snapshots.csv").open(newline="") as source:
        snapshots = list(csv.DictReader(source))
    with (directory / "H1a_feasibility.csv").open(newline="") as source:
        feasibility = list(csv.DictReader(source))
    assert len(trials) == manifest["csv"]["H1a_trials.csv"]["rows"] == 12607
    assert len(snapshots) == manifest["csv"]["H1a_snapshots.csv"]["rows"] == 7
    assert len(feasibility) == manifest["csv"]["H1a_feasibility.csv"]["rows"] == 560
    identities = [
        (
            row["stream_id"],
            int(row["position"]),
            float(row["requested_sampled_fraction"]),
            int(row["trial_id"]),
        )
        for row in trials
    ]
    assert identities == sorted(identities)
    assert len(identities) == len(set(identities))
    assert all(
        row["candidate_seed_hex"].startswith("0x")
        and len(row["candidate_seed_hex"]) == 18
        and 0 <= int(row["candidate_seed_hex"], 16) < 2**64
        for row in trials
    )
    assert sum(column.startswith("sampled_position_bin_") for column in trials[0]) == 20

    trials_by_fraction: dict[tuple[int, float], list[dict[str, str]]] = {}
    for row in trials:
        trials_by_fraction.setdefault(
            (int(row["position"]), float(row["requested_sampled_fraction"])), []
        ).append(row)
    for row in feasibility:
        candidates = trials_by_fraction[
            (int(row["position"]), float(row["requested_sampled_fraction"]))
        ]
        threshold = float(row["threshold"])
        metric = row["metric"]
        if row["comparison"] == "<=":
            success_count = sum(float(candidate[metric]) <= threshold for candidate in candidates)
        else:
            assert row["comparison"] == ">="
            success_count = sum(float(candidate[metric]) >= threshold for candidate in candidates)
        assert int(row["success_count"]) == success_count
        assert int(row["attempts"]) == len(candidates)
        assert float(row["success_probability"]) == success_count / len(candidates)

    for position, frontiers in manifest["original_frontiers"]["frontiers_by_position"].items():
        position_trials = [row for row in trials if row["position"] == position]
        for tolerance, frontier in frontiers.items():
            successful = [
                row
                for row in position_trials
                if float(row["kl_divergence_nats"]) <= float(tolerance)
            ]
            assert frontier["status"] == "found"
            discovered = min(
                successful,
                key=lambda row: (
                    float(row["realized_sampled_fraction"]),
                    float(row["kl_divergence_nats"]),
                    int(row["trial_id"]),
                ),
            )
            assert (
                float(discovered["realized_sampled_fraction"])
                == frontier["smallest_discovered_fraction"]
            )
            assert (
                float(discovered["requested_sampled_fraction"])
                == frontier["requested_sampled_fraction"]
            )
            assert int(discovered["trial_id"]) == frontier["trial_id"]
            assert int(discovered["candidate_seed_hex"], 16) == frontier["candidate_seed"]
            assert float(discovered["kl_divergence_nats"]) == frontier["kl_divergence_nats"]


def test_checked_in_h1a_expansion_export_is_complete_and_compressed() -> None:
    directory = Path("docs/data/h1a-context-density-expansion")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["format"] == "qkvs-h1a-csv-v1"
    assert manifest["run_id"] == "h1a-context-density-expansion"
    assert manifest["source_identity"]["git_revision"] == (
        "8b9b38969bda9fe10f4fd70b8a5c6d52ec57e4e6"
    )
    assert manifest["configuration_sha256"] == (
        "36536e12d8736492a42aadbcf06b18eb562810f0466fbee000288ef4b96e71cf"
    )
    assert manifest["sampled_location_histogram_bins"] == 20
    assert len(manifest["original_artifacts"]["trial_sets"]) == 720
    assert len(manifest["original_artifacts"]["dense_prefixes"]) == 60
    assert len(manifest["original_frontiers"]["frontiers_by_position"]) == 60
    for name, record in manifest["csv"].items():
        path = directory / name
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]

    trial_record = manifest["csv"]["H1a_trials.csv.gz"]
    assert trial_record["compression"] == "gzip"
    assert trial_record["rows"] == 264060
    assert trial_record["uncompressed_name"] == "H1a_trials.csv"
    previous_identity = None
    trial_count = 0
    with gzip.open(directory / "H1a_trials.csv.gz", "rt", newline="") as source:
        for row in csv.DictReader(source):
            identity = (
                row["stream_id"],
                int(row["position"]),
                float(row["requested_sampled_fraction"]),
                int(row["trial_id"]),
            )
            if previous_identity is not None:
                assert previous_identity < identity
            previous_identity = identity
            assert row["candidate_seed_hex"].startswith("0x")
            assert len(row["candidate_seed_hex"]) == 18
            trial_count += 1
    assert trial_count == trial_record["rows"]
    with (directory / "H1a_snapshots.csv").open(newline="") as source:
        assert sum(1 for _ in csv.DictReader(source)) == 60
    with (directory / "H1a_feasibility.csv").open(newline="") as source:
        feasibility = list(csv.DictReader(source))
    assert len(feasibility) == 5760
    with (directory / "H1a_KL_0_01_feasibility.csv").open(newline="") as source:
        compact_feasibility = list(csv.DictReader(source))
    assert compact_feasibility == [
        row
        for row in feasibility
        if row["metric"] == "kl_divergence_nats" and float(row["threshold"]) == 0.01
    ]
    assert len(compact_feasibility) == 720


def test_checked_in_h1a_heatmaps_are_reproducible(tmp_path) -> None:
    directory = Path("docs/data/h1a-context-density-expansion")
    subprocess.run(
        [
            sys.executable,
            directory / "visualize.py",
            "--output-directory",
            tmp_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for name in (
        "H1a_KL_0_001.svg",
        "H1a_KL_0_001.png",
        "H1a_KL_0_01.svg",
        "H1a_KL_0_01.png",
    ):
        assert (tmp_path / name).read_bytes() == (directory / name).read_bytes()


def test_completed_followup_is_legacy_compatible_only_for_matched_baselines() -> None:
    result = {
        "source_identity": {
            "git_revision": "b63e9b54f11543c571298de215bff567e1529efb",
            "files": {
                "qkvs/model.py": (
                    "e0873329d2eda3db219b7cff463016a987a6906c16d83b4a0425097ee22ad7bd"
                ),
                "qkvs/selection.py": (
                    "ba195153672aa9b77a50b441bf8c26453c5da788f2207e7387b1c92783bb0ac3"
                ),
            },
        }
    }

    assert _predictive_execution_version(result, "rolling")
    assert _predictive_execution_version(result, "fixed_offsets")
    assert _predictive_execution_version(result, "full_qkv")
    assert _predictive_execution_version(result, "heatfield_untrained") is None


def test_snapshot_study_resumes_atomic_trial_sets_without_duplicate_observations(
    tmp_path, monkeypatch, capfd
) -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import qkvs.model as model_module

    class ByteTokenizer:
        init_kwargs = {"_commit_hash": "1" * 40}

        def encode(self, text, *, add_special_tokens, verbose):
            assert not add_special_tokens
            assert not verbose
            return [byte % 64 for byte in text.encode()]

        def get_vocab(self):
            return {str(index): index for index in range(64)}

    reader = model_module.HuggingFaceQwen2Reader(
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                attention_dropout=0.0,
            )
        ),
        ByteTokenizer(),
        specification={
            "id": "test/tiny-qwen2",
            "revision": "1" * 40,
            "dtype": "float32",
            "frozen": True,
        },
    )
    source = tmp_path / "snapshot-source.txt"
    source.write_text("abcdefghijklmnop", encoding="utf-8")
    specification = {
        "schema_version": 1,
        "experiment": "snapshot_harness_smoke",
        "status": "harness_verification",
        "run": {"id": "snapshot-harness-smoke", "seed": 91},
        "model": {
            "backend": "huggingface",
            "id": "test/tiny-qwen2",
            "revision": "1" * 40,
            "dtype": "float32",
            "frozen": True,
        },
        "data": {
            "kind": "tokenized_text",
            "source": {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            "boundaries": [
                {
                    "id": "snapshot-stream",
                    "split": "evaluation",
                    "start_token": 0,
                    "stop_token": 12,
                }
            ],
        },
        "study": {
            "kind": "unconstrained_snapshot_compressibility",
            "positions": [3, 7],
            "block_width": 2,
            "sampling_percentages": [25, 100],
            "trial_start": 0,
            "trials_per_fraction": 3,
            "sampling_algorithm": "uniform-aligned-blocks-v1",
            "fidelity_tolerances_nats": [0.001, 0.1],
            "dense_parity_tolerance": {"logit_atol": 1e-6, "loss_atol": 1e-6},
        },
        "execution": {
            "allowed_devices": ["cpu"],
            "output_root": str(tmp_path),
            "prefill_chunk_tokens": 2,
        },
    }
    assert validate_config(specification)["runnable"]
    config = tmp_path / "snapshot-config.json"
    config.write_text(json.dumps(specification), encoding="utf-8")
    monkeypatch.setattr(model_module, "load_reader", lambda model, device: reader)
    original_probe = reader.snapshot_probe
    probe_calls = 0

    def interrupted_probe(*args, **kwargs):
        nonlocal probe_calls
        probe_calls += 1
        if probe_calls == 5:
            raise RuntimeError("test interruption after one atomic snapshot trial set")
        return original_probe(*args, **kwargs)

    monkeypatch.setattr(reader, "snapshot_probe", interrupted_probe)
    with pytest.raises(RuntimeError, match="test interruption"):
        run_experiment(
            config,
            device="cpu",
            output_root=tmp_path / "results",
            allow_dirty_source=True,
        )
    partial_output = tmp_path / "results" / "snapshot-harness-smoke" / "result.json"
    partial = json.loads(partial_output.read_text())
    assert len(partial["snapshot_prefixes"]) == 1
    assert len(partial["snapshot_observations"]) == 1
    partial_observation_path = partial_output.parent / partial["snapshot_observations"][0]["path"]
    partial_observation = json.loads(partial_observation_path.read_text())
    partial_overlaps = [
        candidate["probability_distribution_overlap"]
        for candidate in partial_observation["candidates"]
    ]
    partial_status = status(partial_output)
    assert partial_status["completed_units"] == 2
    assert partial_status["total_units"] == 6
    assert partial_status["phase_units"] == {
        "snapshot_prefixes": {"completed": 1, "total": 2},
        "snapshot_trial_sets": {"completed": 1, "total": 4},
    }

    monkeypatch.setattr(reader, "snapshot_probe", original_probe)
    output, resumed = run_experiment(
        config,
        device="cpu",
        output_root=tmp_path / "results",
        allow_dirty_source=True,
    )
    assert resumed["complete"]
    assert len(resumed["snapshot_prefixes"]) == 2
    assert len(resumed["snapshot_observations"]) == 4
    assert (
        len(
            {
                (record["position"], record["requested_sampling_percentage"])
                for record in resumed["snapshot_observations"]
            }
        )
        == 4
    )

    observations = [
        json.loads((output.parent / record["path"]).read_text())
        for record in resumed["snapshot_observations"]
    ]
    dense = [record for record in observations if record["requested_sampling_percentage"] == 100]
    assert all(len(record["candidates"]) == 1 for record in dense)
    assert all(record["candidates"][0]["full_logit_max_abs_error"] == 0.0 for record in dense)
    assert all(record["candidates"][0]["full_loss_abs_error"] == 0.0 for record in dense)
    assert all(
        record["candidates"][0]["probability_distribution_overlap"] == pytest.approx(1.0, abs=1e-6)
        for record in dense
    )
    assert all(
        -1e-6 <= candidate["probability_distribution_overlap"] <= 1.0 + 1e-6
        for observation in observations
        for candidate in observation["candidates"]
    )
    resumed_partial = next(
        observation
        for observation in observations
        if observation["position"] == partial_observation["position"]
        and observation["requested_sampling_percentage"]
        == partial_observation["requested_sampling_percentage"]
    )
    assert [
        candidate["probability_distribution_overlap"] for candidate in resumed_partial["candidates"]
    ] == partial_overlaps
    assert all(
        set(observation["summary"]["overlap_successful_candidate_fraction"])
        == {"0.9999", "0.9995", "0.995", "0.95"}
        for observation in observations
    )
    assert all(
        candidate["selected_token_occurrences"]
        == candidate["actual_gathered_key_occurrences"]
        == candidate["actual_gathered_value_occurrences"]
        for observation in observations
        for candidate in observation["candidates"]
    )
    assert all(
        candidate["selected_token_occurrences"]
        == sum(layer["selected_token_count"] for layer in candidate["per_layer_selection"])
        and candidate["aggregate_block_count"] == sum(candidate["sampled_block_count_per_layer"])
        and all(
            layer["selected_token_count"]
            == layer["gathered_key_occurrences"]
            == layer["gathered_value_occurrences"]
            for layer in candidate["per_layer_selection"]
        )
        for observation in observations
        for candidate in observation["candidates"]
    )
    assert all(
        "per_layer_selected_ranges" in observation["summary"]["best_candidate"]
        for observation in observations
    )
    assert resumed["snapshot_summary"]["interpretation"] == (
        "minimum_discovered_random_sample_not_theoretical_minimum"
    )
    print_summary(resumed)
    assert "KL<=0.1" in capfd.readouterr().out

    export_directory = tmp_path / "export"
    original_hashes = {
        record["path"]: hashlib.sha256((output.parent / record["path"]).read_bytes()).hexdigest()
        for record in resumed["snapshot_observations"]
    }
    manifest = export_snapshot_data(output, export_directory)
    with (export_directory / "H1a_trials.csv").open(newline="") as source:
        trial_rows = list(csv.DictReader(source))
    with (export_directory / "H1a_snapshots.csv").open(newline="") as source:
        snapshot_rows = list(csv.DictReader(source))
    with (export_directory / "H1a_feasibility.csv").open(newline="") as source:
        feasibility_rows = list(csv.DictReader(source))
    assert manifest["format"] == "qkvs-h1a-csv-v1"
    assert manifest["sampled_location_histogram_bins"] == 20
    assert manifest["csv"]["H1a_trials.csv"]["rows"] == len(trial_rows) == 8
    assert manifest["csv"]["H1a_snapshots.csv"]["rows"] == len(snapshot_rows) == 2
    assert manifest["csv"]["H1a_feasibility.csv"]["rows"] == len(feasibility_rows) == 24
    assert all(
        row["candidate_seed_hex"].startswith("0x") and len(row["candidate_seed_hex"]) == 18
        for row in trial_rows
    )
    assert len(
        {
            (
                row["stream_id"],
                int(row["position"]),
                float(row["requested_sampled_fraction"]),
                int(row["trial_id"]),
            )
            for row in trial_rows
        }
    ) == len(trial_rows)
    assert [
        (
            row["stream_id"],
            int(row["position"]),
            float(row["requested_sampled_fraction"]),
            int(row["trial_id"]),
        )
        for row in trial_rows
    ] == sorted(
        (
            row["stream_id"],
            int(row["position"]),
            float(row["requested_sampled_fraction"]),
            int(row["trial_id"]),
        )
        for row in trial_rows
    )
    original_candidates = {
        (
            observation["position"],
            observation["requested_sampled_fraction"],
            candidate["trial_id"],
        ): candidate
        for observation in observations
        for candidate in observation["candidates"]
    }
    for row in trial_rows:
        candidate = original_candidates[
            (
                int(row["position"]),
                float(row["requested_sampled_fraction"]),
                int(row["trial_id"]),
            )
        ]
        assert int(row["candidate_seed_hex"], 16) == candidate["candidate_seed"]
        assert float(row["kl_divergence_nats"]) == candidate["kl_divergence_nats"]
        assert (
            float(row["probability_distribution_overlap"])
            == candidate["probability_distribution_overlap"]
        )
        assert float(row["next_token_ce_delta_nats"]) == candidate["next_token_ce_delta_nats"]
        assert [int(row[f"sampled_position_bin_{index:02d}"]) for index in range(20)] == candidate[
            "sampled_position_histogram"
        ]
    assert {
        record["path"]: hashlib.sha256((output.parent / record["path"]).read_bytes()).hexdigest()
        for record in resumed["snapshot_observations"]
    } == original_hashes
    for name, record in manifest["csv"].items():
        csv_path = export_directory / name
        assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == record["sha256"]
        assert csv_path.stat().st_size == record["bytes"]
    second_export = tmp_path / "second-export"
    export_snapshot_data(output, second_export)
    assert all(
        (export_directory / name).read_bytes() == (second_export / name).read_bytes()
        for name in (
            "H1a_trials.csv",
            "H1a_snapshots.csv",
            "H1a_feasibility.csv",
            "manifest.json",
        )
    )
    compressed_export = tmp_path / "compressed-export"
    compressed_manifest = export_snapshot_data(
        output,
        compressed_export,
        compress_trials=True,
    )
    compressed_record = compressed_manifest["csv"]["H1a_trials.csv.gz"]
    assert compressed_record["compression"] == "gzip"
    assert compressed_record["uncompressed_name"] == "H1a_trials.csv"
    assert compressed_record["uncompressed_sha256"] == manifest["csv"]["H1a_trials.csv"]["sha256"]
    assert compressed_record["uncompressed_bytes"] == manifest["csv"]["H1a_trials.csv"]["bytes"]
    with gzip.open(compressed_export / "H1a_trials.csv.gz", "rb") as source:
        assert source.read() == (export_directory / "H1a_trials.csv").read_bytes()

    _, repeated = run_experiment(
        config,
        device="cpu",
        output_root=tmp_path / "results",
        allow_dirty_source=True,
    )
    assert repeated == resumed

    reuse_specification = json.loads(json.dumps(specification))
    reuse_specification["run"]["id"] = "snapshot-harness-reuse-smoke"
    reuse_specification["study"]["trial_start"] = 3
    reuse_specification["study"]["trials_per_fraction"] = 1
    reuse_specification["study"]["reuse_dense_snapshots_from"] = str(output)
    reuse_config = tmp_path / "snapshot-reuse-config.json"
    reuse_config.write_text(json.dumps(reuse_specification), encoding="utf-8")

    def unexpected_dense_snapshot(*args, **kwargs):
        raise AssertionError("reused dense snapshots must not be recomputed")

    monkeypatch.setattr(reader, "dense_causal_snapshot", unexpected_dense_snapshot)
    reused_output, reused = run_experiment(
        reuse_config,
        device="cpu",
        output_root=tmp_path / "reuse-results",
        allow_dirty_source=True,
    )
    assert reused["complete"]
    assert reused["dense_snapshot_reuse"] == {
        "configured_path": str(output),
        "run_id": "snapshot-harness-smoke",
        "result_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    assert all(
        record["reused_from"]["run_id"] == "snapshot-harness-smoke"
        for record in reused["snapshot_prefixes"].values()
    )
    assert not (reused_output.parent / "snapshot_prefixes").exists()


def test_predictive_keys_track_condition_specific_invalidation_boundaries() -> None:
    specification = json.loads(Path("configs/heatfield_learning_followup.json").read_text())
    stream = {
        "stream_id": "evaluation-sherlock-000",
        "token_sha256": "1" * 64,
        "support_positions": [None, 0],
    }
    result = {
        "config": specification,
        "model_identity": {"model": "frozen", "dtype": "bfloat16"},
        "dense_prefill": {stream["stream_id"]: {"identity": {"prefix": "2" * 64}}},
        "device": "cuda",
        "environment": {
            "python": "3.14.7",
            "platform": "test-platform",
            "torch": "2.9.0+cu130",
            "torch_cuda": "13.0",
        },
        "predictive_execution_versions": {
            "rolling": "qwen2-eager-rolling-v1",
            "fixed_offsets": "qwen2-eager-fixed-offsets-v1",
            "full_qkv": "qwen2-eager-full-qkv-v1",
            "heatfield_untrained": "qwen2-eager-untrained-heatfield-v2",
            "heatfield": "qwen2-eager-trained-heatfield-v2",
        },
        "untrained_policy_sha256": "3" * 64,
    }
    conditions = {
        condition["policy"]: condition for condition in specification["selection"]["conditions"]
    }
    original = {
        policy: _predictive_measurement_key(result, condition, stream)
        for policy, condition in conditions.items()
    }
    training_only = json.loads(json.dumps(result))
    training_only["config"]["training"]["coverage_weight"] = 0.25
    assert {
        policy: _predictive_measurement_key(training_only, condition, stream)
        for policy, condition in conditions.items()
    } == original

    changed_working_memory = json.loads(json.dumps(result))
    changed_working_memory["config"]["attention"]["rolling_tokens"] += 1
    assert (
        _predictive_measurement_key(changed_working_memory, conditions["rolling"], stream)
        != original["rolling"]
    )

    changed_offsets = json.loads(json.dumps(result))
    changed_offsets["config"]["selection"]["conditions"][1]["offsets"][0] += 1
    assert (
        _predictive_measurement_key(
            changed_offsets,
            changed_offsets["config"]["selection"]["conditions"][1],
            stream,
        )
        != original["fixed_offsets"]
    )

    changed_untrained = json.loads(json.dumps(result))
    changed_untrained["untrained_policy_sha256"] = "4" * 64
    assert (
        _predictive_measurement_key(
            changed_untrained,
            conditions["heatfield_untrained"],
            stream,
        )
        != original["heatfield_untrained"]
    )
    assert original["heatfield"] is None


def test_pinned_huggingface_text_configuration_is_runnable_in_schema() -> None:
    specification = json.loads(Path("configs/harness_smoke.json").read_text())
    specification["model"] = {
        "backend": "huggingface",
        "id": "organization/test-qwen2",
        "revision": "1" * 40,
        "dtype": "float32",
        "frozen": True,
    }
    specification["data"] = {
        "kind": "tokenized_text",
        "source": {"path": "source.txt", "sha256": "2" * 64},
        "boundaries": [{"id": "stream", "start_token": 0, "stop_token": 16}],
    }
    report = validate_config(specification)
    assert report["valid"]
    assert report["runnable"]

    specification["model"]["revision"] = "main"
    report = validate_config(specification)
    assert not report["valid"]
    assert "model.revision must be a full 40-character Git commit" in report["errors"]


def test_configuration_keeps_budget_independent_and_rejects_missing_rolling() -> None:
    specification = json.loads(Path("configs/harness_smoke.json").read_text())
    specification["attention"]["historical_token_budget"] = 5
    specification["selection"]["conditions"] = [
        {
            "name": "fixed",
            "policy": "fixed_offsets",
            "offsets": [8, 16],
        },
        {
            "name": "fixed_two",
            "policy": "fixed_offsets",
            "offsets": [10, 18],
        },
    ]
    specification["evaluation"]["baseline"] = "fixed"
    report = validate_config(specification)
    assert not report["valid"]
    assert "a rolling-only condition is mandatory" in report["errors"]
    assert not any("proposal_count * window_width" in error for error in report["errors"])


def test_configuration_rejects_a_historical_window_that_cannot_execute() -> None:
    specification = json.loads(Path("configs/harness_smoke.json").read_text())
    specification["attention"]["historical_token_budget"] = 1
    specification["selection"]["window_width"] = 2
    report = validate_config(specification)
    assert not report["valid"]
    assert (
        "selection.window_width cannot exceed the historical token budget for fixed_offsets"
        in report["errors"]
    )


def test_configuration_rejects_invalid_or_unreached_activation() -> None:
    specification = json.loads(Path("configs/harness_smoke.json").read_text())
    specification["attention"]["activation_tokens"] = 3
    report = validate_config(specification)
    assert "attention.activation_tokens must be at least rolling_tokens" in report["errors"]

    specification["attention"]["activation_tokens"] = specification["data"]["sequence_tokens"]
    report = validate_config(specification)
    assert "data.sequence_tokens must extend beyond attention.activation_tokens" in report["errors"]


def test_tokenized_text_streams_have_fixed_boundaries_and_source_identity(tmp_path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("deterministic real token stream", encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    class ByteTokenizer:
        def encode(self, text, *, add_special_tokens, verbose):
            assert not add_special_tokens
            assert not verbose
            return list(text.encode())

    specification = {
        "data": {
            "kind": "tokenized_text",
            "source": {"path": "source.txt", "sha256": source_hash},
            "boundaries": [
                {"id": "first", "start_token": 0, "stop_token": 8},
                {"id": "second", "start_token": 8, "stop_token": 18},
            ],
        }
    }
    first = _tokenized_text_streams(
        specification,
        config_path=tmp_path / "config.json",
        tokenizer=ByteTokenizer(),
    )
    second = _tokenized_text_streams(
        specification,
        config_path=tmp_path / "config.json",
        tokenizer=ByteTokenizer(),
    )
    assert first == second
    streams, identity = first
    assert [stream["stream_id"] for stream in streams] == ["first", "second"]
    assert all(set(stream["support_positions"]) == {None} for stream in streams)
    assert identity["source_sha256"] == source_hash
    assert identity["boundaries"] == specification["data"]["boundaries"]

    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source.sha256"):
        _tokenized_text_streams(
            specification,
            config_path=tmp_path / "config.json",
            tokenizer=ByteTokenizer(),
        )


def test_missing_pinned_text_is_downloaded_atomically(tmp_path, monkeypatch) -> None:
    import urllib.request

    payload = b"one genuinely fetched source, not a repeated synthetic passage"
    source = tmp_path / "downloaded.txt"

    class Response(io.BytesIO):
        pass

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, timeout: Response(payload),
    )
    specification = {
        "data": {
            "kind": "tokenized_text",
            "source": {
                "path": source.name,
                "url": "https://example.invalid/pinned.txt",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            "boundaries": [{"id": "text", "start_token": 0, "stop_token": 8}],
        }
    }

    class ByteTokenizer:
        def encode(self, text, *, add_special_tokens, verbose):
            assert not add_special_tokens
            assert not verbose
            return list(text.encode())

    _, identity = _tokenized_text_streams(
        specification,
        config_path=tmp_path / "config.json",
        tokenizer=ByteTokenizer(),
    )
    assert source.read_bytes() == payload
    assert identity["source_url"] == "https://example.invalid/pinned.txt"
    assert identity["source_bytes"] == len(payload)
    assert not source.with_suffix(".txt.tmp").exists()


def test_cpu_run_is_reproducible_resumable_and_paired(tmp_path) -> None:
    output, first = run_experiment(
        "configs/harness_smoke.json",
        device="cpu",
        output_root=tmp_path,
        allow_dirty_source=True,
    )
    assert output.exists()
    assert not output.with_suffix(".json.tmp").exists()
    assert first["complete"]
    assert len(first["measurements"]) == 8
    assert set(first["summary"]["conditions"]) == {"rolling", "fixed_offsets"}
    assert first["summary"]["paired_utility"]["fixed_offsets"]["paired_streams"] == 4
    assert (
        first["summary"]["conditions"]["rolling"]["retrieval"]["historical_token_occurrences"] == 0
    )
    assert (
        first["summary"]["conditions"]["fixed_offsets"]["retrieval"]["historical_token_occurrences"]
        > 0
    )

    _, resumed = run_experiment(
        "configs/harness_smoke.json",
        device="cpu",
        output_root=tmp_path,
        allow_dirty_source=True,
    )
    assert resumed == first
    progress = status(output.parent)
    assert progress["complete"]
    assert progress["completed_units"] == progress["total_units"] == 8
    assert load_run(output)["summary"] == first["summary"]

    for identity_key in ("source_identity", "model_identity", "data_identity"):
        corrupted = json.loads(json.dumps(first))
        corrupted[identity_key]["corrupted"] = True
        output.write_text(json.dumps(corrupted))
        with pytest.raises(ValueError, match="does not match"):
            run_experiment(
                "configs/harness_smoke.json",
                device="cpu",
                output_root=tmp_path,
                allow_dirty_source=True,
            )
    output.write_text(json.dumps(first))

    partial = json.loads(json.dumps(first))
    partial["measurements"].pop()
    partial["complete"] = False
    output.write_text(json.dumps(partial))
    _, completed = run_experiment(
        "configs/harness_smoke.json",
        device="cpu",
        output_root=tmp_path,
        allow_dirty_source=True,
    )
    units = [
        (measurement["condition"], measurement["stream_id"])
        for measurement in completed["measurements"]
    ]
    assert completed["complete"]
    assert len(units) == len(set(units)) == 8


def test_draft_heatfield_run_is_refused_before_model_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="not runnable"):
        run_experiment(
            "configs/heatfield_initial.json",
            device="cuda",
            output_root=tmp_path,
            allow_dirty_source=True,
        )


def test_validate_status_and_summary_paths_do_not_import_torch_or_execute_model(tmp_path) -> None:
    root = Path(__file__).resolve().parent.parent
    artifact = tmp_path / "result.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "harness-smoke",
                "config": json.loads(Path("configs/harness_smoke.json").read_text()),
                "config_sha256": "0" * 64,
                "source_identity": {"git_revision": "test"},
                "streams": [{"stream_id": "synthetic-000", "split": "evaluation"}],
                "dense_prefill": {},
                "training": {},
                "measurements": [{"condition": "rolling", "stream_id": "synthetic-000"}],
                "summary": {},
                "complete": False,
            }
        ),
        encoding="utf-8",
    )
    script = """
import sys
from qkvs.harness import load_run, print_summary, status, validate_config_file
report = validate_config_file('configs/harness_smoke.json')
assert report['valid']
result = load_run(sys.argv[1])
assert result['measurements']
assert status(sys.argv[1])['run_id'] == 'harness-smoke'
print_summary(result)
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, str(artifact)], cwd=root, check=True)


def test_module_help_exposes_one_common_interface() -> None:
    root = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        [sys.executable, "-m", "qkvs", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.startswith("usage: qkvs ")
    assert "{run,validate,status,summarize,export-snapshot}" in completed.stdout
