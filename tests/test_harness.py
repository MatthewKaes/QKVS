from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qkvs.harness import (
    load_run,
    run_experiment,
    status,
    validate_config,
    validate_config_file,
)


def test_checked_in_configurations_are_valid_and_only_smoke_is_runnable() -> None:
    smoke = validate_config_file("configs/harness_smoke.json")
    heatfield = validate_config_file("configs/heatfield_initial.json")
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


def test_configuration_rejects_budget_mismatch_and_missing_rolling() -> None:
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
    assert "proposal_count * window_width must equal historical_token_budget" in report["errors"]
    assert "a rolling-only condition is mandatory" in report["errors"]


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

    corrupted = json.loads(output.read_text())
    corrupted["source_identity"]["git_revision"] = "different-source"
    output.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="does not match"):
        run_experiment(
            "configs/harness_smoke.json",
            device="cpu",
            output_root=tmp_path,
            allow_dirty_source=True,
        )


def test_draft_heatfield_run_is_refused_before_model_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="not runnable"):
        run_experiment(
            "configs/heatfield_initial.json",
            device="cuda",
            output_root=tmp_path,
            allow_dirty_source=True,
        )


def test_validate_cli_does_not_import_torch_or_execute_model() -> None:
    root = Path(__file__).resolve().parent.parent
    script = """
import sys
from qkvs.harness import validate_config_file
report = validate_config_file('configs/harness_smoke.json')
assert report['valid']
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], cwd=root, check=True)


def test_module_help_exposes_one_common_interface() -> None:
    root = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        [sys.executable, "-m", "qkvs", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "{run,validate,status,summarize}" in completed.stdout
