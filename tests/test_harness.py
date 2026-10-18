from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from qkvs.harness import (
    _LiveProgress,
    _progress_line,
    _tokenized_text_streams,
    load_run,
    run_experiment,
    status,
    validate_config,
    validate_config_file,
)


def test_live_progress_uses_fixed_counter_and_space_filled_bar() -> None:
    line = _progress_line(3, 384, "heatfield / stream-002", 65)
    assert line.startswith("[   3 / 384 ] [#")
    bar = line.split("] [", maxsplit=1)[1].split("]", maxsplit=1)[0]
    assert len(bar) == 40
    assert set(bar) <= {"#", " "}
    assert line.endswith("heatfield / stream-002 : 00:01:05")


def test_live_progress_rewrites_one_line_and_terminates_once() -> None:
    output = io.StringIO()
    progress = _LiveProgress(
        completed=2,
        total=12,
        stream=output,
        interval_seconds=60.0,
    )
    progress.start_job("fixed_offsets / synthetic-002")
    progress.complete_job()
    progress.close()
    rendered = output.getvalue()
    assert rendered.count("\r") >= 3
    assert rendered.count("\n") == 1
    assert "[  3 / 12 ]" in rendered


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


def test_tokenized_text_streams_have_fixed_boundaries_and_source_identity(tmp_path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("deterministic real token stream", encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    class ByteTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
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


def test_validate_status_and_summary_paths_do_not_import_torch_or_execute_model() -> None:
    root = Path(__file__).resolve().parent.parent
    script = """
import sys
from qkvs.harness import load_run, print_summary, status, validate_config_file
report = validate_config_file('configs/harness_smoke.json')
assert report['valid']
result = load_run('artifacts/results/harness-smoke')
assert result['measurements']
assert status('artifacts/results/harness-smoke')['run_id'] == 'harness-smoke'
print_summary(result)
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
