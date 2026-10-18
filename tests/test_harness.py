from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qkvs.harness import (
    _Progress,
    _progress_counts,
    _progress_line,
    _tokenized_text_streams,
    load_run,
    run_experiment,
    status,
    validate_config,
    validate_config_file,
)


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


def test_live_progress_uses_fixed_counter_and_space_filled_bar() -> None:
    line = _progress_line(3, 384, "heatfield / stream-002", 65)
    assert line.startswith("[   3 / 384 ] [#")
    bar = line.split("] [", maxsplit=1)[1].split("]", maxsplit=1)[0]
    assert len(bar) == 40
    assert set(bar) <= {"#", " "}
    assert line.endswith("heatfield / stream-002 : 00:01:05")


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


def test_checked_in_configurations_are_valid_and_pilot_is_approved() -> None:
    smoke = validate_config_file("configs/harness_smoke.json")
    heatfield = validate_config_file("configs/heatfield_initial.json")
    learning_study = validate_config_file("configs/heatfield_learning_study.json")
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
    assert "{run,validate,status,summarize}" in completed.stdout
