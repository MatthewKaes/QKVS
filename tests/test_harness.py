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
    _predictive_execution_version,
    _predictive_measurement_key,
    _Progress,
    _progress_bar,
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


def test_checked_in_configurations_preserve_prior_runs_and_draft_full_qkvs_training() -> None:
    smoke = validate_config_file("configs/harness_smoke.json")
    heatfield = validate_config_file("configs/heatfield_initial.json")
    pilot_path = Path("configs/heatfield_learning_study.json")
    followup_path = Path("configs/heatfield_learning_followup.json")
    hard_aligned_path = Path("configs/heatfield_hard_aligned_study.json")
    corrected_path = Path("configs/heatfield_hard_aligned_corrected_study.json")
    full_qkvs_path = Path("configs/heatfield_full_qkvs_learning_study.json")
    learning_study = validate_config_file(pilot_path)
    followup = validate_config_file(followup_path)
    hard_aligned = validate_config_file(hard_aligned_path)
    corrected = validate_config_file(corrected_path)
    full_qkvs = validate_config_file(full_qkvs_path)
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
    assert "{run,validate,status,summarize}" in completed.stdout
