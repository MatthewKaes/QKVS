from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

from qkvs.metrics import (
    SNAPSHOT_OVERLAP_THRESHOLDS,
    summarize_measurements,
    summarize_snapshot_frontiers,
    summarize_snapshot_trial_set,
    summarize_training_alignment,
)

_SOURCE_FILES = (
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "uv.lock",
    "qkvs/__init__.py",
    "qkvs/__main__.py",
    "qkvs/harness.py",
    "qkvs/model.py",
    "qkvs/selection.py",
    "qkvs/metrics.py",
)
_PROGRESS_WIDTH = 24
_SPINNER_FRAMES = ("|", "/", "-", "\\")
_ERASE_LINE = "\r\x1b[2K"
_PREDICTIVE_EXECUTION_VERSIONS = {
    "rolling": "qwen2-eager-rolling-v1",
    "fixed_offsets": "qwen2-eager-fixed-offsets-v1",
    "full_qkv": "qwen2-eager-full-qkv-v1",
    "heatfield_untrained": "qwen2-eager-untrained-heatfield-v2",
    "heatfield": "qwen2-eager-trained-heatfield-v2",
}
# The completed absolute-mass follow-up predates explicit execution versions. Its three frozen
# reader controls have exact per-token/access parity with the prior pilot; untrained S is excluded
# because that run did not yet guarantee selector placement on the requested device.
_LEGACY_BASELINE_SOURCE = {
    "git_revision": "b63e9b54f11543c571298de215bff567e1529efb",
    "model_sha256": "e0873329d2eda3db219b7cff463016a987a6906c16d83b4a0425097ee22ad7bd",
    "selection_sha256": "ba195153672aa9b77a50b441bf8c26453c5da788f2207e7387b1c92783bb0ac3",
}
_LEGACY_REUSABLE_POLICIES = {"rolling", "fixed_offsets", "full_qkv"}
_SNAPSHOT_STUDY_KIND = "unconstrained_snapshot_compressibility"
_SNAPSHOT_EXECUTION_VERSION = "qwen2-h1a-current-token-v1"
_SNAPSHOT_HISTOGRAM_BINS = 20
_SNAPSHOT_PROBABILITY_TOLERANCE = 1e-6
_SNAPSHOT_EXPORT_FORMAT = "qkvs-h1a-csv-v1"


def _progress_bar(current: float, total: int, *, width: int = _PROGRESS_WIDTH) -> str:
    if total <= 0:
        return "█" * width
    bounded = max(0, min(current, total))
    thirds = int(width * 3 * bounded / total + 0.5)
    full, partial = divmod(thirds, 3)
    partial_glyph = ("", "▒", "▓")[partial]
    return "█" * full + partial_glyph + " " * (width - full - bool(partial))


def _progress_line(
    current: int,
    total: int,
    job: str,
    elapsed_seconds: float,
    *,
    spinner: str = "|",
    columns: int = 80,
    bar_current: float | None = None,
) -> str:
    counter_width = len(str(total)) + 1
    elapsed = max(0, int(elapsed_seconds))
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    prefix = (
        f"{spinner} [{current:>{counter_width}} / {total} ] "
        f"[{_progress_bar(current if bar_current is None else bar_current, total)}] "
        f"{hours:02d}:{minutes:02d}:{seconds:02d} "
    )
    maximum = max(1, columns - 1)
    available = maximum - len(prefix)
    if available <= 0:
        return prefix[:maximum]
    if len(job) > available:
        job = job[: max(0, available - 1)] + "~"
    return prefix + job


class _Progress:
    """Render a one-row spinner while preserving durable unit accounting."""

    def __init__(
        self,
        *,
        completed: int,
        total: int,
        stream: TextIO | None = None,
        interval_seconds: float = 1.0,
    ) -> None:
        self.completed = completed
        self.total = total
        self.stream = sys.stdout if stream is None else stream
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.interval_seconds = interval_seconds
        self.active_job: str | None = None
        self.job_fraction = 1.0
        self.running = False
        self.advances = False
        self.started = time.monotonic()
        self.frame = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = (
            threading.Thread(target=self._heartbeat, daemon=True) if self.interactive else None
        )
        if self.thread is not None:
            self.thread.start()

    def _columns(self) -> int:
        try:
            return os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError):
            return 80

    def _heartbeat(self) -> None:
        while not self.stop.wait(self.interval_seconds):
            self.render()

    def _current(self) -> int:
        if not self.running or not self.advances:
            return self.completed
        return min(self.completed + 1, self.total)

    def _bar_current(self) -> float:
        if not self.running or not self.advances:
            return float(self.completed)
        return min(self.completed + self.job_fraction, self.total)

    def render(self) -> None:
        if not self.interactive:
            return
        with self.lock:
            spinner = _SPINNER_FRAMES[self.frame % len(_SPINNER_FRAMES)]
            self.frame += 1
            line = _progress_line(
                self._current(),
                self.total,
                self.active_job or "complete",
                time.monotonic() - self.started,
                spinner=spinner,
                columns=self._columns(),
                bar_current=self._bar_current(),
            )
            self.stream.write(_ERASE_LINE + line)
            self.stream.flush()

    def start_job(self, job: str, *, advances: bool = True, fraction: float = 1.0) -> None:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("job progress fraction must lie in [0, 1]")
        with self.lock:
            if self.running:
                raise RuntimeError("cannot start progress while another job is active")
            self.active_job = job
            self.running = True
            self.advances = advances
            self.job_fraction = fraction
        if self.interactive:
            self.render()
        else:
            print(
                _progress_line(
                    self._current(),
                    self.total,
                    job,
                    time.monotonic() - self.started,
                    spinner=">",
                    columns=self._columns(),
                    bar_current=self._bar_current(),
                ),
                file=self.stream,
                flush=True,
            )

    def update_job(self, job: str, *, fraction: float) -> None:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("job progress fraction must lie in [0, 1]")
        with self.lock:
            if not self.running:
                raise RuntimeError("cannot update progress without an active job")
            self.active_job = job
            self.job_fraction = fraction
        self.render()

    def complete_job(self) -> None:
        with self.lock:
            if not self.running:
                raise RuntimeError("cannot complete progress without an active job")
            if self.advances:
                self.completed += 1
            self.running = False
            self.advances = False
            self.job_fraction = 1.0
            self.active_job = None
        self.render()

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        with self.lock:
            self.running = False
            self.advances = False
        if self.interactive:
            self.render()
            self.stream.write("\n")
            self.stream.flush()


def _progress_counts(result: dict[str, Any]) -> tuple[int, int, dict[str, tuple[int, int]]]:
    specification = result["config"]
    if specification.get("study", {}).get("kind") == _SNAPSHOT_STUDY_KIND:
        positions = specification["study"]["positions"]
        fractions = specification["study"]["sampling_percentages"]
        prefix_total = len(positions)
        trial_total = len(positions) * len(fractions)
        phases = {
            "snapshot_prefixes": (
                min(prefix_total, len(result.get("snapshot_prefixes", {}))),
                prefix_total,
            ),
            "snapshot_trial_sets": (
                min(trial_total, len(result.get("snapshot_observations", []))),
                trial_total,
            ),
        }
        return (
            sum(completed for completed, _ in phases.values()),
            sum(total for _, total in phases.values()),
            phases,
        )
    streams = result["streams"]
    dense_total = len(streams) if specification["model"]["backend"] == "huggingface" else 0
    dense_completed = min(dense_total, len(result.get("dense_prefill", {})))

    trains_selector = any(
        condition["policy"] == "heatfield" for condition in specification["selection"]["conditions"]
    )
    training = specification.get("training", {})
    checkpoint_interval = training.get("checkpoint_interval", 1)
    training_steps = training.get("steps", 0) if trains_selector else 0
    completed_steps = result.get("training", {}).get("completed_steps", 0)
    training_total = (
        (training_steps + checkpoint_interval - 1) // checkpoint_interval if training_steps else 0
    )
    training_completed = min(
        training_total,
        (
            (completed_steps + checkpoint_interval - 1) // checkpoint_interval
            if completed_steps
            else 0
        ),
    )

    evaluation_streams = [
        stream for stream in streams if stream.get("split", "evaluation") == "evaluation"
    ]
    evaluation_total = len(specification["selection"]["conditions"]) * len(evaluation_streams)
    evaluation_completed = min(
        evaluation_total,
        len(
            {
                (record["condition"], record["stream_id"])
                for record in (
                    *result["measurements"],
                    *result.get("measurement_references", []),
                )
            }
        ),
    )
    phases = {
        "dense_prefill": (dense_completed, dense_total),
        "training_checkpoints": (training_completed, training_total),
        "evaluation": (evaluation_completed, evaluation_total),
    }
    return (
        sum(completed for completed, _ in phases.values()),
        sum(total for _, total in phases.values()),
        phases,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _module_state_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _predictive_execution_version(result: dict[str, Any], policy: str) -> str | None:
    versions = result.get("predictive_execution_versions", {})
    if isinstance(versions, dict) and isinstance(versions.get(policy), str):
        return versions[policy]
    source = result.get("source_identity", {})
    files = source.get("files", {})
    if (
        policy in _LEGACY_REUSABLE_POLICIES
        and source.get("git_revision") == _LEGACY_BASELINE_SOURCE["git_revision"]
        and files.get("qkvs/model.py") == _LEGACY_BASELINE_SOURCE["model_sha256"]
        and files.get("qkvs/selection.py") == _LEGACY_BASELINE_SOURCE["selection_sha256"]
    ):
        return _PREDICTIVE_EXECUTION_VERSIONS[policy]
    return None


def _condition_predictive_identity(
    result: dict[str, Any],
    condition: dict[str, Any],
) -> dict[str, Any] | None:
    policy = condition["policy"]
    version = _predictive_execution_version(result, policy)
    if version is None or policy == "heatfield":
        return None
    specification = result["config"]
    selection = specification["selection"]
    identity: dict[str, Any] = {
        "policy": policy,
        "predictive_execution_version": version,
    }
    if policy == "fixed_offsets":
        identity.update(
            {
                "offsets": condition["offsets"],
                "window_width": selection["window_width"],
                "historical_token_budget": specification["attention"]["historical_token_budget"],
            }
        )
    elif policy == "heatfield_untrained":
        state_sha256 = result.get("untrained_policy_sha256")
        if not isinstance(state_sha256, str):
            return None
        identity.update(
            {
                "state_sha256": state_sha256,
                "proposal_count": selection["proposal_count"],
                "window_width": selection["window_width"],
                "historical_token_budget": specification["attention"]["historical_token_budget"],
                "gamma": selection["gamma"],
                "beta": selection["beta"],
                "learned": selection["learned"],
            }
        )
    return identity


def _predictive_measurement_key(
    result: dict[str, Any],
    condition: dict[str, Any],
    stream: dict[str, Any],
) -> str | None:
    condition_identity = _condition_predictive_identity(result, condition)
    if condition_identity is None:
        return None
    stream_id = stream["stream_id"]
    dense_prefill = result.get("dense_prefill", {}).get(stream_id)
    return _canonical_sha256(
        {
            "condition": condition_identity,
            "model": result["model_identity"],
            "evaluation_stream": {
                "stream_id": stream_id,
                "token_sha256": stream["token_sha256"],
                "support_positions_sha256": _canonical_sha256(stream["support_positions"]),
            },
            "dense_prefill_identity": dense_prefill.get("identity") if dense_prefill else None,
            "attention": {
                "rolling_tokens": result["config"]["attention"]["rolling_tokens"],
                "activation_tokens": result["config"]["attention"]["activation_tokens"],
            },
            "evaluation": {
                "metric": result["config"]["evaluation"]["metric"],
                "proposal_trace_interval": result["config"]["evaluation"][
                    "proposal_trace_interval"
                ],
            },
            "numerical_execution": {
                "device": result["device"],
                "python": result["environment"].get("python"),
                "platform": result["environment"].get("platform"),
                "torch": result["environment"].get("torch"),
                "torch_cuda": result["environment"].get("torch_cuda"),
            },
        }
    )


def _predictive_measurement_sha256(measurement: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "condition": measurement["condition"],
                "policy": measurement["policy"],
                "stream_id": measurement["stream_id"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for score in measurement["token_scores"]:
        digest.update(b"\0")
        digest.update(
            json.dumps(
                score,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        )
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError("experiment configuration must be a JSON object")
    return value


def _measurement_reference(
    *,
    measurement: dict[str, Any],
    predictive_key: str,
    source: dict[str, Any],
    source_path: Path,
    output: Path,
    source_artifact_sha256: str,
) -> dict[str, Any]:
    return {
        "condition": measurement["condition"],
        "policy": measurement["policy"],
        "stream_id": measurement["stream_id"],
        "predictive_key": predictive_key,
        "predictive_measurement_sha256": _predictive_measurement_sha256(measurement),
        "source_result": os.path.relpath(source_path, output.parent),
        "source_run_id": source["run_id"],
        "source_revision": source["source_identity"]["git_revision"],
        "source_artifact_sha256": source_artifact_sha256,
        "performance_reused": False,
    }


def _measurement_from_reference(
    reference: dict[str, Any],
    *,
    result_path: Path,
    source_cache: dict[Path, tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    source_path = (result_path.parent / reference["source_result"]).resolve()
    if not source_path.is_file():
        raise ValueError("reused baseline source artifact is missing")
    if source_path not in source_cache:
        source_cache[source_path] = _sha256(source_path), _load_json(source_path)
    source_artifact_sha256, source = source_cache[source_path]
    if source_artifact_sha256 != reference["source_artifact_sha256"]:
        raise ValueError("reused baseline source artifact identity does not match")
    if not source.get("complete") or source.get("run_id") != reference["source_run_id"]:
        raise ValueError("reused baseline source run is incomplete or mismatched")
    candidates = [
        measurement
        for measurement in source.get("measurements", [])
        if measurement["condition"] == reference["condition"]
        and measurement["stream_id"] == reference["stream_id"]
    ]
    if len(candidates) != 1:
        raise ValueError("reused baseline source measurement is missing or ambiguous")
    source_measurement = candidates[0]
    if (
        source_measurement["policy"] != reference["policy"]
        or _predictive_measurement_sha256(source_measurement)
        != reference["predictive_measurement_sha256"]
    ):
        raise ValueError("reused baseline predictive measurement identity does not match")
    measurement = {key: value for key, value in source_measurement.items() if key != "performance"}
    measurement["predictive_reuse"] = {
        "source_result": reference["source_result"],
        "source_run_id": reference["source_run_id"],
        "source_revision": reference["source_revision"],
        "source_artifact_sha256": reference["source_artifact_sha256"],
        "predictive_measurement_sha256": reference["predictive_measurement_sha256"],
        "performance_reused": False,
    }
    return measurement


def _resolved_measurements(result: dict[str, Any], result_path: Path) -> list[dict[str, Any]]:
    measurements = list(result["measurements"])
    source_cache: dict[Path, tuple[str, dict[str, Any]]] = {}
    measurements.extend(
        _measurement_from_reference(
            reference,
            result_path=result_path,
            source_cache=source_cache,
        )
        for reference in result.get("measurement_references", [])
    )
    measurements.sort(key=lambda measurement: (measurement["condition"], measurement["stream_id"]))
    units = [(measurement["condition"], measurement["stream_id"]) for measurement in measurements]
    if len(units) != len(set(units)):
        raise ValueError("run contains duplicate measured or reused evaluation units")
    return measurements


def _discover_reusable_measurements(
    result: dict[str, Any],
    *,
    output: Path,
) -> dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]:
    conditions = {
        condition["name"]: condition for condition in result["config"]["selection"]["conditions"]
    }
    streams = {
        stream["stream_id"]: stream
        for stream in result["streams"]
        if stream.get("split") == "evaluation"
    }
    target_keys = {
        (condition_name, stream_id): _predictive_measurement_key(result, condition, stream)
        for condition_name, condition in conditions.items()
        if condition["policy"] != "heatfield"
        for stream_id, stream in streams.items()
    }
    reusable: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    candidates = sorted(
        (path for path in output.parent.parent.glob("*/result.json") if path != output),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for source_path in candidates:
        source = _load_json(source_path)
        if not source.get("complete"):
            continue
        source_conditions = {
            condition["name"]: condition
            for condition in source.get("config", {}).get("selection", {}).get("conditions", [])
        }
        source_streams = {stream["stream_id"]: stream for stream in source.get("streams", [])}
        source_artifact_sha256 = None
        for measurement in source.get("measurements", []):
            unit = (measurement["condition"], measurement["stream_id"])
            if unit in reusable or target_keys.get(unit) is None:
                continue
            source_condition = source_conditions.get(measurement["condition"])
            source_stream = source_streams.get(measurement["stream_id"])
            if source_condition is None or source_stream is None:
                continue
            source_key = _predictive_measurement_key(
                source,
                source_condition,
                source_stream,
            )
            if source_key != target_keys[unit]:
                continue
            if source_artifact_sha256 is None:
                source_artifact_sha256 = _sha256(source_path)
            reference = _measurement_reference(
                measurement=measurement,
                predictive_key=source_key,
                source=source,
                source_path=source_path,
                output=output,
                source_artifact_sha256=source_artifact_sha256,
            )
            reused_measurement = {
                key: value for key, value in measurement.items() if key != "performance"
            }
            reused_measurement["predictive_reuse"] = {
                key: reference[key]
                for key in (
                    "source_result",
                    "source_run_id",
                    "source_revision",
                    "source_artifact_sha256",
                    "predictive_measurement_sha256",
                    "performance_reused",
                )
            }
            reusable[unit] = reference, reused_measurement
        if len(reusable) == sum(key is not None for key in target_keys.values()):
            break
    return reusable


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_config(specification: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if specification.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if not isinstance(specification.get("experiment"), str):
        errors.append("experiment must be a string")
    status = specification.get("status")
    if status not in {"harness_verification", "draft_not_authorized", "approved"}:
        errors.append("status is not recognized")

    run = specification.get("run", {})
    if not isinstance(run.get("id"), str) or not run.get("id"):
        errors.append("run.id must be a nonempty string")
    if not isinstance(run.get("seed"), int):
        errors.append("run.seed must be an integer")

    model = specification.get("model", {})
    backend = model.get("backend")
    if backend not in {"synthetic", "huggingface"}:
        errors.append("model.backend must be synthetic or huggingface")
    if model.get("frozen") is not True:
        errors.append("model.frozen must be true")
    if backend == "synthetic":
        for field in ("vocab_size", "width", "layers", "heads", "seed"):
            if not _positive_integer(model.get(field)):
                errors.append(f"model.{field} must be a positive integer")
        width = model.get("width")
        heads = model.get("heads")
        if _positive_integer(width) and _positive_integer(heads) and width % heads:
            errors.append("model.width must be divisible by model.heads")
    elif backend == "huggingface":
        for field in ("id", "revision"):
            if not isinstance(model.get(field), str) or not model.get(field):
                errors.append(f"model.{field} must be a pinned nonempty string")
        revision = model.get("revision")
        if isinstance(revision, str) and not re.fullmatch(r"[0-9a-f]{40}", revision):
            errors.append("model.revision must be a full 40-character Git commit")
        dtype = model.get("dtype")
        if dtype is not None and dtype not in {"float32", "bfloat16"}:
            errors.append("model.dtype must be float32 or bfloat16")
        for field in ("tokenizer_id", "tokenizer_revision"):
            if field in model and (not isinstance(model[field], str) or not model[field]):
                errors.append(f"model.{field} must be a nonempty string")
        tokenizer_revision = model.get("tokenizer_revision")
        if tokenizer_revision is not None and not re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision):
            errors.append("model.tokenizer_revision must be a full 40-character Git commit")

    data = specification.get("data", {})
    data_kind = data.get("kind")
    if data_kind == "synthetic" or data_kind == "long_context_reading":
        for field in ("streams", "sequence_tokens", "seed"):
            if not _positive_integer(data.get(field)):
                errors.append(f"data.{field} must be a positive integer")
    elif data_kind == "tokenized_text":
        source = data.get("source", {})
        if not isinstance(source.get("path"), str) or not source.get("path"):
            errors.append("data.source.path must be a nonempty string")
        source_hash = source.get("sha256")
        if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
            errors.append("data.source.sha256 must be a 64-character lowercase digest")
        if "url" in source and (
            not isinstance(source["url"], str) or not source["url"].startswith("https://")
        ):
            errors.append("data.source.url must be an HTTPS URL")
        if "bytes" in source and not _positive_integer(source["bytes"]):
            errors.append("data.source.bytes must be a positive integer")
        boundaries = data.get("boundaries")
        if not isinstance(boundaries, list) or not boundaries:
            errors.append("data.boundaries must be a nonempty list")
            boundaries = []
        boundary_ids: list[str] = []
        for index, boundary in enumerate(boundaries):
            if not isinstance(boundary, dict):
                errors.append(f"data.boundaries[{index}] must be an object")
                continue
            stream_id = boundary.get("id")
            start = boundary.get("start_token")
            stop = boundary.get("stop_token")
            if not isinstance(stream_id, str) or not stream_id:
                errors.append(f"data.boundaries[{index}].id must be nonempty")
            else:
                boundary_ids.append(stream_id)
            if not isinstance(start, int) or isinstance(start, bool) or start < 0:
                errors.append(f"data.boundaries[{index}].start_token must be nonnegative")
            if not _positive_integer(stop):
                errors.append(f"data.boundaries[{index}].stop_token must be positive")
            if isinstance(start, int) and isinstance(stop, int) and stop - start < 2:
                errors.append(f"data.boundaries[{index}] must contain at least two tokens")
            if "split" in boundary and boundary["split"] not in {"train", "evaluation"}:
                errors.append(f"data.boundaries[{index}].split must be train or evaluation")
        if len(boundary_ids) != len(set(boundary_ids)):
            errors.append("data boundary ids must be unique")
    else:
        errors.append("data.kind is unsupported")

    study = specification.get("study")
    if study is not None:
        if not isinstance(study, dict) or study.get("kind") != _SNAPSHOT_STUDY_KIND:
            errors.append("study.kind is unsupported")
            study = {}
        positions = study.get("positions")
        if (
            not isinstance(positions, list)
            or not positions
            or any(not _positive_integer(position) for position in positions)
            or len(positions) != len(set(positions))
        ):
            errors.append("snapshot study positions must be unique positive integers")
            positions = []
        percentages = study.get("sampling_percentages")
        if (
            not isinstance(percentages, list)
            or not percentages
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 < float(value) <= 100.0
                for value in percentages
            )
            or len(percentages) != len(set(percentages))
        ):
            errors.append("snapshot sampling percentages must be unique values in (0, 100]")
            percentages = []
        elif 100 not in percentages:
            errors.append("snapshot sampling percentages must include the 100% dense control")
        if not _positive_integer(study.get("block_width")):
            errors.append("snapshot block_width must be a positive integer")
        if not _positive_integer(study.get("trials_per_fraction")):
            errors.append("snapshot trials_per_fraction must be a positive integer")
        trial_start = study.get("trial_start")
        if not isinstance(trial_start, int) or isinstance(trial_start, bool) or trial_start < 0:
            errors.append("snapshot trial_start must be a nonnegative integer")
        cohorts = study.get("position_cohorts")
        if cohorts is not None:
            if (
                not isinstance(cohorts, dict)
                or not cohorts
                or any(
                    not isinstance(name, str)
                    or not name
                    or not isinstance(cohort, list)
                    or not cohort
                    or any(not _positive_integer(position) for position in cohort)
                    for name, cohort in cohorts.items()
                )
            ):
                errors.append("snapshot position cohorts must be named nonempty position lists")
            else:
                cohort_positions = [position for cohort in cohorts.values() for position in cohort]
                if len(cohort_positions) != len(set(cohort_positions)):
                    errors.append("snapshot position cohorts must not overlap")
                if set(cohort_positions) != set(positions):
                    errors.append("snapshot position cohorts must partition study.positions")
        expected_candidate_passes = study.get("expected_candidate_passes")
        if expected_candidate_passes is not None:
            calculated_candidate_passes = (
                len(positions) * ((len(percentages) - 1) * study.get("trials_per_fraction", 0) + 1)
                if positions and percentages and _positive_integer(study.get("trials_per_fraction"))
                else None
            )
            if (
                not _positive_integer(expected_candidate_passes)
                or expected_candidate_passes != calculated_candidate_passes
            ):
                errors.append("snapshot expected candidate passes do not match the study grid")
        reuse_dense_snapshots_from = study.get("reuse_dense_snapshots_from")
        if reuse_dense_snapshots_from is not None and (
            not isinstance(reuse_dense_snapshots_from, str) or not reuse_dense_snapshots_from
        ):
            errors.append("snapshot dense reuse source must be a nonempty path")
        if study.get("sampling_algorithm") != "uniform-aligned-blocks-v1":
            errors.append("snapshot sampling_algorithm is unsupported")
        tolerances = study.get("fidelity_tolerances_nats")
        if (
            not isinstance(tolerances, list)
            or not tolerances
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or float(value) <= 0.0
                for value in tolerances
            )
            or len(tolerances) != len(set(tolerances))
        ):
            errors.append("snapshot fidelity tolerances must be unique positive values")
        parity = study.get("dense_parity_tolerance", {})
        if any(
            not isinstance(parity.get(field), (int, float))
            or isinstance(parity.get(field), bool)
            or parity[field] < 0.0
            for field in ("logit_atol", "loss_atol")
        ):
            errors.append("snapshot dense parity tolerances must be nonnegative")
        if any(
            field in specification for field in ("attention", "selection", "training", "evaluation")
        ):
            errors.append("snapshot studies must not define QKVS attention or selector controls")
        boundaries = data.get("boundaries", []) if data_kind == "tokenized_text" else []
        if len(boundaries) != 1:
            errors.append("snapshot studies require exactly one tokenized-text boundary")
        elif positions and isinstance(boundaries[0], dict):
            start = boundaries[0].get("start_token")
            stop = boundaries[0].get("stop_token")
            if isinstance(start, int) and isinstance(stop, int):
                stream_length = stop - start
                if max(positions) >= stream_length - 1:
                    errors.append("snapshot positions must retain a next-token target")
        execution = specification.get("execution", {})
        devices = execution.get("allowed_devices")
        if (
            not isinstance(devices, list)
            or not devices
            or any(device not in {"cpu", "cuda"} for device in devices)
        ):
            errors.append("execution.allowed_devices must contain cpu and/or cuda")
        if not isinstance(execution.get("output_root"), str):
            errors.append("execution.output_root must be a string")
        if not _positive_integer(execution.get("prefill_chunk_tokens")):
            errors.append("execution.prefill_chunk_tokens must be a positive integer")
        reasons: list[str] = []
        if status not in {"harness_verification", "approved"}:
            reasons.append("configuration is not authorized for harness execution")
        if backend != "huggingface":
            reasons.append("snapshot studies require the Hugging Face Qwen2 reader")
        if backend == "huggingface" and model.get("dtype") not in {"float32", "bfloat16"}:
            reasons.append("a Hugging Face execution dtype has not been frozen")
        if data_kind != "tokenized_text":
            reasons.append("snapshot studies require identity-pinned tokenized_text data")
        return {
            "valid": not errors,
            "runnable": not errors and not reasons,
            "errors": errors,
            "not_runnable_reasons": reasons,
            "experiment": specification.get("experiment"),
            "status": status,
        }

    attention = specification.get("attention", {})
    for field in ("rolling_tokens", "activation_tokens", "historical_token_budget"):
        if not _positive_integer(attention.get(field)):
            errors.append(f"attention.{field} must be a positive integer")
    if (
        _positive_integer(attention.get("rolling_tokens"))
        and _positive_integer(attention.get("activation_tokens"))
        and attention["activation_tokens"] < attention["rolling_tokens"]
    ):
        errors.append("attention.activation_tokens must be at least rolling_tokens")
    activation_tokens = attention.get("activation_tokens")
    if _positive_integer(activation_tokens):
        if data_kind in {"synthetic", "long_context_reading"} and _positive_integer(
            data.get("sequence_tokens")
        ):
            if data["sequence_tokens"] <= activation_tokens:
                errors.append("data.sequence_tokens must extend beyond attention.activation_tokens")
        elif data_kind == "tokenized_text":
            for index, boundary in enumerate(data.get("boundaries", [])):
                if not isinstance(boundary, dict):
                    continue
                start = boundary.get("start_token")
                stop = boundary.get("stop_token")
                if isinstance(start, int) and isinstance(stop, int):
                    if stop - start <= activation_tokens:
                        errors.append(
                            f"data.boundaries[{index}] must extend beyond "
                            "attention.activation_tokens"
                        )

    selection = specification.get("selection", {})
    proposal_count = selection.get("proposal_count")
    window_width = selection.get("window_width")
    if not _positive_integer(proposal_count):
        errors.append("selection.proposal_count must be a positive integer")
    if not _positive_integer(window_width):
        errors.append("selection.window_width must be a positive integer")
    for control_name in ("gamma", "beta"):
        control = selection.get(control_name, {})
        mode = control.get("mode")
        if mode not in {"disabled", "fixed", "learned"}:
            errors.append(f"selection.{control_name}.mode is not recognized")
        if mode == "fixed":
            value = control.get("value")
            if not isinstance(value, (int, float)):
                errors.append(f"selection.{control_name}.value must be numeric")
            elif control_name == "gamma" and not 0.0 <= float(value) < 1.0:
                errors.append("fixed gamma must lie in [0, 1)")
            elif control_name == "beta" and float(value) <= 0.0:
                errors.append("fixed beta must be positive")

    conditions = selection.get("conditions")
    names: list[str] = []
    if not isinstance(conditions, list) or len(conditions) < 2:
        errors.append("selection.conditions must contain at least two conditions")
        conditions = []
    for index, condition in enumerate(conditions):
        if not isinstance(condition, dict):
            errors.append(f"selection.conditions[{index}] must be an object")
            continue
        name = condition.get("name")
        policy = condition.get("policy")
        if not isinstance(name, str) or not name:
            errors.append(f"selection.conditions[{index}].name must be nonempty")
        else:
            names.append(name)
        if policy not in {
            "rolling",
            "fixed_offsets",
            "full_qkv",
            "heatfield",
            "heatfield_untrained",
        }:
            errors.append(f"selection.conditions[{index}].policy is unsupported")
        if policy == "fixed_offsets":
            offsets = condition.get("offsets")
            if (
                not isinstance(offsets, list)
                or not offsets
                or not _positive_integer(proposal_count)
                or len(offsets) > proposal_count
                or any(not _positive_integer(offset) for offset in offsets)
            ):
                errors.append("fixed_offsets requires positive offsets within proposal capacity")
    if len(names) != len(set(names)):
        errors.append("condition names must be unique")
    condition_objects = [condition for condition in conditions if isinstance(condition, dict)]
    if not any(condition.get("policy") == "rolling" for condition in condition_objects):
        errors.append("a rolling-only condition is mandatory")
    historical_token_budget = attention.get("historical_token_budget")
    if (
        any(condition.get("policy") == "fixed_offsets" for condition in condition_objects)
        and _positive_integer(window_width)
        and _positive_integer(historical_token_budget)
        and window_width > historical_token_budget
    ):
        errors.append(
            "selection.window_width cannot exceed the historical token budget for fixed_offsets"
        )

    evaluation = specification.get("evaluation", {})
    if evaluation.get("metric") != "next_token_cross_entropy":
        errors.append("evaluation.metric must be next_token_cross_entropy")
    if evaluation.get("paired_conditions") is not True:
        errors.append("evaluation.paired_conditions must be true")
    if evaluation.get("baseline") not in names:
        errors.append("evaluation.baseline must name a configured condition")
    if not _positive_integer(evaluation.get("bootstrap_samples")):
        errors.append("evaluation.bootstrap_samples must be positive")
    if not _positive_integer(evaluation.get("proposal_trace_interval")):
        errors.append("evaluation.proposal_trace_interval must be a positive integer")
    comparisons = evaluation.get("comparisons", [])
    comparison_names: list[str] = []
    if not isinstance(comparisons, list):
        errors.append("evaluation.comparisons must be a list")
        comparisons = []
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, dict):
            errors.append(f"evaluation.comparisons[{index}] must be an object")
            continue
        name = comparison.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"evaluation.comparisons[{index}].name must be nonempty")
        else:
            comparison_names.append(name)
        if comparison.get("baseline") not in names or comparison.get("condition") not in names:
            errors.append(f"evaluation.comparisons[{index}] must name configured conditions")
    if len(comparison_names) != len(set(comparison_names)):
        errors.append("evaluation comparison names must be unique")
    if set(comparison_names) & set(names):
        errors.append("evaluation comparison names must not shadow conditions")

    execution = specification.get("execution", {})
    devices = execution.get("allowed_devices")
    if (
        not isinstance(devices, list)
        or not devices
        or any(device not in {"cpu", "cuda"} for device in devices)
    ):
        errors.append("execution.allowed_devices must contain cpu and/or cuda")
    if not isinstance(execution.get("output_root"), str):
        errors.append("execution.output_root must be a string")
    if not _positive_integer(execution.get("prefill_chunk_tokens")):
        errors.append("execution.prefill_chunk_tokens must be a positive integer")

    reasons: list[str] = []
    if status not in {"harness_verification", "approved"}:
        reasons.append("configuration is not authorized for harness execution")
    if backend == "huggingface" and model.get("dtype") not in {"float32", "bfloat16"}:
        reasons.append("a Hugging Face execution dtype has not been frozen")
    if backend == "huggingface" and data_kind != "tokenized_text":
        reasons.append("Hugging Face execution requires identity-pinned tokenized_text data")
    if backend == "synthetic" and data_kind != "synthetic":
        reasons.append("the synthetic reader requires synthetic data")
    if data_kind == "long_context_reading":
        reasons.append(
            "long_context_reading is a draft data description, not an implemented loader"
        )
    policies = {condition.get("policy") for condition in condition_objects}
    has_heatfield = bool(policies & {"heatfield", "heatfield_untrained"})
    learned = selection.get("learned")
    training = specification.get("training")
    if has_heatfield:
        if backend != "huggingface":
            reasons.append("learned heat-field training requires the Hugging Face Qwen2 reader")
        if "heatfield" not in policies:
            reasons.append("learned heat-field study requires a trained heatfield condition")
        if "heatfield_untrained" not in policies:
            reasons.append("learned heat-field study requires an untrained heatfield control")
        if "full_qkv" not in policies:
            reasons.append("learned heat-field study requires an explicit full_qkv reference")
        if data_kind == "tokenized_text":
            boundaries = data.get("boundaries", [])
            train_boundaries = [
                boundary
                for boundary in boundaries
                if isinstance(boundary, dict)
                and boundary.get("split") == "train"
                and isinstance(boundary.get("start_token"), int)
                and isinstance(boundary.get("stop_token"), int)
            ]
            evaluation_boundaries = [
                boundary
                for boundary in boundaries
                if isinstance(boundary, dict)
                and boundary.get("split") == "evaluation"
                and isinstance(boundary.get("start_token"), int)
                and isinstance(boundary.get("stop_token"), int)
            ]
            if not train_boundaries or not evaluation_boundaries:
                reasons.append("learned heat-field study requires train and evaluation boundaries")
            elif any(
                train["start_token"] < evaluation["stop_token"]
                and evaluation["start_token"] < train["stop_token"]
                for train in train_boundaries
                for evaluation in evaluation_boundaries
            ):
                reasons.append("learned heat-field train and evaluation boundaries overlap")
        if selection.get("gamma", {}).get("mode") != "fixed":
            reasons.append("learned heat-field execution currently requires fixed gamma")
        if selection.get("beta", {}).get("mode") != "fixed":
            reasons.append("learned heat-field execution currently requires fixed beta")
        learned_fields = {
            "latent_size": _positive_integer,
            "minimum_spread": lambda value: isinstance(value, (int, float)) and value > 0,
            "maximum_spread": lambda value: isinstance(value, (int, float)) and value > 0,
            "gate_threshold": lambda value: isinstance(value, (int, float)) and 0.0 <= value < 1.0,
        }
        if not isinstance(learned, dict) or any(
            not predicate(learned.get(field)) for field, predicate in learned_fields.items()
        ):
            reasons.append("learned heat-field policy dimensions and bounds are incomplete")
        elif learned["maximum_spread"] < learned["minimum_spread"]:
            reasons.append("learned heat-field maximum_spread is below minimum_spread")
        elif (
            selection.get("beta", {}).get("mode") == "fixed"
            and isinstance(selection["beta"].get("value"), (int, float))
            and selection["beta"]["value"] > 0
            and _positive_integer(window_width)
            and _positive_integer(historical_token_budget)
            and max(
                1,
                round(window_width * learned["minimum_spread"] / selection["beta"]["value"]),
            )
            > historical_token_budget
        ):
            reasons.append("minimum learned window cannot fit the historical token budget")
        training_fields = (
            "steps",
            "checkpoint_interval",
            "rollout_tokens",
            "targets_per_rollout",
        )
        if (
            not isinstance(training, dict)
            or training.get("enabled") is not True
            or any(not _positive_integer(training.get(field)) for field in training_fields)
            or not isinstance(training.get("learning_rate"), (int, float))
            or training.get("learning_rate", 0) <= 0
            or not isinstance(training.get("coverage_weight"), (int, float))
            or training.get("coverage_weight", -1) < 0
            or not isinstance(training.get("utility_weight"), (int, float))
            or training.get("utility_weight", -1) < 0
            or training.get("coverage_weight", 0) + training.get("utility_weight", 0) <= 0
            or training.get("utility_gradient_estimator", "soft_occupancy")
            not in {"soft_occupancy", "hard_multiplicity_ste"}
            or training.get("predictive_continuation", "rolling") not in {"rolling", "qkvs"}
            or (
                training.get("predictive_continuation", "rolling") == "qkvs"
                and training.get("utility_gradient_estimator", "soft_occupancy")
                != "hard_multiplicity_ste"
            )
            or training.get("position_schedule") != "streaming_stratified"
            or training.get("layer_schedule") != "round_robin"
        ):
            reasons.append("learned heat-field training configuration is incomplete")
        elif data_kind == "tokenized_text" and _positive_integer(activation_tokens):
            train_lengths = [
                boundary["stop_token"] - boundary["start_token"]
                for boundary in data.get("boundaries", [])
                if isinstance(boundary, dict)
                and boundary.get("split") == "train"
                and isinstance(boundary.get("start_token"), int)
                and isinstance(boundary.get("stop_token"), int)
                and boundary["stop_token"] - boundary["start_token"] > activation_tokens
            ]
            required_steps = sum(
                (length - 1 - activation_tokens + training["rollout_tokens"] - 1)
                // training["rollout_tokens"]
                for length in train_lengths
            )
            if required_steps and training["steps"] != required_steps:
                reasons.append("training.steps must cover each post-transition stream exactly once")
    elif any(
        selection.get(control_name, {}).get("mode") != "disabled"
        for control_name in ("gamma", "beta")
    ):
        reasons.append("gamma and beta require a learned heat-field condition")
    return {
        "valid": not errors,
        "runnable": not errors and not reasons,
        "errors": errors,
        "not_runnable_reasons": reasons,
        "experiment": specification.get("experiment"),
        "status": status,
    }


def validate_config_file(path: str | Path) -> dict[str, Any]:
    return validate_config(_load_json(path))


def _source_identity(config_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    paths = [root / path for path in _SOURCE_FILES]
    paths.append(config_path.resolve())
    tracked = [path for path in paths if path.is_relative_to(root)]
    relative = [str(path.relative_to(root)) for path in tracked]
    clean = all(
        subprocess.run(
            ["git", "diff", "--quiet", mode, "--", *relative],
            cwd=root,
            check=False,
        ).returncode
        == 0
        for mode in ("HEAD", "--cached")
    )
    return {
        "git_revision": revision,
        "tracked_files_clean": clean,
        "files": {
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path): _sha256(path)
            for path in paths
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def _token_sha256(tokens: tuple[int, ...]) -> str:
    return hashlib.sha256(
        b"".join(token.to_bytes(8, "little", signed=False) for token in tokens)
    ).hexdigest()


def _synthetic_streams(
    specification: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = specification["data"]
    vocab_size = specification["model"]["vocab_size"]
    period = min(max(4, vocab_size // 2), max(4, data["sequence_tokens"] // 2))
    streams = []
    for stream_index in range(data["streams"]):
        generator = random.Random(data["seed"] + stream_index)
        cycle = [generator.randrange(vocab_size) for _ in range(period)]
        tokens = tuple(cycle[index % period] for index in range(data["sequence_tokens"]))
        supports: list[int | None] = []
        for position, target in enumerate(tokens[1:]):
            prior = [index for index in range(position + 1) if tokens[index] == target]
            supports.append(prior[-1] if prior else None)
        token_tuple = tuple(tokens)
        streams.append(
            {
                "stream_id": f"synthetic-{stream_index:03d}",
                "split": "evaluation",
                "tokens": token_tuple,
                "support_positions": tuple(supports),
                "token_sha256": _token_sha256(token_tuple),
            }
        )
    return streams, {
        "kind": "synthetic",
        "seed": data["seed"],
        "stream_count": data["streams"],
        "sequence_tokens": data["sequence_tokens"],
        "stream_token_sha256": {stream["stream_id"]: stream["token_sha256"] for stream in streams},
    }


def _tokenized_text_streams(
    specification: dict[str, Any],
    *,
    config_path: Path,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tokenizer is None:
        raise ValueError("tokenized_text data requires the pretrained reader tokenizer")
    data = specification["data"]
    configured_path = Path(data["source"]["path"])
    source_path = (
        configured_path
        if configured_path.is_absolute()
        else (config_path.parent / configured_path).resolve()
    )
    if not source_path.exists():
        source_url = data["source"].get("url")
        expected_bytes = data["source"].get("bytes")
        if source_url is None or expected_bytes is None:
            raise ValueError("tokenized text source is missing and has no pinned download")
        import urllib.request

        source_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = source_path.with_suffix(source_path.suffix + ".tmp")
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(source_url, timeout=60) as response:
                with temporary.open("wb") as destination:
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > expected_bytes:
                            raise ValueError("pinned tokenized text download exceeds expected size")
                        digest.update(chunk)
                        destination.write(chunk)
            if size != expected_bytes or digest.hexdigest() != data["source"]["sha256"]:
                raise ValueError("pinned tokenized text download identity does not match")
            temporary.replace(source_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    actual_sha256 = _sha256(source_path)
    expected_sha256 = data["source"]["sha256"]
    if actual_sha256 != expected_sha256:
        raise ValueError("tokenized text source does not match data.source.sha256")
    text = source_path.read_text(encoding="utf-8")
    all_tokens = tuple(tokenizer.encode(text, add_special_tokens=False, verbose=False))
    streams: list[dict[str, Any]] = []
    for boundary in data["boundaries"]:
        start = boundary["start_token"]
        stop = boundary["stop_token"]
        if stop > len(all_tokens):
            raise ValueError(f"data boundary {boundary['id']} stops beyond the tokenized source")
        tokens = all_tokens[start:stop]
        streams.append(
            {
                "stream_id": boundary["id"],
                "split": boundary.get("split", "evaluation"),
                "tokens": tokens,
                "support_positions": (None,) * (len(tokens) - 1),
                "token_sha256": _token_sha256(tokens),
            }
        )
    return streams, {
        "kind": "tokenized_text",
        "source_path": data["source"]["path"],
        "source_sha256": actual_sha256,
        "source_url": data["source"].get("url"),
        "source_bytes": source_path.stat().st_size,
        "encoding": "utf-8",
        "add_special_tokens": False,
        "full_source_tokens": len(all_tokens),
        "boundaries": data["boundaries"],
        "stream_token_sha256": {stream["stream_id"]: stream["token_sha256"] for stream in streams},
    }


def _build_streams(
    specification: dict[str, Any],
    *,
    config_path: Path,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kind = specification["data"]["kind"]
    if kind == "synthetic":
        return _synthetic_streams(specification)
    if kind == "tokenized_text":
        return _tokenized_text_streams(
            specification,
            config_path=config_path,
            tokenizer=tokenizer,
        )
    raise ValueError(f"unsupported data loader: {kind}")


def _result_path(
    specification: dict[str, Any],
    *,
    output_root: str | Path | None,
) -> Path:
    root = Path(output_root or specification["execution"]["output_root"])
    return root / specification["run"]["id"] / "result.json"


def _snapshot_fraction_coordinate(percentage: float) -> int:
    return round(float(percentage) * 1_000_000)


def _snapshot_observation_path(
    output: Path,
    *,
    position: int,
    percentage: float,
    trial_start: int,
    trial_count: int,
) -> Path:
    fraction = _snapshot_fraction_coordinate(percentage)
    return (
        output.parent
        / "snapshot_observations"
        / f"position-{position:08d}"
        / (
            f"fraction-{fraction:09d}-trials-{trial_start:06d}-"
            f"{trial_start + trial_count - 1:06d}.json"
        )
    )


def _read_snapshot_observation_sets(
    result: dict[str, Any],
    output: Path,
) -> list[dict[str, Any]]:
    observations = []
    for manifest in result.get("snapshot_observations", []):
        path = (output.parent / manifest["path"]).resolve()
        if not path.is_relative_to(output.parent.resolve()) or not path.is_file():
            raise ValueError("snapshot observation artifact is missing")
        if _sha256(path) != manifest["artifact_sha256"]:
            raise ValueError("snapshot observation artifact identity does not match")
        observation = _load_json(path)
        if (
            observation.get("identity_sha256") != manifest["identity_sha256"]
            or _canonical_sha256(observation.get("identity")) != manifest["identity_sha256"]
            or observation.get("position") != manifest["position"]
            or observation.get("requested_sampling_percentage")
            != manifest["requested_sampling_percentage"]
            or len(observation.get("candidates", [])) != manifest["trial_count"]
        ):
            raise ValueError("snapshot observation manifest does not match its artifact")
        observations.append(observation)
    return observations


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: Iterable[dict[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    row_count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                row_count += 1
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return row_count


def _snapshot_seed_hex(value: Any) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**64:
        raise ValueError("snapshot candidate seed is not an unsigned 64-bit integer")
    return f"0x{value:016x}"


def _gzip_csv(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with source.open("rb") as uncompressed, temporary.open("wb") as compressed:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=compressed,
                mtime=0,
            ) as output:
                shutil.copyfileobj(uncompressed, output)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_snapshot_data(
    path: str | Path,
    output_directory: str | Path,
    *,
    compress_trials: bool = False,
) -> dict[str, Any]:
    """Export a completed snapshot study without model execution or observation recomputation."""

    result_path = resolve_run(path)
    result = _load_json(result_path)
    study = result.get("config", {}).get("study", {})
    if study.get("kind") != _SNAPSHOT_STUDY_KIND:
        raise ValueError("CSV export requires a snapshot study result")
    if not result.get("complete"):
        raise ValueError("CSV export requires a complete snapshot study")
    observations = sorted(
        _read_snapshot_observation_sets(result, result_path),
        key=lambda observation: (
            int(observation["position"]),
            float(observation["requested_sampling_percentage"]),
        ),
    )
    expected_observations = len(study["positions"]) * len(study["sampling_percentages"])
    if len(observations) != expected_observations:
        raise ValueError("snapshot observation count does not match the frozen configuration")
    recomputed_frontiers = summarize_snapshot_frontiers(
        observations,
        fidelity_tolerances=[float(value) for value in study["fidelity_tolerances_nats"]],
    )
    if recomputed_frontiers != result.get("snapshot_summary"):
        raise ValueError("snapshot fidelity frontiers do not match the original observations")

    candidates = [
        candidate for observation in observations for candidate in observation["candidates"]
    ]
    histogram_widths = {len(candidate["sampled_position_histogram"]) for candidate in candidates}
    layer_counts = {len(candidate["sampled_block_count_per_layer"]) for candidate in candidates}
    if len(histogram_widths) != 1 or len(layer_counts) != 1:
        raise ValueError("snapshot candidate dimensions are inconsistent")
    histogram_bins = histogram_widths.pop()
    layer_count = layer_counts.pop()
    stream_id = result["streams"][0]["stream_id"]
    candidate_identities: set[tuple[str, int, float, int]] = set()
    for observation in observations:
        expected_count = (
            1
            if float(observation["requested_sampling_percentage"]) == 100.0
            else study["trials_per_fraction"]
        )
        if len(observation["candidates"]) != expected_count:
            raise ValueError("snapshot candidate count does not match the frozen configuration")
        for candidate in observation["candidates"]:
            identity = (
                stream_id,
                int(observation["position"]),
                float(observation["requested_sampled_fraction"]),
                int(candidate["trial_id"]),
            )
            if identity in candidate_identities:
                raise ValueError("snapshot export contains a duplicate candidate identity")
            candidate_identities.add(identity)

    trial_fields = (
        "stream_id",
        "position",
        "context_length",
        "requested_sampling_percentage",
        "requested_sampled_fraction",
        "realized_sampled_fraction",
        "trial_id",
        "candidate_seed_hex",
        "sampling_algorithm",
        "execution_identity_sha256",
        "sampling_cost_rho",
        "selected_token_occurrences",
        "historical_token_occurrences",
        "current_token_layer_reads",
        "actual_gathered_key_occurrences",
        "actual_gathered_value_occurrences",
        "aggregate_unique_positions",
        "historical_unique_positions",
        "aggregate_block_count",
        *(f"layer_{index:02d}_block_count" for index in range(layer_count)),
        "kl_divergence_nats",
        "probability_distribution_overlap",
        "sparse_next_token_loss",
        "next_token_ce_delta_nats",
        "sparse_top_one_prediction",
        "top_one_agreement",
        "full_logit_max_abs_error",
        "full_loss_abs_error",
        *(f"sampled_position_bin_{index:02d}" for index in range(histogram_bins)),
    )

    def trial_rows() -> Iterable[dict[str, Any]]:
        for observation in observations:
            for candidate in sorted(
                observation["candidates"], key=lambda candidate: int(candidate["trial_id"])
            ):
                row = {
                    "stream_id": stream_id,
                    "position": observation["position"],
                    "context_length": observation["causal_history_length"],
                    "requested_sampling_percentage": observation["requested_sampling_percentage"],
                    "requested_sampled_fraction": candidate["requested_sampled_fraction"],
                    "realized_sampled_fraction": candidate["realized_sampled_fraction"],
                    "trial_id": candidate["trial_id"],
                    "candidate_seed_hex": _snapshot_seed_hex(candidate["candidate_seed"]),
                    "sampling_algorithm": candidate["sampling_algorithm"],
                    "execution_identity_sha256": candidate["execution_identity_sha256"],
                    "sampling_cost_rho": candidate["sampling_cost_rho"],
                    "selected_token_occurrences": candidate["selected_token_occurrences"],
                    "historical_token_occurrences": candidate["historical_token_occurrences"],
                    "current_token_layer_reads": candidate["current_token_layer_reads"],
                    "actual_gathered_key_occurrences": candidate["actual_gathered_key_occurrences"],
                    "actual_gathered_value_occurrences": candidate[
                        "actual_gathered_value_occurrences"
                    ],
                    "aggregate_unique_positions": candidate["aggregate_unique_positions"],
                    "historical_unique_positions": candidate["historical_unique_positions"],
                    "aggregate_block_count": candidate["aggregate_block_count"],
                    "kl_divergence_nats": candidate["kl_divergence_nats"],
                    "probability_distribution_overlap": candidate[
                        "probability_distribution_overlap"
                    ],
                    "sparse_next_token_loss": candidate["sparse_next_token_loss"],
                    "next_token_ce_delta_nats": candidate["next_token_ce_delta_nats"],
                    "sparse_top_one_prediction": candidate["sparse_top_one_prediction"],
                    "top_one_agreement": int(candidate["top_one_agreement"]),
                    "full_logit_max_abs_error": candidate["full_logit_max_abs_error"],
                    "full_loss_abs_error": candidate["full_loss_abs_error"],
                }
                row.update(
                    {
                        f"layer_{index:02d}_block_count": value
                        for index, value in enumerate(candidate["sampled_block_count_per_layer"])
                    }
                )
                row.update(
                    {
                        f"sampled_position_bin_{index:02d}": value
                        for index, value in enumerate(candidate["sampled_position_histogram"])
                    }
                )
                yield row

    reference_by_position: dict[int, dict[str, Any]] = {}
    for observation in observations:
        position = int(observation["position"])
        reference = observation["reference"]
        prior = reference_by_position.setdefault(position, reference)
        if prior != reference:
            raise ValueError("snapshot dense reference differs across sampling fractions")
    snapshot_fields = (
        "stream_id",
        "position",
        "context_length",
        "next_token_target",
        "dense_reference_next_token_ce",
        "dense_reference_top_one_prediction",
    )

    def snapshot_rows() -> Iterable[dict[str, Any]]:
        for position, reference in sorted(reference_by_position.items()):
            yield {
                "stream_id": stream_id,
                "position": position,
                "context_length": position + 1,
                "next_token_target": reference["next_token_target"],
                "dense_reference_next_token_ce": reference["next_token_loss"],
                "dense_reference_top_one_prediction": reference["top_one_prediction"],
            }

    feasibility_fields = (
        "stream_id",
        "position",
        "context_length",
        "requested_sampling_percentage",
        "requested_sampled_fraction",
        "realized_sampled_fraction_minimum",
        "realized_sampled_fraction_mean",
        "realized_sampled_fraction_maximum",
        "metric",
        "comparison",
        "threshold",
        "success_count",
        "attempts",
        "success_probability",
    )

    def feasibility_rows() -> Iterable[dict[str, Any]]:
        for observation in observations:
            trial_candidates = observation["candidates"]
            attempts = len(trial_candidates)
            realized = list(
                float(candidate["realized_sampled_fraction"]) for candidate in trial_candidates
            )
            metrics = (
                (
                    "kl_divergence_nats",
                    "<=",
                    tuple(float(value) for value in study["fidelity_tolerances_nats"]),
                    lambda value, threshold: value <= threshold,
                    observation["summary"]["successful_candidate_fraction"],
                ),
                (
                    "probability_distribution_overlap",
                    ">=",
                    SNAPSHOT_OVERLAP_THRESHOLDS,
                    lambda value, threshold: value >= threshold,
                    observation["summary"]["overlap_successful_candidate_fraction"],
                ),
            )
            for metric, comparison, thresholds, succeeds, original in metrics:
                for threshold in thresholds:
                    success_count = sum(
                        succeeds(float(candidate[metric]), threshold)
                        for candidate in trial_candidates
                    )
                    probability = success_count / attempts
                    threshold_key = format(threshold, ".12g")
                    if not math.isclose(
                        probability,
                        float(original[threshold_key]),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ):
                        raise ValueError("snapshot feasibility summary does not match candidates")
                    yield {
                        "stream_id": stream_id,
                        "position": observation["position"],
                        "context_length": observation["causal_history_length"],
                        "requested_sampling_percentage": observation[
                            "requested_sampling_percentage"
                        ],
                        "requested_sampled_fraction": observation["requested_sampled_fraction"],
                        "realized_sampled_fraction_minimum": min(realized),
                        "realized_sampled_fraction_mean": sum(realized) / attempts,
                        "realized_sampled_fraction_maximum": max(realized),
                        "metric": metric,
                        "comparison": comparison,
                        "threshold": threshold,
                        "success_count": success_count,
                        "attempts": attempts,
                        "success_probability": probability,
                    }

    output = Path(output_directory)
    trial_path = output / "H1a_trials.csv"
    snapshot_path = output / "H1a_snapshots.csv"
    feasibility_path = output / "H1a_feasibility.csv"
    trial_count = _write_csv(trial_path, trial_fields, trial_rows())
    snapshot_count = _write_csv(snapshot_path, snapshot_fields, snapshot_rows())
    feasibility_count = _write_csv(
        feasibility_path,
        feasibility_fields,
        feasibility_rows(),
    )
    trial_content_sha256 = _sha256(trial_path)
    trial_content_bytes = trial_path.stat().st_size
    if compress_trials:
        compressed_trial_path = trial_path.with_suffix(trial_path.suffix + ".gz")
        _gzip_csv(trial_path, compressed_trial_path)
        trial_path.unlink()
        published_trial_path = compressed_trial_path
        trial_record = {
            "sha256": _sha256(compressed_trial_path),
            "bytes": compressed_trial_path.stat().st_size,
            "rows": trial_count,
            "compression": "gzip",
            "uncompressed_name": trial_path.name,
            "uncompressed_sha256": trial_content_sha256,
            "uncompressed_bytes": trial_content_bytes,
        }
    else:
        published_trial_path = trial_path
        trial_record = {
            "sha256": trial_content_sha256,
            "bytes": trial_content_bytes,
            "rows": trial_count,
        }
    csv_records = {
        published_trial_path.name: trial_record,
        snapshot_path.name: {
            "sha256": _sha256(snapshot_path),
            "bytes": snapshot_path.stat().st_size,
            "rows": snapshot_count,
        },
        feasibility_path.name: {
            "sha256": _sha256(feasibility_path),
            "bytes": feasibility_path.stat().st_size,
            "rows": feasibility_count,
        },
    }
    root = Path(__file__).resolve().parent.parent
    resolved_result_path = result_path.resolve()
    displayed_result_path = (
        str(resolved_result_path.relative_to(root))
        if resolved_result_path.is_relative_to(root)
        else str(resolved_result_path)
    )
    observation_bytes = sum(
        (result_path.parent / record["path"]).stat().st_size
        for record in result["snapshot_observations"]
    )
    dense_prefix_paths = [
        result_path.parent / record["path"] for record in result["snapshot_prefixes"].values()
    ]
    manifest = {
        "schema_version": 1,
        "format": _SNAPSHOT_EXPORT_FORMAT,
        "run_id": result["run_id"],
        "source_identity": result["source_identity"],
        "model_identity": result["model_identity"],
        "data_identity": result["data_identity"],
        "configuration": result["config"],
        "configuration_sha256": result["config_sha256"],
        "sampler_version": study["sampling_algorithm"],
        "ordering": ["stream_id", "position", "requested_sampled_fraction", "trial_id"],
        "candidate_seed_encoding": "unsigned-64-bit-lowercase-hex",
        "sampled_location_histogram_bins": histogram_bins,
        "csv": csv_records,
        "original_artifacts": {
            "result": {
                "path": displayed_result_path,
                "sha256": _sha256(result_path),
            },
            "trial_sets": result["snapshot_observations"],
            "dense_prefixes": result["snapshot_prefixes"],
        },
        "original_frontiers": result["snapshot_summary"],
        "storage_bytes": {
            "csv_total": sum(record["bytes"] for record in csv_records.values()),
            "original_result_json": result_path.stat().st_size,
            "original_trial_set_json": observation_bytes,
            "original_structured_measurements": result_path.stat().st_size + observation_bytes,
            "dense_prefix_snapshots": sum(
                prefix_path.stat().st_size
                for prefix_path in dense_prefix_paths
                if prefix_path.is_file()
            ),
            "dense_prefix_snapshot_count": sum(
                prefix_path.is_file() for prefix_path in dense_prefix_paths
            ),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _snapshot_position_histogram(
    layer_indices: tuple[tuple[int, ...], ...],
    *,
    available_positions: int,
) -> list[int]:
    histogram = [0] * _SNAPSHOT_HISTOGRAM_BINS
    for index in (index for indices in layer_indices for index in indices):
        bucket = min(
            _SNAPSHOT_HISTOGRAM_BINS - 1,
            index * _SNAPSHOT_HISTOGRAM_BINS // available_positions,
        )
        histogram[bucket] += 1
    return histogram


def _run_snapshot_study(
    *,
    config_path: Path,
    specification: dict[str, Any],
    source_identity: dict[str, Any],
    reader: Any,
    streams: list[dict[str, Any]],
    data_identity: dict[str, Any],
    model_identity: dict[str, Any],
    device: Any,
    torch: Any,
    cuda_initialized_before_reader: bool,
    output_root: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    from qkvs.model import TokenStream
    from qkvs.selection import (
        RANDOM_BLOCK_SAMPLER_VERSION,
        derive_random_block_seed,
        sample_uniform_random_blocks,
    )

    if len(streams) != 1:
        raise ValueError("snapshot study requires exactly one token stream")
    stream_record = streams[0]
    stream = TokenStream(
        stream_id=stream_record["stream_id"],
        tokens=tuple(stream_record["tokens"]),
        support_positions=tuple(stream_record["support_positions"]),
    )
    study = specification["study"]
    output = _result_path(specification, output_root=output_root)
    reuse_prefixes: dict[int, tuple[Path, dict[str, Any]]] = {}
    reuse_identity = None
    reuse_source = study.get("reuse_dense_snapshots_from")
    if reuse_source is not None:
        configured_reuse_path = Path(reuse_source)
        reuse_result_path = resolve_run(
            configured_reuse_path
            if configured_reuse_path.is_absolute()
            else (config_path.parent / configured_reuse_path).resolve()
        )
        reuse_result = _load_json(reuse_result_path)
        if (
            not reuse_result.get("complete")
            or reuse_result.get("config", {}).get("study", {}).get("kind") != _SNAPSHOT_STUDY_KIND
            or reuse_result.get("model_identity") != model_identity
            or reuse_result.get("data_identity") != data_identity
            or reuse_result.get("device") != str(device)
            or reuse_result.get("snapshot_execution_version") != _SNAPSHOT_EXECUTION_VERSION
            or len(reuse_result.get("streams", [])) != 1
            or reuse_result["streams"][0].get("token_sha256") != stream_record["token_sha256"]
        ):
            raise ValueError("dense snapshot reuse source is incompatible with this study")
        for position in study["positions"]:
            record = reuse_result.get("snapshot_prefixes", {}).get(str(position))
            if record is None:
                raise ValueError("dense snapshot reuse source does not contain every position")
            prefix_path = (reuse_result_path.parent / record["path"]).resolve()
            reuse_prefixes[position] = (prefix_path, record)
        reuse_identity = {
            "configured_path": reuse_source,
            "run_id": reuse_result["run_id"],
            "result_sha256": _sha256(reuse_result_path),
        }
    expected = {
        "config": specification,
        "config_sha256": _sha256(config_path),
        "source_identity": source_identity,
        "model_identity": model_identity,
        "data_identity": data_identity,
        "device": str(device),
        "snapshot_execution_version": _SNAPSHOT_EXECUTION_VERSION,
    }
    if reuse_identity is not None:
        expected["dense_snapshot_reuse"] = reuse_identity
    if output.exists():
        result = _load_json(output)
        if {key: result.get(key) for key in expected} != expected or result.get(
            "streams"
        ) != streams:
            raise ValueError("saved snapshot study does not match current source and configuration")
    else:
        result = {
            "schema_version": 1,
            "run_id": specification["run"]["id"],
            **expected,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pid": os.getpid(),
            },
            "streams": streams,
            "snapshot_prefixes": {},
            "snapshot_observations": [],
            "snapshot_summary": {},
            "measurements": [],
            "measurement_references": [],
            "training": {},
            "summary": {},
            "complete": False,
        }
        _write_json(output, result)
    result["environment"].update(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_initialized_before_reader": cuda_initialized_before_reader,
            "cuda_initialized_after_reader": torch.cuda.is_initialized(),
        }
    )
    observations = _read_snapshot_observation_sets(result, output)
    completed = {
        (observation["position"], observation["requested_sampling_percentage"])
        for observation in observations
    }
    expected_trial_sets = len(study["positions"]) * len(study["sampling_percentages"])
    if result.get("complete") and len(completed) == expected_trial_sets:
        return output, load_run(output)
    completed_units, total_units, _ = _progress_counts(result)
    progress = _Progress(completed=completed_units, total=total_units)
    ownership = f"snapshot-study:{stream.stream_id}"
    prefill_chunk_tokens = specification["execution"]["prefill_chunk_tokens"]
    fidelity_tolerances = [float(value) for value in study["fidelity_tolerances_nats"]]
    base_seed = specification["run"]["seed"]
    trial_start = study["trial_start"]
    try:
        for position in study["positions"]:
            position_key = str(position)
            local_prefix_path = (
                output.parent
                / "snapshot_prefixes"
                / f"{stream.stream_id}-position-{position:08d}.pt"
            )
            reused_prefix = reuse_prefixes.get(position)
            prefix_record = result["snapshot_prefixes"].get(position_key)
            if prefix_record is None:
                progress.start_job(f"dense snapshot / position {position}")
                reused_from = None
                if reused_prefix is not None:
                    prefix_path, reused_record = reused_prefix
                    if not prefix_path.is_file() or reused_record.get("artifact_sha256") != _sha256(
                        prefix_path
                    ):
                        raise ValueError("reused dense causal snapshot artifact does not match")
                    snapshot = reader.load_dense_causal_snapshot(
                        prefix_path,
                        stream,
                        position=position,
                        device=device,
                        ownership=ownership,
                        prefill_chunk_tokens=prefill_chunk_tokens,
                    )
                    reused_from = {
                        "run_id": reuse_identity["run_id"],
                        "artifact_sha256": reused_record["artifact_sha256"],
                    }
                elif local_prefix_path.exists():
                    prefix_path = local_prefix_path
                    snapshot = reader.load_dense_causal_snapshot(
                        prefix_path,
                        stream,
                        position=position,
                        device=device,
                        ownership=ownership,
                        prefill_chunk_tokens=prefill_chunk_tokens,
                    )
                else:
                    prefix_path = local_prefix_path
                    snapshot = reader.dense_causal_snapshot(
                        stream,
                        position=position,
                        device=device,
                        ownership=ownership,
                        prefill_chunk_tokens=prefill_chunk_tokens,
                    )
                    reader.save_dense_causal_snapshot(snapshot, prefix_path)
                prefix_manifest = {
                    "path": os.path.relpath(prefix_path, output.parent),
                    "artifact_sha256": _sha256(prefix_path),
                    "identity": snapshot.identity,
                }
                if reused_from is not None:
                    prefix_manifest["reused_from"] = reused_from
                result["snapshot_prefixes"][position_key] = prefix_manifest
                _write_json(output, result)
                progress.complete_job()
            else:
                prefix_path = (output.parent / prefix_record["path"]).resolve()
                allowed_paths = {local_prefix_path.resolve()}
                if reused_prefix is not None:
                    allowed_paths.add(reused_prefix[0])
                if (
                    prefix_path not in allowed_paths
                    or not prefix_path.is_file()
                    or prefix_record.get("artifact_sha256") != _sha256(prefix_path)
                ):
                    raise ValueError("saved dense causal snapshot artifact does not match")
                snapshot = reader.load_dense_causal_snapshot(
                    prefix_path,
                    stream,
                    position=position,
                    device=device,
                    ownership=ownership,
                    prefill_chunk_tokens=prefill_chunk_tokens,
                )
            state = reader.prepare_snapshot_probe(
                snapshot,
                stream,
                position=position,
                device=device,
                ownership=ownership,
                prefill_chunk_tokens=prefill_chunk_tokens,
            )
            reference = reader.snapshot_probe(
                stream,
                state,
                position=position,
                selected_positions=None,
                device=device,
            )
            if not bool(torch.isfinite(reference.logits).all()) or not bool(
                torch.isfinite(reference.loss)
            ):
                raise RuntimeError("snapshot full-QKV reference is nonfinite")
            reference_values = (
                torch.stack((reference.loss.float(), reference.prediction.float())).cpu().tolist()
            )
            reference_log_probs = torch.log_softmax(reference.logits.float(), dim=-1)
            reference_probabilities = reference_log_probs.exp()

            for percentage in study["sampling_percentages"]:
                unit = (position, percentage)
                if unit in completed:
                    continue
                trial_count = 1 if float(percentage) == 100.0 else study["trials_per_fraction"]
                observation_path = _snapshot_observation_path(
                    output,
                    position=position,
                    percentage=percentage,
                    trial_start=trial_start,
                    trial_count=trial_count,
                )
                requested_fraction = float(percentage) / 100.0
                observation_identity = {
                    "format": "qkvs-snapshot-observation-v1",
                    "source_identity": source_identity,
                    "config_sha256": result["config_sha256"],
                    "model_identity": model_identity,
                    "data_identity": data_identity,
                    "stream_id": stream.stream_id,
                    "stream_token_sha256": stream_record["token_sha256"],
                    "dense_snapshot_identity": snapshot.identity,
                    "position": position,
                    "requested_sampling_percentage": percentage,
                    "block_width": study["block_width"],
                    "base_seed": base_seed,
                    "trial_start": trial_start,
                    "trial_count": trial_count,
                    "sampling_algorithm": RANDOM_BLOCK_SAMPLER_VERSION,
                    "snapshot_execution_version": _SNAPSHOT_EXECUTION_VERSION,
                    "device": str(device),
                    "environment": result["environment"],
                }
                identity_sha256 = _canonical_sha256(observation_identity)
                job_label = f"snapshot {position} / {float(percentage):g}% / {trial_count} trials"
                progress.start_job(
                    job_label,
                    fraction=0.0,
                )
                if observation_path.exists():
                    observation = _load_json(observation_path)
                    if (
                        observation.get("identity") != observation_identity
                        or observation.get("identity_sha256") != identity_sha256
                        or len(observation.get("candidates", [])) != trial_count
                    ):
                        raise ValueError("existing snapshot observation identity does not match")
                    progress.update_job(f"restore {job_label}", fraction=1.0)
                else:
                    candidate_records: list[dict[str, Any]] = []
                    candidate_values: list[Any] = []
                    layer_ranges_by_trial: dict[int, list[list[list[int]]]] = {}
                    for trial_offset in range(trial_count):
                        trial_id = trial_start + trial_offset
                        candidate_seed = derive_random_block_seed(
                            base_seed,
                            position,
                            _snapshot_fraction_coordinate(percentage),
                            trial_id,
                        )
                        layer_samples = tuple(
                            sample_uniform_random_blocks(
                                available_positions=position + 1,
                                block_width=study["block_width"],
                                requested_fraction=requested_fraction,
                                seed=derive_random_block_seed(candidate_seed, layer_index),
                            )
                            for layer_index in range(reader.model.config.num_hidden_layers)
                        )
                        layer_indices = tuple(sample.indices for sample in layer_samples)
                        probe = reader.snapshot_probe(
                            stream,
                            state,
                            position=position,
                            selected_positions=layer_indices,
                            device=device,
                        )
                        selected_occurrences = sum(len(indices) for indices in layer_indices)
                        gathered_keys = sum(probe.gathered_keys_per_layer)
                        gathered_values = sum(probe.gathered_values_per_layer)
                        if (
                            gathered_keys != selected_occurrences
                            or gathered_values != selected_occurrences
                        ):
                            raise RuntimeError(
                                "snapshot probe read accounting does not match selection"
                            )
                        sparse_log_probs = torch.log_softmax(probe.logits.float(), dim=-1)
                        sparse_probabilities = sparse_log_probs.exp()
                        divergence = torch.sum(
                            reference_probabilities * (reference_log_probs - sparse_log_probs)
                        ).clamp_min(0.0)
                        probability_overlap = torch.minimum(
                            reference_probabilities, sparse_probabilities
                        ).sum()
                        loss_delta = probe.loss.float() - reference.loss.float()
                        logit_error = (probe.logits.float() - reference.logits.float()).abs().max()
                        loss_error = (probe.loss.float() - reference.loss.float()).abs()
                        candidate_values.append(
                            torch.stack(
                                (
                                    divergence,
                                    probability_overlap,
                                    probe.loss.float(),
                                    loss_delta,
                                    probe.prediction.float(),
                                    logit_error,
                                    loss_error,
                                )
                            )
                        )
                        historical_occurrences = sum(
                            index < position for indices in layer_indices for index in indices
                        )
                        all_positions = {index for indices in layer_indices for index in indices}
                        historical_positions = {
                            index for index in all_positions if index < position
                        }
                        realized_fraction = selected_occurrences / (
                            reader.model.config.num_hidden_layers * (position + 1)
                        )
                        candidate_records.append(
                            {
                                "position": position,
                                "causal_history_length": position + 1,
                                "requested_sampling_percentage": percentage,
                                "requested_sampled_fraction": requested_fraction,
                                "realized_sampled_fraction": realized_fraction,
                                "sampling_cost_rho": realized_fraction,
                                "selected_token_occurrences": selected_occurrences,
                                "historical_token_occurrences": historical_occurrences,
                                "current_token_layer_reads": selected_occurrences
                                - historical_occurrences,
                                "aggregate_unique_positions": len(all_positions),
                                "historical_unique_positions": len(historical_positions),
                                "sampled_block_count_per_layer": [
                                    sample.block_count for sample in layer_samples
                                ],
                                "aggregate_block_count": sum(
                                    sample.block_count for sample in layer_samples
                                ),
                                "per_layer_selection": [
                                    {
                                        "layer": layer_index,
                                        "seed": sample.seed,
                                        "boundary_alignment": sample.boundary_alignment,
                                        "block_count": sample.block_count,
                                        "selected_token_count": len(sample.indices),
                                        "gathered_key_occurrences": layer_gathered_keys,
                                        "gathered_value_occurrences": layer_gathered_values,
                                    }
                                    for layer_index, (
                                        sample,
                                        layer_gathered_keys,
                                        layer_gathered_values,
                                    ) in enumerate(
                                        zip(
                                            layer_samples,
                                            probe.gathered_keys_per_layer,
                                            probe.gathered_values_per_layer,
                                            strict=True,
                                        )
                                    )
                                ],
                                "sampled_position_histogram": _snapshot_position_histogram(
                                    layer_indices,
                                    available_positions=position + 1,
                                ),
                                "trial_id": trial_id,
                                "candidate_seed": candidate_seed,
                                "sampling_algorithm": RANDOM_BLOCK_SAMPLER_VERSION,
                                "execution_identity_sha256": identity_sha256,
                                "actual_gathered_key_occurrences": gathered_keys,
                                "actual_gathered_value_occurrences": gathered_values,
                            }
                        )
                        layer_ranges_by_trial[trial_id] = [
                            [[start, stop] for start, stop in sample.ranges]
                            for sample in layer_samples
                        ]
                        progress.update_job(
                            f"snapshot {position} / {float(percentage):g}% / "
                            f"trial {trial_offset + 1}/{trial_count}",
                            fraction=(trial_offset + 1) / trial_count,
                        )
                    materialized = torch.stack(candidate_values).cpu().tolist()
                    for candidate, values in zip(candidate_records, materialized, strict=True):
                        if not all(math.isfinite(value) for value in values):
                            raise RuntimeError("snapshot candidate measurement is nonfinite")
                        if not (
                            -_SNAPSHOT_PROBABILITY_TOLERANCE
                            <= values[1]
                            <= 1.0 + _SNAPSHOT_PROBABILITY_TOLERANCE
                        ):
                            raise RuntimeError("snapshot probability overlap is outside [0, 1]")
                        candidate.update(
                            {
                                "kl_divergence_nats": values[0],
                                "probability_distribution_overlap": values[1],
                                "sparse_next_token_loss": values[2],
                                "next_token_ce_delta_nats": values[3],
                                "sparse_top_one_prediction": int(values[4]),
                                "top_one_agreement": int(values[4]) == int(reference_values[1]),
                                "full_logit_max_abs_error": values[5],
                                "full_loss_abs_error": values[6],
                            }
                        )
                    summary = summarize_snapshot_trial_set(
                        candidate_records,
                        fidelity_tolerances=fidelity_tolerances,
                    )
                    total_selected = sum(
                        candidate["selected_token_occurrences"] for candidate in candidate_records
                    )
                    bin_population = [0] * _SNAPSHOT_HISTOGRAM_BINS
                    for index in range(position + 1):
                        bucket = min(
                            _SNAPSHOT_HISTOGRAM_BINS - 1,
                            index * _SNAPSHOT_HISTOGRAM_BINS // (position + 1),
                        )
                        bin_population[bucket] += 1
                    expected_histogram = [
                        total_selected * population / (position + 1)
                        for population in bin_population
                    ]
                    summary["normalized_position_bin_edges"] = [
                        index / _SNAPSHOT_HISTOGRAM_BINS
                        for index in range(_SNAPSHOT_HISTOGRAM_BINS + 1)
                    ]
                    summary["random_sampling_expected_position_histogram"] = expected_histogram
                    summary["sampled_age_histogram"] = list(
                        reversed(summary["sampled_position_histogram"])
                    )
                    summary["random_sampling_expected_age_histogram"] = list(
                        reversed(expected_histogram)
                    )
                    best_trial = summary["best_candidate"]["trial_id"]
                    summary["best_candidate"]["per_layer_selected_ranges"] = layer_ranges_by_trial[
                        best_trial
                    ]
                    if float(percentage) == 100.0:
                        tolerance = study["dense_parity_tolerance"]
                        dense_candidate = candidate_records[0]
                        if (
                            dense_candidate["full_logit_max_abs_error"] > tolerance["logit_atol"]
                            or dense_candidate["full_loss_abs_error"] > tolerance["loss_atol"]
                            or abs(dense_candidate["probability_distribution_overlap"] - 1.0)
                            > _SNAPSHOT_PROBABILITY_TOLERANCE
                        ):
                            raise RuntimeError("100% snapshot candidate does not match full QKV")
                    observation = {
                        "schema_version": 1,
                        "type": "snapshot_trial_set",
                        "identity": observation_identity,
                        "identity_sha256": identity_sha256,
                        "position": position,
                        "causal_history_length": position + 1,
                        "requested_sampling_percentage": percentage,
                        "requested_sampled_fraction": requested_fraction,
                        "reference": {
                            "next_token_target": stream.tokens[position + 1],
                            "next_token_loss": reference_values[0],
                            "top_one_prediction": int(reference_values[1]),
                            "selected_position_count_per_layer": list(
                                reference.selected_positions_per_layer
                            ),
                        },
                        "sampling": {
                            "algorithm": RANDOM_BLOCK_SAMPLER_VERSION,
                            "base_seed": base_seed,
                            "block_width": study["block_width"],
                            "trial_start": trial_start,
                            "trial_count": trial_count,
                            "layers_sampled_independently": True,
                            "without_replacement": True,
                            "intentional_overlap": False,
                            "mandatory_current_token": False,
                            "mandatory_recent_tokens": False,
                        },
                        "candidates": candidate_records,
                        "summary": summary,
                    }
                    _write_json(observation_path, observation)
                manifest = {
                    "position": position,
                    "requested_sampling_percentage": percentage,
                    "path": str(observation_path.relative_to(output.parent)),
                    "artifact_sha256": _sha256(observation_path),
                    "identity_sha256": identity_sha256,
                    "trial_start": trial_start,
                    "trial_count": trial_count,
                }
                result["snapshot_observations"].append(manifest)
                result["snapshot_observations"].sort(
                    key=lambda item: (
                        item["position"],
                        item["requested_sampling_percentage"],
                    )
                )
                observations.append(observation)
                completed.add(unit)
                result["snapshot_summary"] = summarize_snapshot_frontiers(
                    observations,
                    fidelity_tolerances=fidelity_tolerances,
                )
                result["complete"] = len(completed) == expected_trial_sets
                _write_json(output, result)
                progress.complete_job()
            del state, snapshot, reference
    finally:
        progress.close()
    return output, load_run(output)


def _prepare_dense_prefills(
    *,
    reader: Any,
    streams: list[dict[str, Any]],
    specification: dict[str, Any],
    result: dict[str, Any],
    output: Path,
    device: Any,
    progress: _Progress,
) -> dict[str, Any]:
    """Load or atomically create one S-independent prefix per owner and token prefix."""

    if specification["model"]["backend"] != "huggingface":
        result["dense_prefill"] = {}
        return {}

    from qkvs.model import TokenStream

    rolling_tokens = specification["attention"]["rolling_tokens"]
    activation_tokens = specification["attention"]["activation_tokens"]
    prefill_chunk_tokens = specification["execution"]["prefill_chunk_tokens"]
    snapshots: dict[str, Any] = {}
    shared: dict[tuple[str, tuple[int, ...]], tuple[Any, Path]] = {}
    records = dict(result.get("dense_prefill", {}))
    for record in streams:
        stream = TokenStream(
            stream_id=record["stream_id"],
            tokens=tuple(record["tokens"]),
            support_positions=tuple(record["support_positions"]),
        )
        ownership = record.get("split", "evaluation")
        if len(stream.tokens) < activation_tokens:
            raise ValueError(
                f"stream {stream.stream_id} is shorter than attention.activation_tokens"
            )
        prefix = stream.tokens[:activation_tokens]
        key = (ownership, prefix)
        prior = records.get(stream.stream_id)
        progress.start_job(
            f"dense prefill / {stream.stream_id}",
            advances=prior is None,
        )
        if key not in shared:
            prefix_sha256 = hashlib.sha256(
                json.dumps(prefix, separators=(",", ":")).encode()
            ).hexdigest()
            path = output.parent / "dense_prefill" / f"{ownership}-{prefix_sha256}.pt"
            if prior is not None:
                if (
                    not path.exists()
                    or prior.get("path") != str(path.relative_to(output.parent))
                    or prior.get("artifact_sha256") != _sha256(path)
                ):
                    raise ValueError("dense prefill artifact does not match the saved run")
            if path.exists():
                snapshot = reader.load_dense_prefill(
                    path,
                    stream,
                    rolling_tokens=rolling_tokens,
                    activation_tokens=activation_tokens,
                    device=device,
                    ownership=ownership,
                    prefill_chunk_tokens=prefill_chunk_tokens,
                )
            else:
                snapshot = reader.dense_prefill(
                    stream,
                    rolling_tokens=rolling_tokens,
                    activation_tokens=activation_tokens,
                    device=device,
                    ownership=ownership,
                    prefill_chunk_tokens=prefill_chunk_tokens,
                )
                reader.save_dense_prefill(snapshot, path)
            shared[key] = (snapshot, path)
        snapshot, path = shared[key]
        snapshots[stream.stream_id] = snapshot
        records[stream.stream_id] = {
            "ownership": ownership,
            "path": str(path.relative_to(output.parent)),
            "artifact_sha256": _sha256(path),
            "identity": snapshot.identity,
        }
        result["dense_prefill"] = records
        _write_json(output, result)
        progress.complete_job()
    return snapshots


def _build_learned_policy(reader: Any, specification: dict[str, Any], torch: Any) -> Any:
    from qkvs.selection import LearnedHeatFieldPolicy

    selection = specification["selection"]
    learned = selection["learned"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(specification["run"]["seed"])
        policy = LearnedHeatFieldPolicy(
            hidden_size=reader.model.config.hidden_size,
            layers=reader.model.config.num_hidden_layers,
            proposal_count=selection["proposal_count"],
            latent_size=learned["latent_size"],
            gamma=float(selection["gamma"]["value"]),
            beta=float(selection["beta"]["value"]),
            minimum_spread=float(learned["minimum_spread"]),
            maximum_spread=float(learned["maximum_spread"]),
            gate_threshold=float(learned["gate_threshold"]),
        )
    return policy


def _train_learned_policy(
    *,
    reader: Any,
    policy: Any,
    streams: list[dict[str, Any]],
    specification: dict[str, Any],
    result: dict[str, Any],
    output: Path,
    device: Any,
    torch: Any,
    dense_prefills: dict[str, Any],
    progress: _Progress,
) -> None:
    from qkvs.model import TokenStream

    training = specification["training"]
    attention = specification["attention"]
    selection = specification["selection"]
    checkpoint = output.parent / "selector.pt"
    checkpoint_identity = {
        "config_sha256": result["config_sha256"],
        "source_identity": result["source_identity"],
        "model_identity": result["model_identity"],
        "data_identity": result["data_identity"],
    }
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(training["learning_rate"]))
    token_streams = [
        TokenStream(
            stream_id=stream["stream_id"],
            tokens=tuple(stream["tokens"]),
            support_positions=tuple(stream["support_positions"]),
        )
        for stream in streams
        if stream.get("split") == "train"
    ]
    if not token_streams:
        raise ValueError("learned heat-field training has no training streams")
    prefill_chunk_tokens = specification["execution"]["prefill_chunk_tokens"]
    step = 0
    target_count = 0
    next_stream_index = 0
    history: list[dict[str, Any]] = []
    states: dict[str, Any] = {}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("identity") != checkpoint_identity:
            raise ValueError("learned selector checkpoint identity does not match this run")
        policy.load_state_dict(saved["policy"])
        optimizer.load_state_dict(saved["optimizer"])
        step = saved["step"]
        target_count = saved["target_count"]
        next_stream_index = saved["next_stream_index"]
        history = saved["history"]
        for stream in token_streams:
            states[stream.stream_id] = reader.restore_heatfield_rollout_state(
                saved["rollout_states"][stream.stream_id],
                stream,
                dense_prefills[stream.stream_id],
                rolling_tokens=attention["rolling_tokens"],
                activation_tokens=attention["activation_tokens"],
                device=device,
                ownership="train",
                prefill_chunk_tokens=prefill_chunk_tokens,
            )
    else:
        if result.get("training", {}).get("completed_steps", 0):
            raise ValueError("learned selector checkpoint is missing")
        for stream in token_streams:
            states[stream.stream_id] = reader.new_heatfield_rollout_state(
                stream,
                dense_prefills[stream.stream_id],
                rolling_tokens=attention["rolling_tokens"],
                activation_tokens=attention["activation_tokens"],
                device=device,
                ownership="train",
                prefill_chunk_tokens=prefill_chunk_tokens,
            )

    policy.train()
    planned_target_count = 0
    for stream in token_streams:
        continuation = len(stream.tokens) - 1 - attention["activation_tokens"]
        complete_rollouts, trailing_tokens = divmod(continuation, training["rollout_tokens"])
        planned_target_count += complete_rollouts * min(
            training["targets_per_rollout"],
            training["rollout_tokens"],
        )
        planned_target_count += min(training["targets_per_rollout"], trailing_tokens)
    timed_steps = 0
    timed_seconds = 0.0
    checkpoint_start = step
    checkpoint_stop = step
    while step < training["steps"]:
        if step % training["checkpoint_interval"] == 0:
            checkpoint_start = step
            checkpoint_stop = min(
                training["steps"],
                step + training["checkpoint_interval"],
            )
            job = (
                f"train {step + 1}/{training['steps']} "
                f"{target_count}/{planned_target_count} targets"
            )
            progress.start_job(
                job,
                fraction=0.0,
            )
        step_started = time.perf_counter()
        selected_stream_index = None
        for offset in range(len(token_streams)):
            candidate_index = (next_stream_index + offset) % len(token_streams)
            candidate = token_streams[candidate_index]
            if states[candidate.stream_id].next_position < len(candidate.tokens) - 1:
                selected_stream_index = candidate_index
                break
        if selected_stream_index is None:
            raise ValueError("streaming training exhausted all causal continuations")
        stream = token_streams[selected_stream_index]
        state = states[stream.stream_id]
        next_stream_index = (selected_stream_index + 1) % len(token_streams)
        start_position = state.next_position
        stop_position = min(
            len(stream.tokens) - 1,
            start_position + training["rollout_tokens"],
        )
        rollout_length = stop_position - start_position
        target_total = min(training["targets_per_rollout"], rollout_length)
        if target_total == 1:
            target_positions = [start_position + rollout_length // 2]
        else:
            target_positions = [
                start_position + round(index * (rollout_length - 1) / (target_total - 1))
                for index in range(target_total)
            ]
        target_layers = {
            position: (target_count + index) % reader.model.config.num_hidden_layers
            for index, position in enumerate(target_positions)
        }
        optimizer.zero_grad(set_to_none=True)
        objectives, rollout_metrics = reader.heatfield_training_rollout(
            stream,
            policy,
            state,
            stop_position=stop_position,
            target_layers=target_layers,
            rolling_tokens=attention["rolling_tokens"],
            historical_token_budget=attention["historical_token_budget"],
            window_width=selection["window_width"],
            coverage_weight=float(training["coverage_weight"]),
            utility_weight=float(training["utility_weight"]),
            utility_gradient_estimator=training.get("utility_gradient_estimator", "soft_occupancy"),
            predictive_continuation=training.get("predictive_continuation", "rolling"),
            device=device,
        )
        if not objectives:
            raise RuntimeError("streaming training rollout produced no target objectives")
        objective = torch.stack(objectives).mean()
        objective.backward()
        gradient_norm = torch.sqrt(
            sum(
                parameter.grad.detach().float().square().sum()
                for parameter in policy.parameters()
                if parameter.grad is not None
            )
        )
        optimizer.step()
        step += 1
        target_count += len(rollout_metrics)
        denominator = max(1, len(stream.tokens) - 1 - attention["activation_tokens"])
        history.extend(
            {
                "step": step,
                "stream_id": stream.stream_id,
                "gradient_norm": float(gradient_norm),
                "context_fraction": (metrics["position"] - attention["activation_tokens"])
                / denominator,
                **metrics,
            }
            for metrics in rollout_metrics
        )
        timed_steps += 1
        timed_seconds += time.perf_counter() - step_started
        progress.update_job(
            f"train {step}/{training['steps']} {target_count}/{planned_target_count} targets "
            f"{timed_seconds / timed_steps:.1f}s/step",
            fraction=(step - checkpoint_start) / (checkpoint_stop - checkpoint_start),
        )
        if step % training["checkpoint_interval"] == 0 or step == training["steps"]:
            progress.update_job(
                f"checkpoint after step {step}/{training['steps']}",
                fraction=1.0,
            )
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(
                {
                    "identity": checkpoint_identity,
                    "step": step,
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "target_count": target_count,
                    "next_stream_index": next_stream_index,
                    "rollout_states": {
                        stream_id: reader.serialize_heatfield_rollout_state(state)
                        for stream_id, state in states.items()
                    },
                },
                temporary,
            )
            temporary.replace(checkpoint)
            result["training"] = {
                "completed_steps": step,
                "target_steps": training["steps"],
                "checkpoint_sha256": _sha256(checkpoint),
                "history": history,
                "soft_hard_alignment": summarize_training_alignment(history),
                "streaming": {
                    "rollout_tokens": training["rollout_tokens"],
                    "targets_per_rollout": training["targets_per_rollout"],
                    "utility_gradient_estimator": training.get(
                        "utility_gradient_estimator", "soft_occupancy"
                    ),
                    "predictive_continuation": training.get("predictive_continuation", "rolling"),
                    "effective_supervised_targets": len(history),
                    "effective_supervised_targets_per_layer": {
                        str(layer): sum(record["layer"] == layer for record in history)
                        for layer in range(reader.model.config.num_hidden_layers)
                    },
                    "next_positions": {
                        stream_id: state.next_position for stream_id, state in states.items()
                    },
                    "selector_updates_only_at_rollout_boundaries": True,
                },
            }
            _write_json(output, result)
            progress.complete_job()
    if any(
        states[stream.stream_id].next_position != len(stream.tokens) - 1 for stream in token_streams
    ):
        raise ValueError("training steps do not cover every post-transition continuation")
    policy.eval()


def run_experiment(
    config_path: str | Path,
    *,
    device: str,
    output_root: str | Path | None = None,
    allow_dirty_source: bool = False,
) -> tuple[Path, dict[str, Any]]:
    config_path = Path(config_path)
    specification = _load_json(config_path)
    validation = validate_config(specification)
    if not validation["valid"]:
        raise ValueError("invalid configuration: " + "; ".join(validation["errors"]))
    if not validation["runnable"]:
        reasons = "; ".join(validation["not_runnable_reasons"])
        raise ValueError("configuration is not runnable: " + reasons)
    if device not in specification["execution"]["allowed_devices"]:
        raise ValueError(f"device {device} is not allowed by this configuration")

    identity = _source_identity(config_path)
    if not identity["tracked_files_clean"] and not allow_dirty_source:
        raise ValueError("harness source must be committed before execution")

    import torch

    from qkvs.model import TokenStream, load_reader
    from qkvs.selection import build_policy

    torch_device = torch.device(device)
    cuda_initialized_before_reader = torch.cuda.is_initialized()
    reader = load_reader(specification["model"], torch_device)
    streams, data_identity = _build_streams(
        specification,
        config_path=config_path,
        tokenizer=reader.tokenizer,
    )
    streams = json.loads(json.dumps(streams))
    model_identity = reader.identity()
    if specification.get("study", {}).get("kind") == _SNAPSHOT_STUDY_KIND:
        return _run_snapshot_study(
            config_path=config_path,
            specification=specification,
            source_identity=identity,
            reader=reader,
            streams=streams,
            data_identity=data_identity,
            model_identity=model_identity,
            device=torch_device,
            torch=torch,
            cuda_initialized_before_reader=cuda_initialized_before_reader,
            output_root=output_root,
        )
    output = _result_path(specification, output_root=output_root)
    expected = {
        "config": specification,
        "config_sha256": _sha256(config_path),
        "source_identity": identity,
        "model_identity": model_identity,
        "data_identity": data_identity,
        "device": device,
        "predictive_execution_versions": _PREDICTIVE_EXECUTION_VERSIONS,
    }
    if output.exists():
        result = _load_json(output)
        actual = {key: result.get(key) for key in expected}
        if actual != expected or result.get("streams") != streams:
            raise ValueError("saved run does not match the current source and configuration")
    else:
        result = {
            "schema_version": 1,
            "run_id": specification["run"]["id"],
            **expected,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pid": os.getpid(),
            },
            "streams": streams,
            "dense_prefill": {},
            "measurements": [],
            "measurement_references": [],
            "training": {},
            "summary": {},
            "complete": False,
        }
        _write_json(output, result)
    result["environment"].update(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_initialized_before_reader": cuda_initialized_before_reader,
            "cuda_initialized_after_reader": torch.cuda.is_initialized(),
        }
    )
    completed_units, total_units, _ = _progress_counts(result)
    progress = _Progress(completed=completed_units, total=total_units)
    try:
        dense_prefills = _prepare_dense_prefills(
            reader=reader,
            streams=streams,
            specification=specification,
            result=result,
            output=output,
            device=torch_device,
            progress=progress,
        )
        learned_policy = None
        untrained_policy = None
        if any(
            condition["policy"] == "heatfield"
            for condition in specification["selection"]["conditions"]
        ):
            untrained_policy = _build_learned_policy(reader, specification, torch).to(torch_device)
            untrained_policy_sha256 = _module_state_sha256(untrained_policy)
            prior_untrained_identity = result.get("untrained_policy_sha256")
            if (
                prior_untrained_identity is not None
                and prior_untrained_identity != untrained_policy_sha256
            ):
                raise ValueError("untrained selector identity does not match this run")
            if prior_untrained_identity is None:
                result["untrained_policy_sha256"] = untrained_policy_sha256
                _write_json(output, result)
            learned_policy = _build_learned_policy(reader, specification, torch)
            _train_learned_policy(
                reader=reader,
                policy=learned_policy,
                streams=streams,
                specification=specification,
                result=result,
                output=output,
                device=torch_device,
                torch=torch,
                dense_prefills=dense_prefills,
                progress=progress,
            )
        resolved_measurements = _resolved_measurements(result, output)
        completed = {
            (measurement["condition"], measurement["stream_id"])
            for measurement in resolved_measurements
        }
        reusable = _discover_reusable_measurements(result, output=output)
        units = [
            (condition, stream)
            for condition in specification["selection"]["conditions"]
            for stream in result["streams"]
            if stream.get("split") == "evaluation"
        ]
    except BaseException:
        progress.close()
        raise
    evaluation_total = len(units)
    try:
        for condition, stream_record in units:
            unit = (condition["name"], stream_record["stream_id"])
            if unit in completed:
                continue
            if unit in reusable:
                progress.start_job(f"reuse {condition['name']} / {stream_record['stream_id']}")
                reference, reused_measurement = reusable[unit]
                result["measurement_references"].append(reference)
                result["measurement_references"].sort(
                    key=lambda item: (item["condition"], item["stream_id"])
                )
                resolved_measurements.append(reused_measurement)
                resolved_measurements.sort(
                    key=lambda measurement: (
                        measurement["condition"],
                        measurement["stream_id"],
                    )
                )
                completed.add(unit)
                result["summary"] = summarize_measurements(
                    resolved_measurements,
                    baseline=specification["evaluation"]["baseline"],
                    bootstrap_samples=specification["evaluation"]["bootstrap_samples"],
                    seed=specification["run"]["seed"],
                    historical_token_budget=specification["attention"]["historical_token_budget"],
                    paired_comparisons=specification["evaluation"].get("comparisons", []),
                )
                result["complete"] = len(completed) == evaluation_total
                _write_json(output, result)
                progress.complete_job()
                continue
            progress.start_job(f"{condition['name']} / {stream_record['stream_id']}")
            policy = build_policy(
                condition,
                specification["selection"],
                learned_policy=learned_policy,
                untrained_policy=untrained_policy,
            )
            stream = TokenStream(
                stream_id=stream_record["stream_id"],
                tokens=tuple(stream_record["tokens"]),
                support_positions=tuple(stream_record["support_positions"]),
            )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
                torch.cuda.reset_peak_memory_stats(torch_device)
            execution_started = time.perf_counter()
            token_scores = reader.execute(
                stream,
                policy,
                rolling_tokens=specification["attention"]["rolling_tokens"],
                historical_token_budget=specification["attention"]["historical_token_budget"],
                window_width=specification["selection"]["window_width"],
                device=torch_device,
                activation_tokens=specification["attention"]["activation_tokens"],
                dense_snapshot=dense_prefills.get(stream.stream_id),
                snapshot_ownership="evaluation",
                prefill_chunk_tokens=specification["execution"]["prefill_chunk_tokens"],
                proposal_trace_interval=specification["evaluation"]["proposal_trace_interval"],
            )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            wall_seconds = time.perf_counter() - execution_started
            continuation_tokens = sum(
                score.phase in {"post_transition", "qkvs"} for score in token_scores
            )
            measurement = {
                "condition": condition["name"],
                "policy": condition["policy"],
                "stream_id": stream.stream_id,
                "token_scores": [score.as_dict() for score in token_scores],
                "performance": {
                    "scope": "snapshot_restore_plus_autoregressive_continuation",
                    "wall_seconds_including_snapshot_restore": wall_seconds,
                    "post_transition_tokens": continuation_tokens,
                    "post_transition_tokens_per_second": (
                        continuation_tokens / wall_seconds if wall_seconds else 0.0
                    ),
                    "peak_accelerator_bytes": (
                        torch.cuda.max_memory_allocated(torch_device)
                        if torch_device.type == "cuda"
                        else None
                    ),
                    "theoretical_speedup_claimed": False,
                },
            }
            result["measurements"].append(measurement)
            resolved_measurements.append(measurement)
            result["measurements"].sort(
                key=lambda measurement: (measurement["condition"], measurement["stream_id"])
            )
            resolved_measurements.sort(
                key=lambda measurement: (measurement["condition"], measurement["stream_id"])
            )
            completed.add(unit)
            result["summary"] = summarize_measurements(
                resolved_measurements,
                baseline=specification["evaluation"]["baseline"],
                bootstrap_samples=specification["evaluation"]["bootstrap_samples"],
                seed=specification["run"]["seed"],
                historical_token_budget=specification["attention"]["historical_token_budget"],
                paired_comparisons=specification["evaluation"].get("comparisons", []),
            )
            result["complete"] = len(completed) == evaluation_total
            _write_json(output, result)
            progress.complete_job()
    finally:
        progress.close()
    del reusable, resolved_measurements, result
    return output, load_run(output)


def resolve_run(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / "result.json" if candidate.is_dir() else candidate


def load_run(path: str | Path) -> dict[str, Any]:
    result_path = resolve_run(path)
    result = _load_json(result_path)
    if result.get("schema_version") != 1 or "measurements" not in result:
        raise ValueError("run artifact is not a QKVS harness result")
    if result.get("config", {}).get("study", {}).get("kind") == _SNAPSHOT_STUDY_KIND:
        _read_snapshot_observation_sets(result, result_path)
        return result
    result["measurements"] = _resolved_measurements(result, result_path)
    return result


def status(path: str | Path) -> dict[str, Any]:
    result_path = resolve_run(path)
    result = _load_json(result_path)
    if result.get("schema_version") != 1 or "measurements" not in result:
        raise ValueError("run artifact is not a QKVS harness result")
    if result.get("config", {}).get("study", {}).get("kind") == _SNAPSHOT_STUDY_KIND:
        _read_snapshot_observation_sets(result, result_path)
    completed, total, phases = _progress_counts(result)
    training = result.get("training", {})
    study_kind = result.get("config", {}).get("study", {}).get("kind")
    return {
        "run_id": result["run_id"],
        "study_kind": study_kind,
        "completed_units": completed,
        "total_units": total,
        "phase_units": {
            name: {"completed": phase_completed, "total": phase_total}
            for name, (phase_completed, phase_total) in phases.items()
        },
        "training_steps": {
            "completed": training.get("completed_steps", 0),
            "total": result["config"].get("training", {}).get("steps", 0),
        },
        "complete": result["complete"],
        "source_revision": result["source_identity"]["git_revision"],
        "config_sha256": result["config_sha256"],
        "artifact_sha256": _sha256(result_path),
    }


def print_status(value: dict[str, Any]) -> None:
    total = value["total_units"]
    completed = value["completed_units"]
    state = "complete" if value["complete"] else "incomplete"
    counter_width = len(str(total)) + 1
    print(
        f"QKVS [{completed:>{counter_width}} / {total} ] "
        f"[{_progress_bar(completed, total)}] {state}"
    )
    phases = value["phase_units"]
    if value.get("study_kind") == _SNAPSHOT_STUDY_KIND:
        print(
            "Phases: "
            f"snapshot prefixes {phases['snapshot_prefixes']['completed']}/"
            f"{phases['snapshot_prefixes']['total']}, "
            f"trial sets {phases['snapshot_trial_sets']['completed']}/"
            f"{phases['snapshot_trial_sets']['total']}"
        )
        print(f"Run: {value['run_id']}")
        print(f"Source: {value['source_revision']}")
        print(f"Artifact SHA-256: {value['artifact_sha256']}")
        return
    training = value["training_steps"]
    print(
        "Phases: "
        f"prefill {phases['dense_prefill']['completed']}/{phases['dense_prefill']['total']}, "
        f"training checkpoints "
        f"{phases['training_checkpoints']['completed']}/"
        f"{phases['training_checkpoints']['total']} "
        f"(steps {training['completed']}/{training['total']}), "
        f"evaluation {phases['evaluation']['completed']}/{phases['evaluation']['total']}"
    )
    print(f"Run: {value['run_id']}")
    print(f"Source: {value['source_revision']}")
    print(f"Artifact SHA-256: {value['artifact_sha256']}")


def print_summary(result: dict[str, Any]) -> None:
    print(f"Run {result['run_id']} ({'complete' if result['complete'] else 'incomplete'})")
    if result.get("config", {}).get("study", {}).get("kind") == _SNAPSHOT_STUDY_KIND:
        frontiers = result.get("snapshot_summary", {}).get("frontiers_by_position", {})
        for position, tolerances in sorted(frontiers.items(), key=lambda item: int(item[0])):
            values = []
            for tolerance, frontier in sorted(tolerances.items(), key=lambda item: float(item[0])):
                fraction = frontier["smallest_discovered_fraction"]
                values.append(
                    f"KL<={tolerance}: "
                    + (f"{100.0 * fraction:.3f}%" if fraction is not None else "not found")
                )
            print(f"position {position}: " + ", ".join(values))
        return
    reused_conditions = {
        measurement["condition"]
        for measurement in result.get("measurements", [])
        if "predictive_reuse" in measurement
    }
    alignment = result.get("training", {}).get("soft_hard_alignment")
    if alignment:
        correlation = alignment["soft_hard_utility_correlation"]
        sign_agreement = alignment["utility_sign_agreement"]
        print(
            "training soft/hard alignment: "
            f"steps={alignment['paired_steps']}, "
            f"correlation={correlation if correlation is not None else 'undefined'}, "
            f"sign_agreement={sign_agreement if sign_agreement is not None else 'undefined'}, "
            f"mean_abs_gap={alignment['mean_absolute_utility_gap']}"
        )
    for name, condition in result["summary"].get("conditions", {}).items():
        retrieval = condition["retrieval"]
        budget = retrieval.get("historical_budget_utilization")
        budget_text = f"{budget:.3f}" if budget is not None else "unbounded"
        post = condition.get("post_transition", {})
        print(
            f"{name}: loss={condition['mean_next_token_cross_entropy']:.6f}, "
            f"post_transition_loss={post.get('mean_next_token_cross_entropy', 0.0):.6f}, "
            f"accuracy={condition['accuracy']:.3f}, "
            f"historical_reads={retrieval['historical_token_occurrences']}, "
            f"budget_utilization={budget_text}, "
            f"overlap={retrieval['overlap_occurrences']}, "
            f"temporal_stability={retrieval['temporal_stability']:.3f}, "
            f"predictive={'reused' if name in reused_conditions else 'measured'}"
        )
    for name, comparison in result["summary"].get("paired_utility", {}).items():
        interval = comparison["paired_bootstrap_95_interval_nats"]
        if interval is None:
            print(
                f"{name}: paired utility={comparison['mean_utility_nats']:.6f} nat "
                f"({comparison['statistical_interpretation']})"
            )
        else:
            print(
                f"{name}: paired utility={comparison['mean_utility_nats']:.6f} nat "
                f"95% interval=[{interval[0]:.6f}, {interval[1]:.6f}]"
            )


def dump_json(value: Any) -> None:
    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
