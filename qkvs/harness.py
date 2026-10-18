from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from qkvs.metrics import summarize_measurements

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
_PROGRESS_WIDTH = 40


def _progress_bar(current: int, total: int, *, width: int = _PROGRESS_WIDTH) -> str:
    if total <= 0:
        filled = width
    else:
        filled = width * min(current, total) // total
        if current > 0:
            filled = max(1, filled)
    return "#" * filled + " " * (width - filled)


def _progress_line(current: int, total: int, job: str, elapsed_seconds: float) -> str:
    counter_width = len(str(total)) + 1
    elapsed = max(0, int(elapsed_seconds))
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"[{current:>{counter_width}} / {total} ] "
        f"[{_progress_bar(current, total)}] {job} : "
        f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    )


class _LiveProgress:
    """Render one durable job counter with a transient one-second heartbeat."""

    def __init__(
        self,
        *,
        completed: int,
        total: int,
        stream: TextIO = sys.stdout,
        interval_seconds: float = 1.0,
    ) -> None:
        self.completed = completed
        self.total = total
        self.stream = stream
        self.interval_seconds = interval_seconds
        self.active_job: str | None = None
        self.running = False
        self.started = time.monotonic()
        self.last_width = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _heartbeat(self) -> None:
        while not self.stop.wait(self.interval_seconds):
            self.render()

    def _current(self) -> int:
        if not self.running:
            return self.completed
        return min(self.completed + 1, self.total)

    def render(self) -> None:
        with self.lock:
            job = self.active_job or "complete"
            line = _progress_line(
                self._current(),
                self.total,
                job,
                time.monotonic() - self.started,
            )
            self.stream.write("\r" + line + " " * max(0, self.last_width - len(line)))
            self.stream.flush()
            self.last_width = len(line)

    def start_job(self, job: str) -> None:
        with self.lock:
            self.active_job = job
            self.running = True
            self.started = time.monotonic()
        self.render()

    def complete_job(self) -> None:
        with self.lock:
            self.completed += 1
            self.running = False
        self.render()

    def close(self) -> None:
        self.stop.set()
        self.thread.join()
        self.render()
        self.stream.write("\n")
        self.stream.flush()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError("experiment configuration must be a JSON object")
    return value


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

    data = specification.get("data", {})
    if data.get("kind") != "synthetic" and status == "harness_verification":
        errors.append("harness verification requires synthetic data")
    for field in ("streams", "sequence_tokens", "seed"):
        if not _positive_integer(data.get(field)):
            errors.append(f"data.{field} must be a positive integer")

    attention = specification.get("attention", {})
    for field in ("rolling_tokens", "historical_token_budget"):
        if not _positive_integer(attention.get(field)):
            errors.append(f"attention.{field} must be a positive integer")

    selection = specification.get("selection", {})
    proposal_count = selection.get("proposal_count")
    window_width = selection.get("window_width")
    if not _positive_integer(proposal_count):
        errors.append("selection.proposal_count must be a positive integer")
    if not _positive_integer(window_width):
        errors.append("selection.window_width must be a positive integer")
    budget = attention.get("historical_token_budget")
    if (
        _positive_integer(proposal_count)
        and _positive_integer(window_width)
        and _positive_integer(budget)
        and proposal_count * window_width != budget
    ):
        errors.append("proposal_count * window_width must equal historical_token_budget")

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
        if policy not in {"rolling", "fixed_offsets", "heatfield"}:
            errors.append(f"selection.conditions[{index}].policy is unsupported")
        if policy == "fixed_offsets":
            offsets = condition.get("offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != proposal_count
                or any(not _positive_integer(offset) for offset in offsets)
            ):
                errors.append("fixed_offsets requires one positive offset per proposal")
    if len(names) != len(set(names)):
        errors.append("condition names must be unique")
    if not any(condition.get("policy") == "rolling" for condition in conditions):
        errors.append("a rolling-only condition is mandatory")

    evaluation = specification.get("evaluation", {})
    if evaluation.get("metric") != "next_token_cross_entropy":
        errors.append("evaluation.metric must be next_token_cross_entropy")
    if evaluation.get("paired_conditions") is not True:
        errors.append("evaluation.paired_conditions must be true")
    if evaluation.get("baseline") not in names:
        errors.append("evaluation.baseline must name a configured condition")
    if not _positive_integer(evaluation.get("bootstrap_samples")):
        errors.append("evaluation.bootstrap_samples must be positive")

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

    reasons: list[str] = []
    if status not in {"harness_verification", "approved"}:
        reasons.append("configuration is not authorized for harness execution")
    if backend != "synthetic":
        reasons.append("the frozen Hugging Face reader adapter is not implemented")
    policies = {condition.get("policy") for condition in conditions}
    if not policies <= {"rolling", "fixed_offsets"}:
        reasons.append("learned heat-field selection is not implemented")
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


def _synthetic_streams(specification: dict[str, Any]) -> list[dict[str, Any]]:
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
        streams.append(
            {
                "stream_id": f"synthetic-{stream_index:03d}",
                "tokens": tokens,
                "support_positions": tuple(supports),
                "token_sha256": hashlib.sha256(
                    b"".join(token.to_bytes(4, "little") for token in tokens)
                ).hexdigest(),
            }
        )
    return streams


def _result_path(
    specification: dict[str, Any],
    *,
    output_root: str | Path | None,
) -> Path:
    root = Path(output_root or specification["execution"]["output_root"])
    return root / specification["run"]["id"] / "result.json"


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
    output = _result_path(specification, output_root=output_root)
    expected = {
        "config": specification,
        "config_sha256": _sha256(config_path),
        "source_identity": identity,
        "device": device,
    }
    if output.exists():
        result = _load_json(output)
        actual = {key: result.get(key) for key in expected}
        if actual != expected:
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
            "streams": _synthetic_streams(specification),
            "measurements": [],
            "summary": {},
            "complete": False,
        }
        _write_json(output, result)

    import torch

    from qkvs.model import TokenStream, load_reader
    from qkvs.selection import build_policy

    torch_device = torch.device(device)
    result["environment"].update(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_initialized": torch.cuda.is_initialized(),
        }
    )
    reader = load_reader(specification["model"], torch_device)
    completed = {
        (measurement["condition"], measurement["stream_id"])
        for measurement in result["measurements"]
    }
    units = [
        (condition, stream)
        for condition in specification["selection"]["conditions"]
        for stream in result["streams"]
    ]
    total = len(units)
    progress = _LiveProgress(completed=len(completed), total=total)
    try:
        for condition, stream_record in units:
            unit = (condition["name"], stream_record["stream_id"])
            if unit in completed:
                continue
            progress.start_job(f"{condition['name']} / {stream_record['stream_id']}")
            policy = build_policy(condition, specification["selection"])
            stream = TokenStream(
                stream_id=stream_record["stream_id"],
                tokens=tuple(stream_record["tokens"]),
                support_positions=tuple(stream_record["support_positions"]),
            )
            token_scores = reader.execute(
                stream,
                policy,
                rolling_tokens=specification["attention"]["rolling_tokens"],
                historical_token_budget=specification["attention"]["historical_token_budget"],
                window_width=specification["selection"]["window_width"],
                device=torch_device,
            )
            result["measurements"].append(
                {
                    "condition": condition["name"],
                    "policy": condition["policy"],
                    "stream_id": stream.stream_id,
                    "token_scores": [score.as_dict() for score in token_scores],
                }
            )
            result["measurements"].sort(
                key=lambda measurement: (measurement["condition"], measurement["stream_id"])
            )
            completed.add(unit)
            result["summary"] = summarize_measurements(
                result["measurements"],
                baseline=specification["evaluation"]["baseline"],
                bootstrap_samples=specification["evaluation"]["bootstrap_samples"],
                seed=specification["run"]["seed"],
            )
            result["complete"] = len(completed) == total
            _write_json(output, result)
            progress.complete_job()
    finally:
        progress.close()
    return output, _load_json(output)


def resolve_run(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / "result.json" if candidate.is_dir() else candidate


def load_run(path: str | Path) -> dict[str, Any]:
    result_path = resolve_run(path)
    result = _load_json(result_path)
    if result.get("schema_version") != 1 or "measurements" not in result:
        raise ValueError("run artifact is not a QKVS harness result")
    return result


def status(path: str | Path) -> dict[str, Any]:
    result = load_run(path)
    conditions = result["config"]["selection"]["conditions"]
    total = len(conditions) * len(result["streams"])
    completed = len(result["measurements"])
    return {
        "run_id": result["run_id"],
        "completed_units": completed,
        "total_units": total,
        "complete": result["complete"],
        "source_revision": result["source_identity"]["git_revision"],
        "config_sha256": result["config_sha256"],
        "artifact_sha256": _sha256(resolve_run(path)),
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
    print(f"Run: {value['run_id']}")
    print(f"Source: {value['source_revision']}")
    print(f"Artifact SHA-256: {value['artifact_sha256']}")


def print_summary(result: dict[str, Any]) -> None:
    print(f"Run {result['run_id']} ({'complete' if result['complete'] else 'incomplete'})")
    for name, condition in result["summary"].get("conditions", {}).items():
        print(
            f"{name}: loss={condition['mean_next_token_cross_entropy']:.6f}, "
            f"accuracy={condition['accuracy']:.3f}, "
            f"historical_reads={condition['retrieval']['historical_token_occurrences']}"
        )
    for name, comparison in result["summary"].get("paired_utility", {}).items():
        interval = comparison["paired_bootstrap_95_interval_nats"]
        print(
            f"{name}: paired utility={comparison['mean_utility_nats']:.6f} nat "
            f"95% interval=[{interval[0]:.6f}, {interval[1]:.6f}]"
        )


def dump_json(value: Any) -> None:
    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
