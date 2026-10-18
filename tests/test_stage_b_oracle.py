from __future__ import annotations

import pytest

from qkvs.stage_b_oracle import _validate_full_audit, summarize


def _example(causal_loss: float) -> dict:
    rolling_loss = 1.0
    return {
        "modes": {
            "rolling": {
                "loss": rolling_loss,
                "correct": False,
                "utility_nats": 0.0,
            },
            "causal_oracle": {
                "loss": causal_loss,
                "correct": causal_loss < rolling_loss,
                "utility_nats": rolling_loss - causal_loss,
            },
            "stage_a_oracle": {
                "loss": 0.1,
                "correct": True,
                "utility_nats": 0.9,
            },
        }
    }


def test_causal_oracle_gate_permits_s_training_only_after_complete_success() -> None:
    summary = summarize(
        [_example(0.2)] * 4,
        configured_examples=4,
        seed=9,
        minimum_accuracy=0.5,
        minimum_lower_bound=0.05,
    )
    assert summary["gate"]["passed"]
    assert summary["gate"]["decision"] == "permit_s_training"
    assert summary["causal_oracle_recovered_stage_a_utility"] == 0.8 / 0.9


def test_causal_oracle_gate_stops_before_s_training_on_no_utility() -> None:
    summary = summarize(
        [_example(1.0)] * 4,
        configured_examples=4,
        seed=9,
        minimum_accuracy=0.5,
        minimum_lower_bound=0.05,
    )
    assert not summary["gate"]["passed"]
    assert summary["gate"]["decision"] == "stop_before_s_training"


def test_oracle_gate_requires_matching_successful_full_length_audit() -> None:
    specification = {"stage": "B"}
    identity = {"git_revision": "approved"}
    audit = {"passed": True, "config": specification, "source_identity": identity}
    _validate_full_audit(audit, specification=specification, identity=identity)
    with pytest.raises(ValueError, match="source identity"):
        _validate_full_audit(
            {**audit, "source_identity": {"git_revision": "different"}},
            specification=specification,
            identity=identity,
        )
