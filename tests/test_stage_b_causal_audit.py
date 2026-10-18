from __future__ import annotations

import json
import os

import pytest

from qkvs.stage_b_causal_audit import run_causal_audit


@pytest.mark.skipif(
    os.environ.get("QKVS_RUN_ACTUAL_CHECKPOINT_TEST") != "1",
    reason="set QKVS_RUN_ACTUAL_CHECKPOINT_TEST=1 to load the pinned 0.5B checkpoint",
)
def test_actual_checkpoint_incremental_execution_matches_causal_batch() -> None:
    with open("configs/retrieval_utility_stage_a_v3.json") as source:
        specification = json.load(source)
    result = run_causal_audit(specification, sequence_tokens=96)
    assert result["passed"], result
