from __future__ import annotations

import json
import os

import pytest

from qkvs.checkpoint_mask_audit import run_mask_equivalence


@pytest.mark.skipif(
    os.environ.get("QKVS_RUN_ACTUAL_CHECKPOINT_TEST") != "1",
    reason="set QKVS_RUN_ACTUAL_CHECKPOINT_TEST=1 to load the pinned 0.5B checkpoint",
)
def test_actual_checkpoint_explicit_mask_matches_builtin_causal_mask() -> None:
    with open("configs/retrieval_utility_stage_a_v3.json") as source:
        specification = json.load(source)
    result = run_mask_equivalence(specification, sequence_tokens=128)
    assert result["passed"], result
