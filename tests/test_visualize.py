from __future__ import annotations

import csv
import runpy


def test_heatmap_uses_ignored_history_axis_without_measurement_dots(tmp_path) -> None:
    module = runpy.run_path(
        "docs/data/h1a-context-density-expansion/visualize.py"
    )
    source = tmp_path / "feasibility.csv"
    rows = [
        {
            "metric": "kl_divergence_nats",
            "threshold": 0.01,
            "position": position,
            "requested_sampling_percentage": coverage,
            "success_count": successes,
            "attempts": 10,
            "success_probability": successes / 10,
        }
        for position, coverage, successes in ((10, 10, 1), (20, 20, 5))
    ]
    with source.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    extension = tmp_path / "extension.csv"
    extension_rows = [
        {**rows[0], "success_count": 3, "success_probability": 0.3},
        {
            **rows[0],
            "position": 30,
            "success_count": 2,
            "success_probability": 0.2,
        },
    ]
    with extension.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=extension_rows[0])
        writer.writeheader()
        writer.writerows(extension_rows)

    positions, percentages, cells = module["_load"]([source, extension], 0.01)
    destination = tmp_path / "heatmap.svg"
    module["_render_svg"]([source, extension], destination, 0.01)
    rendered = destination.read_text()

    assert module["_available_thresholds"]([source, extension]) == {0.01}
    assert positions == [10, 20, 30]
    assert percentages == [10.0, 20.0, 100.0]
    assert float(cells[(10, 10.0)]["success_probability"]) == 0.2
    assert (20, 10.0) not in cells
    assert (10, 20.0) not in cells
    assert all(
        float(cells[(position, 100.0)]["success_probability"]) == 1.0
        for position in positions
    )
    assert "K/V history ignored" in rendered
    assert "Interpretive interpolation" in rendered
    assert "<circle" not in rendered
    assert ">0%</text>" in rendered
    assert ">90%</text>" in rendered
