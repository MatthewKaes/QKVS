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
            "success_probability": probability,
        }
        for position, probabilities in ((10, (0.1, 0.4)), (20, (0.2, 0.5)))
        for coverage, probability in zip((10, 20), probabilities, strict=True)
    ]
    with source.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    positions, percentages, cells = module["_load"](source, 0.01)
    destination = tmp_path / "heatmap.svg"
    module["_render_svg"](source, destination, 0.01)
    rendered = destination.read_text()

    assert positions == [10, 20]
    assert percentages == [10.0, 20.0, 100.0]
    assert all(
        float(cells[(position, 100.0)]["success_probability"]) == 1.0
        for position in positions
    )
    assert "K/V history ignored" in rendered
    assert "Interpretive interpolation" in rendered
    assert "<circle" not in rendered
    assert ">0%</text>" in rendered
    assert ">90%</text>" in rendered
