#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_THRESHOLDS = (
    (0.001, "H1a_KL_0_001.svg"),
    (0.01, "H1a_KL_0_01.svg"),
)
_COLOR_STOPS = (
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
)


def _color(value: float) -> str:
    value = min(1.0, max(0.0, value))
    for index in range(len(_COLOR_STOPS) - 1):
        lower_value, lower_color = _COLOR_STOPS[index]
        upper_value, upper_color = _COLOR_STOPS[index + 1]
        if value <= upper_value:
            distance = (value - lower_value) / (upper_value - lower_value)
            rgb = tuple(
                round(lower + distance * (upper - lower))
                for lower, upper in zip(lower_color, upper_color, strict=True)
            )
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    return "#fde725"


def _load(
    path: Path,
    threshold: float,
) -> tuple[list[int], list[float], dict[tuple[int, float], dict[str, str]]]:
    cells: dict[tuple[int, float], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["metric"] != "kl_divergence_nats":
                continue
            if float(row["threshold"]) != threshold:
                continue
            position = int(row["position"])
            coverage = float(row["requested_sampling_percentage"])
            key = (position, coverage)
            if key in cells:
                raise ValueError(f"duplicate feasibility cell: {key}")
            cells[key] = row
    positions = sorted({position for position, _ in cells})
    percentages = sorted({coverage for _, coverage in cells})
    expected = {(position, coverage) for position in positions for coverage in percentages}
    missing = expected - cells.keys()
    if missing:
        raise ValueError(f"missing {len(missing)} feasibility cells")
    return positions, percentages, cells


def _text(x: float, y: float, value: str, **attributes: object) -> str:
    properties = " ".join(f'{name.replace("_", "-")}="{item}"' for name, item in attributes.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {properties}>{escape(value)}</text>'


def _render(
    source: Path,
    destination: Path,
    threshold: float,
) -> None:
    positions, percentages, cells = _load(source, threshold)
    cell_width = 18
    cell_height = 26
    plot_x = 105
    plot_y = 92
    plot_width = cell_width * len(positions)
    plot_height = cell_height * len(percentages)
    colorbar_x = plot_x + plot_width + 42
    width = colorbar_x + 125
    height = plot_y + plot_height + 150
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="title description">'
        ),
        f'<title id="title">H1a candidate success rate for KL at most {threshold:g} nat</title>',
        (
            '<desc id="description">Heatmap of random sparse candidates. Columns are sampled token '
            "positions, rows are requested K/V coverage percentages, and color is the percentage "
            "of candidates whose KL divergence from full QKV is within the stated tolerance.</desc>"
        ),
        '<rect width="100%" height="100%" fill="#10151c"/>',
        _text(
            width / 2,
            30,
            f"H1a random-block feasibility — KL ≤ {threshold:g} nat",
            fill="#f2f5f7",
            font_family="sans-serif",
            font_size="20",
            font_weight="600",
            text_anchor="middle",
        ),
        _text(
            width / 2,
            54,
            "Heat = successful candidates / attempts; categorical columns show absolute token "
            "positions",
            fill="#aeb8c2",
            font_family="sans-serif",
            font_size="12",
            text_anchor="middle",
        ),
    ]

    rendered_percentages = list(reversed(percentages))
    for row_index, coverage in enumerate(rendered_percentages):
        y = plot_y + row_index * cell_height
        parts.append(
            _text(
                plot_x - 10,
                y + cell_height * 0.68,
                f"{coverage:g}%",
                fill="#d9e0e6",
                font_family="sans-serif",
                font_size="11",
                text_anchor="end",
            )
        )
        for column_index, position in enumerate(positions):
            row = cells[(position, coverage)]
            probability = float(row["success_probability"])
            x = plot_x + column_index * cell_width
            title = (
                f"token {position}; coverage {coverage:g}%; "
                f"{row['success_count']}/{row['attempts']} hits ({100.0 * probability:.2f}%)"
            )
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" '
                f'fill="{_color(probability)}" stroke="#10151c" stroke-width="0.5">'
                f"<title>{escape(title)}</title></rect>"
            )

    tick_indices = sorted({0, len(positions) - 1, *range(4, len(positions), 5)})
    for index in tick_indices:
        x = plot_x + (index + 0.5) * cell_width
        y = plot_y + plot_height + 10
        parts.append(
            f'<g transform="translate({x:.2f},{y:.2f}) rotate(55)">'
            + _text(
                0,
                0,
                str(positions[index]),
                fill="#c6ced6",
                font_family="monospace",
                font_size="10",
                text_anchor="start",
            )
            + "</g>"
        )

    parts.extend(
        [
            _text(
                plot_x + plot_width / 2,
                height - 18,
                "Sampled token position",
                fill="#f2f5f7",
                font_family="sans-serif",
                font_size="13",
                text_anchor="middle",
            ),
            (
                f'<text x="24" y="{plot_y + plot_height / 2:.2f}" fill="#f2f5f7" '
                'font-family="sans-serif" font-size="13" text-anchor="middle" '
                f'transform="rotate(-90 24 {plot_y + plot_height / 2:.2f})">'
                "Requested K/V coverage</text>"
            ),
            _text(
                colorbar_x + 10,
                plot_y - 16,
                "Hit rate",
                fill="#f2f5f7",
                font_family="sans-serif",
                font_size="12",
                text_anchor="middle",
            ),
        ]
    )
    colorbar_steps = 100
    for index in range(colorbar_steps):
        probability = 1.0 - index / (colorbar_steps - 1)
        y = plot_y + index * plot_height / colorbar_steps
        parts.append(
            f'<rect x="{colorbar_x}" y="{y:.2f}" width="20" '
            f'height="{plot_height / colorbar_steps + 0.5:.2f}" fill="{_color(probability)}"/>'
        )
    for probability in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = plot_y + (1.0 - probability) * plot_height
        parts.append(
            _text(
                colorbar_x + 28,
                y + 4,
                f"{100 * probability:.0f}%",
                fill="#d9e0e6",
                font_family="sans-serif",
                font_size="10",
            )
        )
    parts.append("</svg>")
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render H1a KL feasibility heatmaps as SVG")
    parser.add_argument(
        "--data",
        type=Path,
        default=_HERE / "H1a_feasibility.csv",
        help="feasibility CSV produced by qkvs export-snapshot",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=_HERE,
        help="directory for the two generated SVG files",
    )
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    for threshold, filename in _THRESHOLDS:
        destination = arguments.output_directory / filename
        _render(arguments.data, destination, threshold)
        print(destination)


if __name__ == "__main__":
    main()
