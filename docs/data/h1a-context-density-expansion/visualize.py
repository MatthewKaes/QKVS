#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import bisect
import csv
import shutil
import struct
import subprocess
import zlib
from collections.abc import Iterable
from html import escape
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_THRESHOLDS = (
    (0.001, "H1a_KL_0_001"),
    (0.01, "H1a_KL_0_01"),
)
_COLOR_STOPS = (
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
)


def _available_thresholds(paths: Iterable[Path]) -> set[float]:
    thresholds = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                if row.get("metric") == "kl_divergence_nats":
                    thresholds.add(float(row["threshold"]))
    return thresholds


def _color(value: float) -> tuple[int, int, int]:
    value = min(1.0, max(0.0, value))
    for index in range(len(_COLOR_STOPS) - 1):
        lower_value, lower_color = _COLOR_STOPS[index]
        upper_value, upper_color = _COLOR_STOPS[index + 1]
        if value <= upper_value:
            distance = (value - lower_value) / (upper_value - lower_value)
            return tuple(
                round(lower + distance * (upper - lower))
                for lower, upper in zip(lower_color, upper_color, strict=True)
            )
    return _COLOR_STOPS[-1][1]


def _hex_color(value: float) -> str:
    red, green, blue = _color(value)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _load(
    paths: Path | Iterable[Path],
    threshold: float,
) -> tuple[list[int], list[float], dict[tuple[int, float], dict[str, str]]]:
    if isinstance(paths, Path):
        paths = (paths,)
    pooled: dict[tuple[int, float], list[int]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                if row.get("metric") in {None, "metric"}:
                    continue
                if row["metric"] != "kl_divergence_nats":
                    continue
                if float(row["threshold"]) != threshold:
                    continue
                position = int(row["position"])
                coverage = float(row["requested_sampling_percentage"])
                successes = int(row["success_count"])
                attempts = int(row["attempts"])
                if attempts <= 0 or not 0 <= successes <= attempts:
                    raise ValueError("invalid feasibility success counts")
                counts = pooled.setdefault((position, coverage), [0, 0])
                counts[0] += successes
                counts[1] += attempts
    if not pooled:
        raise ValueError(f"no KL <= {threshold:g} feasibility cells")
    cells = {
        key: {
            "success_count": str(successes),
            "attempts": str(attempts),
            "success_probability": str(successes / attempts),
        }
        for key, (successes, attempts) in pooled.items()
    }
    positions = sorted({position for position, _ in cells})
    percentages = sorted({coverage for _, coverage in cells})
    for position in positions:
        cells[(position, 100.0)] = {
            "success_count": "1",
            "attempts": "1",
            "success_probability": "1.0",
        }
    if percentages[-1] < 100.0:
        percentages.append(100.0)
    return positions, percentages, cells


def _probability_grid(
    positions: list[int],
    percentages: list[float],
    cells: dict[tuple[int, float], dict[str, str]],
) -> np.ndarray:
    grid = np.full((len(percentages), len(positions)), np.nan, dtype=np.float64)
    position_indices = {position: index for index, position in enumerate(positions)}
    percentage_indices = {percentage: index for index, percentage in enumerate(percentages)}
    for (position, percentage), row in cells.items():
        grid[percentage_indices[percentage], position_indices[position]] = float(
            row["success_probability"]
        )
    missing = np.argwhere(np.isnan(grid))
    if not len(missing):
        return grid

    x_scale = max(1.0, float(positions[-1] - positions[0]))
    y_scale = max(1.0, float(percentages[-1] - percentages[0]))
    measured = np.argwhere(~np.isnan(grid))
    measured_coordinates = np.column_stack(
        (
            (np.asarray(positions)[measured[:, 1]] - positions[0]) / x_scale,
            (np.asarray(percentages)[measured[:, 0]] - percentages[0]) / y_scale,
        )
    )
    measured_values = grid[measured[:, 0], measured[:, 1]]
    neighbor_count = min(8, len(measured))
    for start in range(0, len(missing), 512):
        rows = missing[start : start + 512]
        targets = np.column_stack(
            (
                (np.asarray(positions)[rows[:, 1]] - positions[0]) / x_scale,
                (np.asarray(percentages)[rows[:, 0]] - percentages[0]) / y_scale,
            )
        )
        distances = np.sum(
            (targets[:, None, :] - measured_coordinates[None, :, :]) ** 2,
            axis=2,
        )
        neighbors = np.argpartition(distances, neighbor_count - 1, axis=1)[
            :, :neighbor_count
        ]
        neighbor_distances = np.take_along_axis(distances, neighbors, axis=1)
        weights = 1.0 / np.maximum(neighbor_distances, 1e-12)
        values = measured_values[neighbors]
        grid[rows[:, 0], rows[:, 1]] = np.sum(weights * values, axis=1) / np.sum(
            weights,
            axis=1,
        )
    return grid


def _bracket(values: list[float] | list[int], target: float) -> tuple[int, int, float]:
    upper = bisect.bisect_right(values, target)
    if upper == 0:
        return 0, 0, 0.0
    if upper == len(values):
        last = len(values) - 1
        return last, last, 0.0
    lower = upper - 1
    distance = values[upper] - values[lower]
    return lower, upper, (target - values[lower]) / distance


def _interpolated_raster(
    positions: list[int],
    percentages: list[float],
    cells: dict[tuple[int, float], dict[str, str]],
    *,
    width: int,
    height: int,
) -> bytes:
    x_samples = [
        positions[0] + column * (positions[-1] - positions[0]) / (width - 1)
        for column in range(width)
    ]
    y_samples = [
        percentages[0] + row * (percentages[-1] - percentages[0]) / (height - 1)
        for row in range(height)
    ]
    x_brackets = [_bracket(positions, sample) for sample in x_samples]
    y_brackets = [_bracket(percentages, sample) for sample in y_samples]
    probabilities = _probability_grid(positions, percentages, cells)
    raster_rows = []
    for lower_y, upper_y, y_weight in y_brackets:
        row = bytearray()
        for lower_x, upper_x, x_weight in x_brackets:
            lower = probabilities[lower_y, lower_x] * (
                1.0 - x_weight
            ) + probabilities[lower_y, upper_x] * x_weight
            upper = probabilities[upper_y, lower_x] * (
                1.0 - x_weight
            ) + probabilities[upper_y, upper_x] * x_weight
            row.extend(_color(lower * (1.0 - y_weight) + upper * y_weight))
        raster_rows.append(bytes(row))
    return _png(width, height, raster_rows)


def _png(width: int, height: int, rows: list[bytes]) -> bytes:
    def chunk(name: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + name
            + payload
            + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + row for row in rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def _text(x: float, y: float, value: str, **attributes: object) -> str:
    properties = " ".join(f'{name.replace("_", "-")}="{item}"' for name, item in attributes.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {properties}>{escape(value)}</text>'


def _render_svg(source: Path | Iterable[Path], destination: Path, threshold: float) -> None:
    positions, percentages, cells = _load(source, threshold)
    plot_x = 105
    plot_y = 92
    plot_width = 1080
    plot_height = 390
    colorbar_x = plot_x + plot_width + 42
    width = colorbar_x + 125
    height = plot_y + plot_height + 112
    heat = _interpolated_raster(
        positions,
        percentages,
        cells,
        width=plot_width,
        height=plot_height,
    )
    heat_url = "data:image/png;base64," + base64.b64encode(heat).decode("ascii")

    def x_coordinate(position: float) -> float:
        return plot_x + (position - positions[0]) / (positions[-1] - positions[0]) * plot_width

    def y_coordinate(coverage: float) -> float:
        return (
            plot_y + (coverage - percentages[0]) / (percentages[-1] - percentages[0]) * plot_height
        )

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="title description">'
        ),
        (
            f'<title id="title">H1a interpolated success density for KL at most '
            f"{threshold:g} nat</title>"
        ),
        (
            '<desc id="description">Bilinearly interpolated random sparse candidate success rate. '
            "The horizontal axis is absolute token position and the vertical axis is requested K/V "
            "history ignored. The dense endpoint is fixed at its known 100 percent hit rate.</desc>"
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
            "Interpretive interpolation of successful candidates / attempts",
            fill="#aeb8c2",
            font_family="sans-serif",
            font_size="12",
            text_anchor="middle",
        ),
        (
            f'<image x="{plot_x}" y="{plot_y}" width="{plot_width}" height="{plot_height}" '
            f'href="{heat_url}" preserveAspectRatio="none"/>'
        ),
    ]

    x_ticks = [positions[0], 5000, 10000, 15000, 20000, 25000, 30000, positions[-1]]
    for position in x_ticks:
        x = x_coordinate(position)
        parts.extend(
            [
                (
                    f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" '
                    f'y2="{plot_y + plot_height}" stroke="#ffffff" opacity="0.12"/>'
                ),
                _text(
                    x,
                    plot_y + plot_height + 20,
                    f"{position:,}",
                    fill="#d9e0e6",
                    font_family="sans-serif",
                    font_size="10",
                    text_anchor="middle",
                ),
            ]
        )
    for ignored in range(0, 91, 10):
        coverage = 100.0 - ignored
        y = y_coordinate(coverage)
        parts.extend(
            [
                (
                    f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_width}" '
                    f'y2="{y:.2f}" stroke="#ffffff" opacity="0.12"/>'
                ),
                _text(
                    plot_x - 10,
                    y + 4,
                    f"{ignored}%",
                    fill="#d9e0e6",
                    font_family="sans-serif",
                    font_size="10",
                    text_anchor="end",
                ),
            ]
        )
    parts.extend(
        [
            _text(
                plot_x + plot_width / 2,
                height - 18,
                "Absolute token position",
                fill="#f2f5f7",
                font_family="sans-serif",
                font_size="13",
                text_anchor="middle",
            ),
            (
                f'<text x="24" y="{plot_y + plot_height / 2:.2f}" fill="#f2f5f7" '
                'font-family="sans-serif" font-size="13" text-anchor="middle" '
                f'transform="rotate(-90 24 {plot_y + plot_height / 2:.2f})">'
                "K/V history ignored</text>"
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
            f'height="{plot_height / colorbar_steps + 0.5:.2f}" fill="{_hex_color(probability)}"/>'
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


def _render_png(svg: Path, png: Path) -> None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("rsvg-convert is required to render the labeled PNG output")
    subprocess.run([converter, svg, "-o", png], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render continuous H1a KL feasibility heatmaps as SVG and PNG"
    )
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        help="feasibility CSV to pool; repeat for independently sampled files",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=_HERE,
        help="directory for the generated SVG and PNG files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        help="KL threshold to render; repeat as needed (defaults to available standard thresholds)",
    )
    arguments = parser.parse_args()
    data = arguments.data or [_HERE / "H1a_feasibility.csv"]
    available = _available_thresholds(data)
    thresholds = (
        tuple(
            (
                threshold,
                dict(_THRESHOLDS).get(
                    threshold,
                    f"H1a_KL_{format(threshold, 'g').replace('.', '_')}",
                ),
            )
            for threshold in arguments.threshold
        )
        if arguments.threshold
        else tuple(item for item in _THRESHOLDS if item[0] in available)
    )
    if not thresholds:
        raise ValueError("no requested KL thresholds are present in the feasibility data")
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    for threshold, stem in thresholds:
        svg = arguments.output_directory / f"{stem}.svg"
        png = arguments.output_directory / f"{stem}.png"
        _render_svg(data, svg, threshold)
        _render_png(svg, png)
        print(svg)
        print(png)


if __name__ == "__main__":
    main()
