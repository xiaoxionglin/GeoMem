"""Analyze recurrent matrix families used by the GeoMem reservoir benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

from plot_current_results import esc
from rnn_navigation_benchmark import RESERVOIR_KINDS, make_reservoir_matrix, matrix_metrics


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def collect(hidden_dim: int = 48) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind in RESERVOIR_KINDS:
        matrix = make_reservoir_matrix(kind, hidden_dim=hidden_dim, seed=17)
        rows.append({
            "matrix_kind": kind,
            "hidden_dim": hidden_dim,
            **matrix_metrics(matrix),
        })
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    path = FIGURES / "reservoir_matrix_analysis.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def heatmap(rows: list[dict[str, object]]) -> str:
    metrics = [
        "spectral_radius",
        "symmetry_index",
        "normality_index",
        "effective_rank",
        "transient_amplification",
        "power_decay_10",
        "oscillatory_score",
        "nilpotent_score",
    ]
    kinds = [str(row["matrix_kind"]) for row in rows]
    values = {(str(row["matrix_kind"]), metric): float(row[metric]) for row in rows for metric in metrics}
    metric_max = {metric: max(values[(kind, metric)] for kind in kinds) for metric in metrics}
    width, height = 1120, 560
    margin_left, margin_right, margin_top, margin_bottom = 180, 50, 90, 120
    cell_w = (width - margin_left - margin_right) / len(metrics)
    cell_h = (height - margin_top - margin_bottom) / len(kinds)

    def color(raw_value: float, metric: str) -> str:
        denom = metric_max[metric] if metric_max[metric] > 1e-9 else 1.0
        value = max(0.0, min(1.0, raw_value / denom))
        low = (239, 246, 255)
        high = (29, 78, 216)
        r = round(low[0] + (high[0] - low[0]) * value)
        g = round(low[1] + (high[1] - low[1]) * value)
        b = round(low[2] + (high[2] - low[2]) * value)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:19px}.tick{font-size:17px;fill:#52606d}.cell{font-size:17px;font-weight:700}</style>',
        '<text class="title" x="70" y="38">Reservoir recurrent matrix diagnostics</text>',
        '<text class="label" x="70" y="62">Cells are normalized within each metric for visual comparison</text>',
    ]
    for row_idx, kind in enumerate(kinds):
        y0 = margin_top + row_idx * cell_h
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 14}" y="{y0 + cell_h / 2 + 5:.1f}">{esc(kind.replace("_", " "))}</text>')
        for col_idx, metric in enumerate(metrics):
            raw_value = values[(kind, metric)]
            x0 = margin_left + col_idx * cell_w
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{color(raw_value, metric)}" stroke="#ffffff" stroke-width="2"/>')
            parts.append(f'<text class="cell" text-anchor="middle" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 5:.1f}">{raw_value:.2f}</text>')
    for col_idx, metric in enumerate(metrics):
        x0 = margin_left + col_idx * cell_w + cell_w / 2
        parts.append(f'<text class="tick" text-anchor="end" x="{x0 + 24:.1f}" y="{height - 44}" transform="rotate(-35 {x0 + 24:.1f} {height - 44})">{esc(metric.replace("_", " "))}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = collect()
    write_csv(rows)
    path = FIGURES / "reservoir_matrix_analysis.svg"
    path.write_text(heatmap(rows), encoding="utf-8")
    print("Generated:")
    print((FIGURES / "reservoir_matrix_analysis.csv").relative_to(ROOT))
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
