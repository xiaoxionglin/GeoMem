"""Temporal-basis benchmark for nilpotent and oscillatory memory."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

DELAYS = tuple(range(4, 97, 4))
MAX_LAG = 96
N_TRAIN = 1200
N_TEST = 1200
DISTRACTOR_P = 0.15
RIDGE = 1e-3
SEED = 7


@dataclass(frozen=True)
class BasisSpec:
    name: str
    family: str
    dim: int
    span: int | None = None
    radius: float | None = None
    modes: int | None = None
    threshold: float | None = None


BASIS_SPECS = (
    BasisSpec("nilpotent_L32", "nilpotent", dim=32, span=32),
    BasisSpec("nilpotent_L64", "nilpotent", dim=64, span=64),
    BasisSpec("nilpotent_L96", "nilpotent", dim=96, span=96),
    BasisSpec("osc_r099_m16", "oscillatory", dim=32, radius=0.99, modes=16),
    BasisSpec("osc_r095_m16", "oscillatory", dim=32, radius=0.95, modes=16),
    BasisSpec("osc_r090_m16", "oscillatory", dim=32, radius=0.90, modes=16),
    BasisSpec("osc_r099_m48", "oscillatory", dim=96, radius=0.99, modes=48),
    BasisSpec("threshold_osc_r099_m48", "thresholded_oscillatory", dim=96, radius=0.99, modes=48, threshold=0.08),
    BasisSpec("matched_phase_osc_T16", "matched_phase_oscillatory", dim=2, radius=1.0, modes=1),
)


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def lag_codes(spec: BasisSpec, max_lag: int = MAX_LAG) -> np.ndarray:
    """Rows are lag codes phi_k = A^k B."""
    if spec.family == "nilpotent":
        assert spec.span is not None
        codes = np.zeros((max_lag + 1, spec.span), dtype=np.float64)
        for lag in range(min(max_lag + 1, spec.span)):
            codes[lag, lag] = 1.0
        return codes

    if spec.family == "matched_phase_oscillatory":
        lags = np.arange(max_lag + 1, dtype=np.float64)
        omega = 2 * math.pi / 16
        return np.stack([np.cos(omega * lags), np.sin(omega * lags)], axis=1)

    if spec.family in {"oscillatory", "thresholded_oscillatory"}:
        assert spec.radius is not None and spec.modes is not None
        # Avoid exact zero and Nyquist modes; use a broad Fourier-like bank.
        freqs = np.linspace(math.pi / (2 * spec.modes), math.pi - math.pi / (2 * spec.modes), spec.modes)
        lags = np.arange(max_lag + 1, dtype=np.float64)[:, None]
        envelope = spec.radius ** lags
        cos = envelope * np.cos(lags * freqs[None, :])
        sin = envelope * np.sin(lags * freqs[None, :])
        codes = np.concatenate([cos, sin], axis=1) / math.sqrt(spec.modes)
        if spec.family == "thresholded_oscillatory":
            assert spec.threshold is not None
            codes = np.where(np.abs(codes) >= spec.threshold, codes, 0.0)
        return codes

    raise ValueError(f"unknown basis family: {spec.family}")


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    eye = np.eye(x_aug.shape[1], dtype=x.dtype)
    eye[-1, -1] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + ridge * eye, x_aug.T @ y)


def ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ w


def delayed_cue_dataset(codes: np.ndarray, delay: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Binary cue at lag D with random signed distractors at more recent lags."""
    dim = codes.shape[1]
    x = np.zeros((n, dim), dtype=np.float64)
    y = rng.choice(np.array([-1.0, 1.0]), size=n)
    x += y[:, None] * codes[delay][None, :]
    if delay > 0:
        distractor_mask = rng.random((n, delay)) < DISTRACTOR_P
        distractor_values = rng.choice(np.array([-1.0, 1.0]), size=(n, delay))
        for lag in range(delay):
            x += (distractor_mask[:, lag] * distractor_values[:, lag])[:, None] * codes[lag][None, :]
    x += 0.02 * rng.normal(size=x.shape)
    return x, y


def delayed_cue_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in BASIS_SPECS:
        codes = lag_codes(spec)
        for delay in DELAYS:
            rng = np.random.default_rng(SEED + delay + 1000 * BASIS_SPECS.index(spec))
            x_train, y_train = delayed_cue_dataset(codes, delay, N_TRAIN, rng)
            x_test, y_test = delayed_cue_dataset(codes, delay, N_TEST, rng)
            w = ridge_fit(x_train, y_train)
            pred = np.sign(ridge_predict(x_test, w))
            pred[pred == 0] = 1
            accuracy = float(np.mean(pred == y_test))
            target_norm = float(np.linalg.norm(codes[delay]))
            adjacent = float("nan")
            if delay > 0 and np.linalg.norm(codes[delay - 1]) > 1e-9 and target_norm > 1e-9:
                adjacent = float(np.dot(codes[delay], codes[delay - 1]) / (target_norm * np.linalg.norm(codes[delay - 1])))
            rows.append({
                "task": "delayed_cue_with_distractors",
                "basis": spec.name,
                "family": spec.family,
                "dim": spec.dim,
                "delay": delay,
                "accuracy": accuracy,
                "target_norm": target_norm,
                "adjacent_lag_similarity": adjacent,
            })
    return rows


def phase_dataset(codes: np.ndarray, max_delay: int, period: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    delays = rng.integers(0, max_delay + 1, size=n)
    x = codes[delays] + 0.01 * rng.normal(size=(n, codes.shape[1]))
    y = np.zeros((n, 2), dtype=np.float64)
    phase = 2 * math.pi * (delays % period) / period
    y[:, 0] = np.cos(phase)
    y[:, 1] = np.sin(phase)
    return x, y


def phase_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    period = 16
    for spec in BASIS_SPECS:
        codes = lag_codes(spec, max_lag=MAX_LAG)
        rng = np.random.default_rng(SEED + 4000 + 1000 * BASIS_SPECS.index(spec))
        x_train, y_train = phase_dataset(codes, MAX_LAG, period, N_TRAIN, rng)
        x_test, y_test = phase_dataset(codes, MAX_LAG, period, N_TEST, rng)
        w = ridge_fit(x_train, y_train)
        pred = ridge_predict(x_test, w)
        mse = float(np.mean((pred - y_test) ** 2))
        corr_num = float(np.sum(pred * y_test))
        corr_den = float(np.linalg.norm(pred) * np.linalg.norm(y_test) + 1e-12)
        rows.append({
            "task": "phase_modulo_reconstruction",
            "basis": spec.name,
            "family": spec.family,
            "dim": spec.dim,
            "period": period,
            "mse": mse,
            "cosine_alignment": corr_num / corr_den,
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222} .title{font-size:28px;font-weight:700} .label{font-size:20px} .small{font-size:16px} .axis{stroke:#222;stroke-width:2} .grid{stroke:#ddd;stroke-width:1}</style>',
    ]


def line_plot(path: Path, title: str, series: dict[str, list[tuple[float, float]]], y_label: str, y_min: float, y_max: float) -> None:
    width, height = 1120, 720
    left, right, top, bottom = 110, 40, 80, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = ["#1b6ca8", "#e4572e", "#2e8b57", "#7b2cbf", "#6c757d", "#d18f00", "#008c95", "#b23a48"]
    max_x = max(x for points in series.values() for x, _ in points)
    min_x = min(x for points in series.values() for x, _ in points)

    def sx(x: float) -> float:
        return left + (x - min_x) / max(1e-9, max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / max(1e-9, y_max - y_min) * plot_h

    lines = svg_header(width, height)
    lines.append(f'<text x="{width / 2}" y="42" text-anchor="middle" class="title">{esc(title)}</text>')
    for tick in np.linspace(y_min, y_max, 6):
        y = sy(float(tick))
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 6:.1f}" text-anchor="end" class="small">{tick:.2f}</text>')
    for tick in np.linspace(min_x, max_x, 7):
        x = sx(float(tick))
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{height - bottom + 30}" text-anchor="middle" class="small">{tick:.0f}</text>')
    lines.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis"/>')
    lines.append(f'<text x="{width / 2}" y="{height - 25}" text-anchor="middle" class="label">delay / lag</text>')
    lines.append(f'<text x="28" y="{height / 2}" text-anchor="middle" transform="rotate(-90 28 {height / 2})" class="label">{esc(y_label)}</text>')

    for idx, (name, points) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        path_data = " ".join(("M" if i == 0 else "L") + f" {sx(x):.1f} {sy(y):.1f}" for i, (x, y) in enumerate(points))
        lines.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="4"/>')
        lx, ly = width - right - 280, top + 32 + idx * 28
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 34}" y2="{ly}" stroke="{color}" stroke-width="5"/>')
        lines.append(f'<text x="{lx + 44}" y="{ly + 6}" class="small">{esc(name)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def heatmap_plot(path: Path, title: str, matrices: list[tuple[str, np.ndarray]]) -> None:
    cell = 10
    n = matrices[0][1].shape[0]
    panel_w = n * cell
    width = 80 + len(matrices) * (panel_w + 80)
    height = 140 + panel_w
    lines = svg_header(width, height)
    lines.append(f'<text x="{width / 2}" y="42" text-anchor="middle" class="title">{esc(title)}</text>')

    def color(value: float) -> str:
        value = max(-1.0, min(1.0, value))
        if value >= 0:
            r = int(255 - 40 * value)
            g = int(255 - 150 * value)
            b = int(255 - 205 * value)
        else:
            r = int(255 + 205 * value)
            g = int(255 + 120 * value)
            b = 255
        return f"#{r:02x}{g:02x}{b:02x}"

    for panel_idx, (name, matrix) in enumerate(matrices):
        x0 = 60 + panel_idx * (panel_w + 80)
        y0 = 90
        lines.append(f'<text x="{x0 + panel_w / 2}" y="74" text-anchor="middle" class="label">{esc(name)}</text>')
        for row in range(n):
            for col in range(n):
                lines.append(
                    f'<rect x="{x0 + col * cell}" y="{y0 + row * cell}" width="{cell}" height="{cell}" fill="{color(float(matrix[row, col]))}"/>'
                )
        lines.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_w}" fill="none" stroke="#222" stroke-width="2"/>')
        lines.append(f'<text x="{x0 + panel_w / 2}" y="{height - 18}" text-anchor="middle" class="small">lag code similarity</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def bar_plot(path: Path, title: str, rows: list[dict[str, object]]) -> None:
    width, height = 1120, 620
    left, right, top, bottom = 100, 40, 85, 150
    plot_w, plot_h = width - left - right, height - top - bottom
    rows = sorted(rows, key=lambda row: float(row["cosine_alignment"]), reverse=True)
    max_y = 1.0
    bar_w = plot_w / len(rows) * 0.7
    lines = svg_header(width, height)
    lines.append(f'<text x="{width / 2}" y="42" text-anchor="middle" class="title">{esc(title)}</text>')
    for tick in np.linspace(0, max_y, 6):
        y = top + (max_y - tick) / max_y * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 6:.1f}" text-anchor="end" class="small">{tick:.1f}</text>')
    for idx, row in enumerate(rows):
        value = float(row["cosine_alignment"])
        x = left + (idx + 0.15) * plot_w / len(rows)
        y = top + (max_y - value) / max_y * plot_h
        color = "#1b6ca8" if row["family"] == "nilpotent" else "#e4572e" if "oscillatory" in str(row["family"]) else "#6c757d"
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height - bottom - y:.1f}" fill="{color}"/>')
        label = str(row["basis"]).replace("_", " ")
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - bottom + 18}" text-anchor="end" transform="rotate(-45 {x + bar_w / 2:.1f} {height - bottom + 18})" class="small">{esc(label)}</text>')
    lines.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis"/>')
    lines.append(f'<text x="28" y="{height / 2}" text-anchor="middle" transform="rotate(-90 28 {height / 2})" class="label">phase alignment</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def normalized_gram(codes: np.ndarray, n_lags: int = 48) -> np.ndarray:
    x = codes[:n_lags].copy()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x, norms, out=np.zeros_like(x), where=norms > 1e-9)
    return x @ x.T


def make_figures(delayed_rows: list[dict[str, object]], phase_summary: list[dict[str, object]]) -> None:
    forget_series = {
        "nilpotent L64": [(lag, 1.0 if lag < 64 else 0.0) for lag in range(MAX_LAG + 1)],
        "osc r=0.99": [(lag, 0.99 ** lag) for lag in range(MAX_LAG + 1)],
        "osc r=0.95": [(lag, 0.95 ** lag) for lag in range(MAX_LAG + 1)],
        "osc r=0.90": [(lag, 0.90 ** lag) for lag in range(MAX_LAG + 1)],
    }
    line_plot(FIGURES / "temporal_basis_forgetting_curves.svg", "Forgetting Curves: Exponential Damping Versus Hard Cutoff", forget_series, "stored amplitude", 0.0, 1.05)

    nil = lag_codes(BasisSpec("nilpotent_L48", "nilpotent", dim=48, span=48), max_lag=47)
    osc = lag_codes(BasisSpec("osc_r099_m24", "oscillatory", dim=48, radius=0.99, modes=24), max_lag=47)
    heatmap_plot(
        FIGURES / "temporal_basis_lag_similarity.svg",
        "Lag-Code Similarity",
        [("nilpotent slots", normalized_gram(nil)), ("oscillatory phases", normalized_gram(osc))],
    )

    selected = ("nilpotent_L64", "nilpotent_L96", "osc_r099_m16", "osc_r095_m16", "osc_r099_m48", "threshold_osc_r099_m48")
    series: dict[str, list[tuple[float, float]]] = {}
    for name in selected:
        points = [(float(row["delay"]), float(row["accuracy"])) for row in delayed_rows if row["basis"] == name]
        series[name.replace("_", " ")] = points
    line_plot(FIGURES / "temporal_basis_delayed_cue_accuracy.svg", "Delayed Cue With Distractors", series, "accuracy", 0.45, 1.02)

    bar_plot(FIGURES / "temporal_basis_phase_task.svg", "Modulo-Phase Reconstruction", phase_summary)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    delayed_rows = delayed_cue_rows()
    phase_summary = phase_rows()
    write_csv(FIGURES / "temporal_basis_delayed_cue.csv", delayed_rows)
    write_csv(FIGURES / "temporal_basis_phase_task.csv", phase_summary)
    make_figures(delayed_rows, phase_summary)
    print("wrote temporal basis benchmark outputs")


if __name__ == "__main__":
    main()
