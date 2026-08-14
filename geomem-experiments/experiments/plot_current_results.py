"""Generate dependency-free SVG overview plots for current GeoMem results."""

from __future__ import annotations

import csv
from math import isnan, sqrt
from pathlib import Path

from commutativity_probe import (
    Environment,
    default_environments,
    effective_commutativity,
    pairwise_commutator_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


COLORS = {
    "torus_lattice": "#1f5fbf",
    "clamped_lattice": "#5a8dee",
    "bottleneck_rooms": "#2a9d8f",
    "tree": "#d45d5d",
    "shortcut_tree_025": "#f0c04a",
    "shortcut_tree_050": "#e9a227",
    "shortcut_tree_100": "#c87812",
    "trace_commute_000": "#7b2cbf",
    "trace_commute_033": "#9d4edd",
    "trace_commute_067": "#c77dff",
    "trace_commute_100": "#4cc9f0",
}


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def label(name: str) -> str:
    return name.replace("_", " ")


def mean(values: list[float]) -> float:
    clean = [value for value in values if not isnan(value)]
    return sum(clean) / len(clean) if clean else float("nan")


def stdev(values: list[float]) -> float:
    clean = [value for value in values if not isnan(value)]
    if len(clean) < 2:
        return 0.0
    m = mean(clean)
    return sqrt(sum((value - m) ** 2 for value in clean) / (len(clean) - 1))


def collect_results() -> tuple[list[Environment], list[int], dict[str, dict[str, dict[int, list[float]]]]]:
    envs = default_environments()
    lengths = [2, 3, 4, 5, 6, 7, 8]
    seeds = list(range(1, 11))
    results: dict[str, dict[str, dict[int, list[float]]]] = {}

    for env in envs:
        results[env.name] = {"raw": {}, "valid_only": {}}
        for length in lengths:
            for mode, valid_only in [("raw", False), ("valid_only", True)]:
                results[env.name][mode][length] = [
                    effective_commutativity(
                        env,
                        length=length,
                        n_sequences=500,
                        seed=seed,
                        valid_only=valid_only,
                    )
                    for seed in seeds
                ]
    return envs, lengths, results


def write_csv(envs: list[Environment], lengths: list[int], results: dict[str, dict[str, dict[int, list[float]]]]) -> None:
    path = FIGURES / "current_commutativity_results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["environment", "mode", "sequence_length", "mean", "std", "n_seeds"])
        for env in envs:
            for mode in ["raw", "valid_only"]:
                for length in lengths:
                    values = results[env.name][mode][length]
                    writer.writerow([env.name, mode, length, f"{mean(values):.6f}", f"{stdev(values):.6f}", len(values)])

    matrix_path = FIGURES / "pairwise_commutator_matrices.csv"
    with matrix_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["environment", "mode", "action_a", "action_b", "score"])
        for env in envs:
            for mode, valid_only in [("raw", False), ("valid_only", True)]:
                matrix = pairwise_commutator_matrix(env, valid_only=valid_only)
                for first in env.actions:
                    for second in env.actions:
                        value = matrix[(first, second)]
                        writer.writerow([env.name, mode, first, second, f"{value:.6f}" if not isnan(value) else "nan"])


def y_scale(value: float, baseline: float, plot_h: float) -> float:
    return baseline - max(0.0, min(1.0, value)) * plot_h


def bar_plot(envs: list[Environment], results: dict[str, dict[str, dict[int, list[float]]]]) -> str:
    width, height = 1180, 760
    margin_left, margin_right, margin_top, margin_bottom = 95, 45, 78, 295
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    baseline = margin_top + plot_h
    group_w = plot_w / len(envs)
    bar_w = 34
    length = 4
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:19px}.tick{font-size:17px;fill:#52606d}.value{font-size:17px;font-weight:700}</style>',
        '<text class="title" x="95" y="38">Raw vs boundary-filtered commutativity</text>',
        '<text class="label" x="95" y="60">Sequence length 4, mean across 10 seeds. Filtered score ignores trajectories that hit invalid transitions.</text>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = y_scale(tick, baseline, plot_h)
        parts.append(f'<line x1="{margin_left}" y1="{yy:.1f}" x2="{width - margin_right}" y2="{yy:.1f}" stroke="#e4e7eb"/>')
        parts.append(f'<text class="tick" x="55" y="{yy + 4:.1f}">{tick:.2f}</text>')
    parts.append(f'<line x1="{margin_left}" y1="{baseline}" x2="{width - margin_right}" y2="{baseline}" stroke="#1f2933"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{baseline}" stroke="#1f2933"/>')
    parts.append(f'<text class="label" x="25" y="300" transform="rotate(-90 25 300)">commutativity score</text>')

    for idx, env in enumerate(envs):
        center = margin_left + group_w * idx + group_w / 2
        base_color = COLORS.get(env.name, "#52606d")
        for offset, mode, opacity in [(-bar_w / 2, "raw", "0.45"), (bar_w / 2, "valid_only", "1.0")]:
            values = results[env.name][mode][length]
            m, s = mean(values), stdev(values)
            x = center + offset - bar_w / 2
            yy = y_scale(m, baseline, plot_h)
            h = baseline - yy
            parts.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w}" height="{h:.1f}" fill="{base_color}" opacity="{opacity}" rx="3"/>')
            err_top = y_scale(min(1.0, m + s), baseline, plot_h)
            err_bot = y_scale(max(0.0, m - s), baseline, plot_h)
            cx = x + bar_w / 2
            parts.append(f'<line x1="{cx:.1f}" y1="{err_top:.1f}" x2="{cx:.1f}" y2="{err_bot:.1f}" stroke="#1f2933" stroke-width="1.5"/>')
            parts.append(f'<text class="value" text-anchor="middle" x="{cx:.1f}" y="{yy - 8:.1f}">{m:.2f}</text>')
        parts.append(f'<text class="tick" text-anchor="end" x="{center + 18:.1f}" y="{baseline + 92:.1f}" transform="rotate(-38 {center + 18:.1f} {baseline + 92:.1f})">{esc(label(env.name))}</text>')

    legend_y = height - 42
    parts.append(f'<rect x="95" y="{legend_y - 13}" width="20" height="14" fill="#52606d" opacity="0.45" rx="2"/>')
    parts.append(f'<text class="label" x="123" y="{legend_y}">raw</text>')
    parts.append(f'<rect x="175" y="{legend_y - 13}" width="20" height="14" fill="#52606d" opacity="1.0" rx="2"/>')
    parts.append(f'<text class="label" x="203" y="{legend_y}">valid-only</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def line_plot(envs: list[Environment], lengths: list[int], results: dict[str, dict[str, dict[int, list[float]]]]) -> str:
    width, height = 1040, 610
    margin_left, margin_right, margin_top, margin_bottom = 90, 210, 75, 75
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    baseline = margin_top + plot_h
    x = lambda length: margin_left + (length - min(lengths)) / (max(lengths) - min(lengths)) * plot_w
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:19px}.tick{font-size:17px;fill:#52606d}</style>',
        '<text class="title" x="90" y="38">Valid-only commutativity across sequence length</text>',
        '<text class="label" x="90" y="60">Boundary-filtered score separates geometric order effects from clamped invalid moves</text>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = y_scale(tick, baseline, plot_h)
        parts.append(f'<line x1="{margin_left}" y1="{yy:.1f}" x2="{width - margin_right}" y2="{yy:.1f}" stroke="#e4e7eb"/>')
        parts.append(f'<text class="tick" x="50" y="{yy + 4:.1f}">{tick:.2f}</text>')
    for length in lengths:
        xx = x(length)
        parts.append(f'<line x1="{xx:.1f}" y1="{baseline}" x2="{xx:.1f}" y2="{baseline + 6}" stroke="#1f2933"/>')
        parts.append(f'<text class="tick" text-anchor="middle" x="{xx:.1f}" y="{baseline + 24}">{length}</text>')
    parts.append(f'<line x1="{margin_left}" y1="{baseline}" x2="{width - margin_right}" y2="{baseline}" stroke="#1f2933"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{baseline}" stroke="#1f2933"/>')
    parts.append(f'<text class="label" text-anchor="middle" x="{margin_left + plot_w / 2:.1f}" y="{height - 25}">action-sequence length</text>')
    parts.append(f'<text class="label" x="25" y="300" transform="rotate(-90 25 300)">commutativity score</text>')

    for env in envs:
        color = COLORS.get(env.name, "#52606d")
        points = []
        for length in lengths:
            value = mean(results[env.name]["valid_only"][length])
            if not isnan(value):
                points.append((x(length), y_scale(value, baseline, plot_h)))
        if len(points) > 1:
            point_str = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)
            parts.append(f'<polyline points="{point_str}" fill="none" stroke="{color}" stroke-width="3"/>')
        for xx, yy in points:
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4.2" fill="{color}" stroke="#ffffff" stroke-width="1.4"/>')

    legend_x = width - margin_right + 25
    legend_y = margin_top + 18
    for idx, env in enumerate(envs):
        yy = legend_y + idx * 27
        parts.append(f'<rect x="{legend_x}" y="{yy - 11}" width="16" height="16" fill="{COLORS.get(env.name, "#52606d")}" rx="2"/>')
        parts.append(f'<text class="label" x="{legend_x + 24}" y="{yy + 2}">{esc(label(env.name))}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def heatmap(envs: list[Environment], lengths: list[int], results: dict[str, dict[str, dict[int, list[float]]]]) -> str:
    width, height = 980, 620
    margin_left, margin_right, margin_top, margin_bottom = 190, 80, 80, 70
    cell_w = (width - margin_left - margin_right) / len(lengths)
    cell_h = (height - margin_top - margin_bottom) / len(envs)

    def color(value: float) -> str:
        if isnan(value):
            return "#f5f7fa"
        low = (235, 244, 255)
        high = (31, 90, 180)
        r = round(low[0] + (high[0] - low[0]) * value)
        g = round(low[1] + (high[1] - low[1]) * value)
        b = round(low[2] + (high[2] - low[2]) * value)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:19px}.tick{font-size:17px;fill:#52606d}.cell{font-size:18px;font-weight:700}</style>',
        '<text class="title" x="90" y="38">Valid-only commutativity heatmap</text>',
        '<text class="label" x="90" y="60">Rows are environment families; columns are action-sequence lengths</text>',
    ]
    for row, env in enumerate(envs):
        y0 = margin_top + row * cell_h
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 14}" y="{y0 + cell_h / 2 + 5:.1f}">{esc(label(env.name))}</text>')
        for col, length in enumerate(lengths):
            value = mean(results[env.name]["valid_only"][length])
            x0 = margin_left + col * cell_w
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{color(value)}" stroke="#ffffff" stroke-width="2"/>')
            text = "NA" if isnan(value) else f"{value:.2f}"
            text_color = "#ffffff" if not isnan(value) and value > 0.62 else "#1f2933"
            parts.append(f'<text class="cell" text-anchor="middle" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 5:.1f}" fill="{text_color}">{text}</text>')
    for col, length in enumerate(lengths):
        x0 = margin_left + col * cell_w + cell_w / 2
        parts.append(f'<text class="tick" text-anchor="middle" x="{x0:.1f}" y="{height - 35}">{length}</text>')
    parts.append(f'<text class="label" text-anchor="middle" x="{margin_left + cell_w * len(lengths) / 2:.1f}" y="{height - 12}">action-sequence length</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def matrix_plot(env: Environment, *, valid_only: bool) -> str:
    matrix = pairwise_commutator_matrix(env, valid_only=valid_only)
    n = len(env.actions)
    width = max(520, 160 + n * 62)
    height = max(500, 150 + n * 62)
    margin_left, margin_top = 110, 95
    cell = min(62, (width - margin_left - 45) / n, (height - margin_top - 45) / n)

    def color(value: float) -> str:
        if isnan(value):
            return "#f5f7fa"
        low = (253, 237, 217)
        high = (196, 78, 82)
        r = round(low[0] + (high[0] - low[0]) * value)
        g = round(low[1] + (high[1] - low[1]) * value)
        b = round(low[2] + (high[2] - low[2]) * value)
        return f"rgb({r},{g},{b})"

    mode_label = "valid-only" if valid_only else "raw"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:30px;font-weight:700}.label{font-size:18px}.cell{font-size:17px;font-weight:700}</style>',
        f'<text class="title" x="35" y="35">Pairwise commutator: {esc(label(env.name))}</text>',
        f'<text class="label" x="35" y="58">Mode: {mode_label}; cell = Pr[AB(s) = BA(s)]</text>',
    ]
    for col, action in enumerate(env.actions):
        x0 = margin_left + col * cell + cell / 2
        parts.append(f'<text class="label" text-anchor="middle" x="{x0:.1f}" y="{margin_top - 14}">{esc(action)}</text>')
    for row, first in enumerate(env.actions):
        y0 = margin_top + row * cell + cell / 2
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 14}" y="{y0 + 4:.1f}">{esc(first)}</text>')
        for col, second in enumerate(env.actions):
            value = matrix[(first, second)]
            x = margin_left + col * cell
            y = margin_top + row * cell
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{cell:.1f}" fill="{color(value)}" stroke="#ffffff" stroke-width="2"/>')
            text = "NA" if isnan(value) else f"{value:.2f}"
            text_color = "#ffffff" if not isnan(value) and value > 0.62 else "#1f2933"
            parts.append(f'<text class="cell" text-anchor="middle" x="{x + cell / 2:.1f}" y="{y + cell / 2 + 4:.1f}" fill="{text_color}">{text}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(name: str, content: str) -> None:
    (FIGURES / name).write_text(content, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    envs, lengths, results = collect_results()
    write_csv(envs, lengths, results)
    write_svg("commutativity_bar.svg", bar_plot(envs, results))
    write_svg("commutativity_by_length.svg", line_plot(envs, lengths, results))
    write_svg("commutativity_heatmap.svg", heatmap(envs, lengths, results))
    for env in envs:
        write_svg(f"commutator_matrix_{env.name}_raw.svg", matrix_plot(env, valid_only=False))
        write_svg(f"commutator_matrix_{env.name}_valid.svg", matrix_plot(env, valid_only=True))

    print("Generated:")
    for path in [
        FIGURES / "current_commutativity_results.csv",
        FIGURES / "pairwise_commutator_matrices.csv",
        FIGURES / "commutativity_bar.svg",
        FIGURES / "commutativity_by_length.svg",
        FIGURES / "commutativity_heatmap.svg",
    ]:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
