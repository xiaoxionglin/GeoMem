"""Generate ordered SVG summary figures for the Gaussian navigation report."""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
ENV_SRC = Path(__file__).resolve().parents[2] / "geomem-env" / "src"
if str(ENV_SRC) not in sys.path:
    sys.path.insert(0, str(ENV_SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from commutativity_probe import default_environments, effective_commutativity


OBS_ORDER = ("gaussian_sensory", "vector", "sparse_aliased", "egocentric_landmark", "hybrid")
TEMPORAL_ORDER = ("temporal_full", "temporal_dropout", "temporal_cue_then_blank")
ACTION_INPUT_ORDER = ("none", "one_hot", "embedding")
MODEL_FAMILY_ORDER = ("memoryless", "trained_rnn", "reservoir", "shared_lin_sequence", "independent_lin_block")
LONG_ARCH_ORDER = (
    "memoryless_linear",
    "numpy_rnn",
    "reservoir_diagonal",
    "reservoir_nilpotent",
    "reservoir_block_hybrid",
    "shared_lin_sequence",
    "independent_lin_block",
    "explicit_observed_history",
)

OBS_LABELS = {
    "gaussian_sensory": "Gaussian",
    "vector": "Vector",
    "sparse_aliased": "Sparse",
    "egocentric_landmark": "Egocentric",
    "hybrid": "Hybrid",
}

TEMPORAL_LABELS = {
    "temporal_full": "No masking",
    "temporal_dropout": "Dropout",
    "temporal_cue_then_blank": "Cue blank",
}

ACTION_LABELS = {"none": "No action", "one_hot": "One-hot", "embedding": "Embedding"}

FAMILY_LABELS = {
    "memoryless": "Memoryless",
    "trained_rnn": "RNN",
    "reservoir": "Reservoir",
    "shared_lin_sequence": "Shared Lin",
    "independent_lin_block": "Independent Lin",
}

ARCH_LABELS = {
    "memoryless_linear": "Memoryless",
    "numpy_rnn": "RNN",
    "reservoir_diagonal": "Diag reservoir",
    "reservoir_nilpotent": "Nilpotent res.",
    "reservoir_block_hybrid": "Block hybrid res.",
    "shared_lin_sequence": "Shared Lin",
    "independent_lin_block": "Independent Lin",
    "explicit_observed_history": "Explicit history",
}

FAMILY_COLORS = {
    "memoryless": "#6b7280",
    "trained_rnn": "#2563eb",
    "reservoir": "#15803d",
    "shared_lin_sequence": "#c2410c",
    "independent_lin_block": "#ea580c",
    "explicit_observed_history": "#7c3aed",
    "other": "#64748b",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (FIGURES / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def fmt(value: float, digits: int = 2) -> str:
    return "NA" if value != value else f"{value:.{digits}f}"


def svg_doc(width: int, height: int, body: list[str]) -> str:
    style = (
        "<style>"
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}"
        ".title{font-size:30px;font-weight:700}"
        ".subtitle{font-size:17px;fill:#52606d}"
        ".label{font-size:16px}"
        ".small{font-size:13px;fill:#52606d}"
        ".value{font-size:14px;font-weight:700}"
        ".axis{stroke:#cbd5e1;stroke-width:1}"
        "</style>"
    )
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        style,
        *body,
        "</svg>",
    ])


def interp(low: tuple[int, int, int], high: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    values = [round(low[idx] + (high[idx] - low[idx]) * t) for idx in range(3)]
    return f"rgb({values[0]},{values[1]},{values[2]})"


def short_task(name: str) -> str:
    return name.replace("shortcut_tree_", "st_").replace("room_graph_", "rg_").replace("_", " ")


def model_family(model: str) -> str:
    if model.startswith("memoryless"):
        return "memoryless"
    if model == "numpy_rnn":
        return "trained_rnn"
    if model.startswith("reservoir_"):
        return "reservoir"
    if model.startswith("lin_sequence_"):
        return "shared_lin_sequence"
    if model.startswith("lin_block_"):
        return "independent_lin_block"
    if model.startswith("history_"):
        return "explicit_observed_history"
    return "other"


def ordered_environments() -> list[dict[str, object]]:
    rows = []
    for env in default_environments():
        if env.name.startswith("trace_"):
            continue
        raw = effective_commutativity(env)
        valid = effective_commutativity(env, valid_only=True)
        sort_value = valid if valid == valid else raw
        rows.append({
            "name": env.name,
            "family": env.metadata.family,
            "raw": raw,
            "valid": valid,
            "sort": sort_value,
        })
    return sorted(rows, key=lambda row: (float(row["sort"]), float(row["raw"]), str(row["name"])))


def best_by_key(rows: list[dict[str, str]], key_fields: tuple[str, ...]) -> dict[tuple[str, ...], tuple[float, str]]:
    out: dict[tuple[str, ...], tuple[float, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        value = f(row["action_accuracy_mean"])
        if key not in out or value > out[key][0]:
            out[key] = (value, row["model"])
    return out


def observation_overview() -> None:
    summary = read_csv("rnn_navigation_summary.csv")
    advantage = read_csv("rnn_memory_advantage.csv")
    best = best_by_key(summary, ("obs_mode", "temporal_mode", "task"))
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (obs, temporal, _task), (value, _model) in best.items():
        acc[(obs, temporal)].append(value)
    adv: dict[tuple[str, str], list[float]] = defaultdict(list)
    pos: Counter[tuple[str, str]] = Counter()
    total: Counter[tuple[str, str]] = Counter()
    for row in advantage:
        key = (row["obs_mode"], row["temporal_mode"])
        value = f(row["memory_advantage"])
        adv[key].append(value)
        pos[key] += value > 0.02
        total[key] += 1

    width, height = 980, 555
    left, top = 185, 105
    cell_w, cell_h = 235, 62
    body = [
        '<text class="title" x="48" y="40">Observation Overview</text>',
        '<text class="subtitle" x="48" y="66">Cell: best-model action accuracy / memory advantage; positive count compares memory models to memoryless baselines.</text>',
    ]
    for col, temporal in enumerate(TEMPORAL_ORDER):
        x = left + col * cell_w + cell_w / 2
        body.append(f'<text class="label" text-anchor="middle" x="{x:.1f}" y="{top - 24}">{escape(TEMPORAL_LABELS[temporal])}</text>')
    for row_idx, obs in enumerate(OBS_ORDER):
        y = top + row_idx * cell_h
        body.append(f'<text class="label" text-anchor="end" x="{left - 14}" y="{y + cell_h / 2 + 5:.1f}">{escape(OBS_LABELS[obs])}</text>')
        for col, temporal in enumerate(TEMPORAL_ORDER):
            x = left + col * cell_w
            key = (obs, temporal)
            acc_mean = mean(acc[key])
            adv_mean = mean(adv[key])
            fill = interp((248, 250, 252), (22, 101, 52), (adv_mean + 0.02) / 0.20 if adv_mean == adv_mean else 0.0)
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            body.append(f'<text class="value" text-anchor="middle" x="{x + cell_w / 2:.1f}" y="{y + 23:.1f}">acc {fmt(acc_mean, 3)} / adv {fmt(adv_mean, 3)}</text>')
            body.append(f'<text class="small" text-anchor="middle" x="{x + cell_w / 2:.1f}" y="{y + 43:.1f}">{pos[key]}/{total[key]} tasks positive</text>')
    body.append('<text class="small" x="48" y="500">Darker green means a larger memory advantage. Gaussian sensory is the main realistic noisy channel.</text>')
    (FIGURES / "summary_observation_overview.svg").write_text(svg_doc(width, height, body), encoding="utf-8")


def gaussian_commutativity_order() -> None:
    summary = read_csv("rnn_navigation_summary.csv")
    envs = ordered_environments()
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in summary:
        if row["obs_mode"] != "gaussian_sensory" or row["temporal_mode"] != "temporal_full":
            continue
        fam = model_family(row["model"])
        if fam in MODEL_FAMILY_ORDER:
            values[(row["task"], fam)].append(f(row["action_accuracy_mean"]))
    best_values = {key: max(vals) for key, vals in values.items()}

    width, height = 1040, 760
    left, top = 285, 95
    cell_w, cell_h = 135, 39
    body = [
        '<text class="title" x="48" y="40">Gaussian Performances Ordered By Action Commutativity</text>',
        '<text class="subtitle" x="48" y="66">Each cell is best action accuracy within a model family; rows are sorted by valid-only commutativity.</text>',
    ]
    for col, fam in enumerate(MODEL_FAMILY_ORDER):
        x = left + col * cell_w + cell_w / 2
        body.append(f'<text class="small" text-anchor="middle" x="{x:.1f}" y="{top - 24}">{escape(FAMILY_LABELS[fam])}</text>')
    for idx, env in enumerate(envs):
        task = str(env["name"])
        y = top + idx * cell_h
        valid = float(env["valid"])
        valid_label = "raw" if valid != valid else "valid"
        comm = float(env["raw"]) if valid != valid else valid
        body.append(f'<text class="label" text-anchor="end" x="{left - 18}" y="{y + 18}">{escape(short_task(task))}</text>')
        body.append(f'<text class="small" text-anchor="end" x="{left - 18}" y="{y + 34}">{valid_label} C={fmt(comm, 3)}</text>')
        for col, fam in enumerate(MODEL_FAMILY_ORDER):
            x = left + col * cell_w
            value = best_values.get((task, fam), float("nan"))
            fill = "#f8fafc" if value != value else interp((254, 242, 242), (22, 101, 52), value)
            text_color = "#ffffff" if value == value and value > 0.70 else "#1f2933"
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            body.append(f'<text class="value" text-anchor="middle" x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" fill="{text_color}">{fmt(value, 2)}</text>')
    body.append('<text class="small" x="48" y="718">This replaces the earlier memoryless-versus-best-memory bar view; Lin blocks are included as an independent family.</text>')
    (FIGURES / "summary_gaussian_commutativity_order.svg").write_text(svg_doc(width, height, body), encoding="utf-8")


def gaussian_model_family_heatmap() -> None:
    summary = read_csv("rnn_navigation_summary.csv")
    envs = ordered_environments()
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in summary:
        if row["obs_mode"] != "gaussian_sensory" or row["temporal_mode"] != "temporal_full":
            continue
        fam = model_family(row["model"])
        if fam in MODEL_FAMILY_ORDER:
            values[(row["task"], fam)].append(f(row["action_accuracy_mean"]))
    best_values = {key: max(vals) for key, vals in values.items()}

    width, height = 1000, 760
    left, top = 260, 95
    cell_w, cell_h = 125, 39
    body = [
        '<text class="title" x="48" y="40">Gaussian Model-Family Comparison</text>',
        '<text class="subtitle" x="48" y="66">Best action accuracy within each family; environments use the same commutativity order.</text>',
    ]
    for col, fam in enumerate(MODEL_FAMILY_ORDER):
        x = left + col * cell_w + cell_w / 2
        body.append(f'<text class="small" text-anchor="middle" x="{x:.1f}" y="{top - 24}">{escape(FAMILY_LABELS[fam])}</text>')
    for row_idx, env in enumerate(envs):
        task = str(env["name"])
        y = top + row_idx * cell_h
        body.append(f'<text class="label" text-anchor="end" x="{left - 12}" y="{y + cell_h / 2 + 5:.1f}">{escape(short_task(task))}</text>')
        for col, fam in enumerate(MODEL_FAMILY_ORDER):
            x = left + col * cell_w
            value = best_values.get((task, fam), float("nan"))
            fill = "#f8fafc" if value != value else interp((254, 242, 242), (22, 101, 52), value)
            text_color = "#ffffff" if value == value and value > 0.70 else "#1f2933"
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            body.append(f'<text class="value" text-anchor="middle" x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" fill="{text_color}">{fmt(value, 2)}</text>')
    (FIGURES / "summary_gaussian_model_families.svg").write_text(svg_doc(width, height, body), encoding="utf-8")


def gaussian_action_input_bars() -> None:
    rows = read_csv("rnn_gaussian_updated_summary.csv")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["temporal_mode"], row["action_input"])].append(f(row["action_accuracy_mean"]))

    width, height = 940, 500
    left, top = 90, 105
    plot_w, plot_h = 770, 280
    group_w = plot_w / len(TEMPORAL_ORDER)
    bar_w = group_w / 5
    colors = {"none": "#94a3b8", "one_hot": "#2563eb", "embedding": "#15803d"}
    body = [
        '<text class="title" x="48" y="40">Previous-Action Input Under Gaussian Noise</text>',
        '<text class="subtitle" x="48" y="66">Mean action accuracy across Gaussian tasks and model families.</text>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
    ]
    for tick in (0.5, 0.6, 0.7):
        y = top + plot_h - tick * plot_h
        body.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        body.append(f'<text class="small" text-anchor="end" x="{left - 12}" y="{y + 4:.1f}">{tick:.1f}</text>')
    for group_idx, temporal in enumerate(TEMPORAL_ORDER):
        gx = left + group_idx * group_w
        for idx, action_input in enumerate(ACTION_INPUT_ORDER):
            value = mean(grouped[(temporal, action_input)])
            x = gx + group_w * 0.22 + idx * bar_w
            h = value * plot_h
            y = top + plot_h - h
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.78:.1f}" height="{h:.1f}" fill="{colors[action_input]}" rx="2"/>')
            body.append(f'<text class="value" text-anchor="middle" x="{x + bar_w * 0.39:.1f}" y="{y - 6:.1f}">{fmt(value, 3)}</text>')
        body.append(f'<text class="label" text-anchor="middle" x="{gx + group_w / 2:.1f}" y="{top + plot_h + 34:.1f}">{escape(TEMPORAL_LABELS[temporal])}</text>')
    legend_x = 605
    for idx, action_input in enumerate(ACTION_INPUT_ORDER):
        x = legend_x + idx * 105
        body.append(f'<rect x="{x}" y="34" width="15" height="15" fill="{colors[action_input]}" rx="2"/>')
        body.append(f'<text class="small" x="{x + 22}" y="47">{escape(ACTION_LABELS[action_input])}</text>')
    body.append('<text class="small" x="48" y="468">Action input is outside the environment; the environment still emits only current observations.</text>')
    (FIGURES / "summary_gaussian_action_input.svg").write_text(svg_doc(width, height, body), encoding="utf-8")


def long_delay_architecture_heatmap() -> None:
    rows = read_csv("rnn_lin_long_architecture_families.csv")
    regimes = (
        ("sparse_aliased", "temporal_full"),
        ("sparse_aliased", "temporal_cue_then_blank"),
        ("egocentric_landmark", "temporal_full"),
        ("egocentric_landmark", "temporal_cue_then_blank"),
    )
    values = {
        (row["architecture_family"], row["obs_mode"], row["temporal_mode"]): f(row["best_action_accuracy_mean"])
        for row in rows
    }

    width, height = 980, 675
    left, top = 250, 135
    cell_w, cell_h = 160, 54
    body = [
        '<text class="title" x="48" y="40">Long Delay: Lin Blocks Versus Reservoirs</text>',
        '<text class="subtitle" x="48" y="66">Best action accuracy by architecture family on the 72-step delayed T-maze.</text>',
    ]
    for col, (obs, temporal) in enumerate(regimes):
        x = left + col * cell_w + cell_w / 2
        label = f"{OBS_LABELS[obs]} / {TEMPORAL_LABELS[temporal]}"
        body.append(f'<text class="small" text-anchor="end" x="{x + 48:.1f}" y="{top - 28}" transform="rotate(-28 {x + 48:.1f} {top - 28})">{escape(label)}</text>')
    for row_idx, arch in enumerate(LONG_ARCH_ORDER):
        y = top + row_idx * cell_h
        body.append(f'<text class="label" text-anchor="end" x="{left - 14}" y="{y + cell_h / 2 + 5:.1f}">{escape(ARCH_LABELS[arch])}</text>')
        for col, (obs, temporal) in enumerate(regimes):
            x = left + col * cell_w
            value = values.get((arch, obs, temporal), float("nan"))
            fill = "#f8fafc" if value != value else interp((254, 242, 242), (22, 101, 52), value)
            text_color = "#ffffff" if value == value and value > 0.70 else "#1f2933"
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            body.append(f'<text class="value" text-anchor="middle" x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" fill="{text_color}">{fmt(value, 3)}</text>')
    body.append('<text class="small" x="48" y="618">Independent Lin blocks beat reservoirs under cue gaps; explicit history remains the ceiling control.</text>')
    (FIGURES / "summary_long_delay_architecture.svg").write_text(svg_doc(width, height, body), encoding="utf-8")


def main() -> None:
    observation_overview()
    gaussian_commutativity_order()
    gaussian_model_family_heatmap()
    gaussian_action_input_bars()
    long_delay_architecture_heatmap()
    for name in (
        "summary_observation_overview.svg",
        "summary_gaussian_commutativity_order.svg",
        "summary_gaussian_model_families.svg",
        "summary_gaussian_action_input.svg",
        "summary_long_delay_architecture.svg",
    ):
        print((FIGURES / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
