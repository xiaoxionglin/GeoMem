"""Architecture-selection analysis for the lightweight GeoMem benchmark.

The lightweight benchmark asks which memory representation predicts the
shortest-path policy. This script adds the missing architectural question:
which is the cheapest sufficient memory architecture for each task geometry?
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from plot_current_results import esc


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
SOURCE = FIGURES / "lightweight_empirical_benchmark.csv"

ENV_NAMES = ("trace_commute_000", "trace_commute_033", "trace_commute_067", "trace_commute_100")
THRESHOLDS = (0.95, 0.99)
ARCHITECTURE_PRIORITY = {
    "vector_counts": 0,
    "hybrid_vector_route_history": 1,
    "sequence_history": 2,
}


@dataclass(frozen=True)
class Candidate:
    model: str
    architecture: str
    cost: int
    candidate_set: str


def candidates() -> list[Candidate]:
    base = [Candidate("memoryless_counts", "vector_counts", 3, "conservative_counts")]
    for history_len in (1, 2, 4, 8):
        base.append(Candidate(
            f"history_counts_h{history_len}",
            "sequence_history",
            3 * history_len,
            "conservative_counts",
        ))
    hybrid = [
        Candidate(candidate.model, candidate.architecture, candidate.cost, "hybrid_allowed")
        for candidate in base
    ]
    for history_len in (1, 2, 4, 8):
        hybrid.append(Candidate(
            f"hybrid_counts_depth_last_h{history_len}",
            "hybrid_vector_route_history",
            3 + 2 * history_len,
            "hybrid_allowed",
        ))
    return base + hybrid


def load_benchmark() -> list[dict[str, str]]:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"{SOURCE.relative_to(ROOT)} not found; run experiments/lightweight_empirical_benchmark.py first"
        )
    with SOURCE.open(newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_accuracy(rows: list[dict[str, str]]) -> dict[tuple[str, str], tuple[float, float, int]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["split"] == "all" and row["estimator"] == "lookup":
            grouped[(row["environment"], row["model"])].append(float(row["optimal_action_accuracy"]))
    return {
        key: (mean(values), pstdev(values) if len(values) > 1 else 0.0, len(values))
        for key, values in grouped.items()
    }


def select_architectures(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    accuracies = aggregate_accuracy(rows)
    all_candidates = candidates()
    result_rows: list[dict[str, object]] = []
    for env in ENV_NAMES:
        for candidate_set in ("conservative_counts", "hybrid_allowed"):
            available = [
                candidate for candidate in all_candidates
                if candidate.candidate_set == candidate_set
            ]
            for threshold in THRESHOLDS:
                sufficient = []
                for candidate in available:
                    acc, sd, n_goals = accuracies[(env, candidate.model)]
                    if acc >= threshold:
                        sufficient.append((
                            candidate.cost,
                            ARCHITECTURE_PRIORITY[candidate.architecture],
                            -acc,
                            candidate.model,
                            candidate,
                            acc,
                            sd,
                            n_goals,
                        ))
                if sufficient:
                    cost, priority, negative_acc, model_name, candidate, acc, sd, n_goals = sorted(sufficient)[0]
                    selected_model = candidate.model
                    architecture = candidate.architecture
                    selected_cost = cost
                    selected_acc = acc
                    selected_sd = sd
                else:
                    selected_model = "none"
                    architecture = "none"
                    selected_cost = -1
                    selected_acc = float("nan")
                    selected_sd = float("nan")
                    n_goals = 0
                vector_acc, vector_sd, _ = accuracies[(env, "memoryless_counts")]
                result_rows.append({
                    "environment": env,
                    "candidate_set": candidate_set,
                    "threshold": threshold,
                    "selected_model": selected_model,
                    "selected_architecture": architecture,
                    "memory_cost": selected_cost,
                    "mean_accuracy": selected_acc,
                    "sd_accuracy": selected_sd,
                    "n_goals": n_goals,
                    "vector_accuracy": vector_acc,
                    "vector_sd": vector_sd,
                    "gain_over_vector": selected_acc - vector_acc if selected_acc == selected_acc else float("nan"),
                })
    return result_rows


def write_csv(rows: list[dict[str, object]]) -> None:
    path = FIGURES / "architecture_selection.csv"
    fields = [
        "environment",
        "candidate_set",
        "threshold",
        "selected_model",
        "selected_architecture",
        "memory_cost",
        "mean_accuracy",
        "sd_accuracy",
        "n_goals",
        "vector_accuracy",
        "vector_sd",
        "gain_over_vector",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def architecture_selection_svg(rows: list[dict[str, object]], path: Path) -> None:
    selected = [
        row for row in rows
        if row["threshold"] == 0.95
    ]
    width, height = 930, 500
    margin_left, margin_right, margin_top, margin_bottom = 120, 60, 76, 152
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_cost = 14
    groups = ("conservative_counts", "hybrid_allowed")
    colors = {
        "conservative_counts": "#2563eb",
        "hybrid_allowed": "#15803d",
    }

    def x_pos(env_idx: int, group_idx: int) -> float:
        group_w = plot_w / len(ENV_NAMES)
        return margin_left + env_idx * group_w + group_w * (0.37 + 0.26 * group_idx)

    def y_pos(cost: int) -> float:
        return margin_top + plot_h * (1 - cost / max_cost)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:19px}.tick{font-size:17px;fill:#52606d}.barlabel{font-size:16px;fill:#1f2933}</style>',
        '<text class="title" x="70" y="38">Cheapest sufficient memory architecture</text>',
        '<text class="label" x="70" y="61">Minimum architecture cost reaching 0.95 multi-goal policy accuracy</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#9aa5b1"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#9aa5b1"/>',
    ]
    for tick in (0, 3, 6, 9, 12):
        y = y_pos(tick)
        parts.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" stroke="#e4e7eb"/>')
        parts.append(f'<text class="tick" text-anchor="end" x="{margin_left - 10}" y="{y + 4:.1f}">{tick}</text>')
    for env_idx, env in enumerate(ENV_NAMES):
        for group_idx, group in enumerate(groups):
            row = next(item for item in selected if item["environment"] == env and item["candidate_set"] == group)
            cost = int(row["memory_cost"])
            x = x_pos(env_idx, group_idx)
            y = y_pos(cost)
            bar_w = 34
            parts.append(f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w}" height="{margin_top + plot_h - y:.1f}" fill="{colors[group]}"/>')
            parts.append(f'<text class="barlabel" text-anchor="middle" x="{x:.1f}" y="{y - 6:.1f}">{cost}</text>')
        label_x = margin_left + (env_idx + 0.5) * plot_w / len(ENV_NAMES)
        parts.append(f'<text class="tick" text-anchor="end" x="{label_x + 30:.1f}" y="{height - 96}" transform="rotate(-28 {label_x + 30:.1f} {height - 96})">{esc(env.replace("_", " "))}</text>')
    legend_x = width - 260
    for idx, group in enumerate(groups):
        y = margin_top + idx * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 12}" width="16" height="16" fill="{colors[group]}"/>')
        parts.append(f'<text class="tick" x="{legend_x + 24}" y="{y + 1}">{esc(group.replace("_", " "))}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary(rows: list[dict[str, object]]) -> None:
    path = FIGURES / "architecture_selection_summary.md"
    lines = [
        "# Architecture Selection Summary",
        "",
        "Threshold is mean multi-goal optimal-action accuracy under the all-state lookup diagnostic.",
        "",
        "| Environment | Candidate set | Threshold | Selected model | Cost | Accuracy | Gain over vector |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['environment']} | {row['candidate_set']} | {float(row['threshold']):.2f} | "
            f"{row['selected_model']} | {int(row['memory_cost'])} | "
            f"{float(row['mean_accuracy']):.3f} | {float(row['gain_over_vector']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = select_architectures(load_benchmark())
    write_csv(rows)
    write_summary(rows)
    architecture_selection_svg(rows, FIGURES / "architecture_selection.svg")
    print("Generated:")
    print((FIGURES / "architecture_selection.csv").relative_to(ROOT))
    print((FIGURES / "architecture_selection_summary.md").relative_to(ROOT))
    print((FIGURES / "architecture_selection.svg").relative_to(ROOT))
    print()
    print("0.95 threshold selections:")
    for row in rows:
        if row["threshold"] == 0.95:
            print(
                f"{row['environment']},{row['candidate_set']},"
                f"{row['selected_model']},cost={row['memory_cost']},acc={float(row['mean_accuracy']):.3f}"
            )


if __name__ == "__main__":
    main()
