"""Measure whether observations are Markov under each trace environment."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Hashable

from commutativity_probe import Environment, trace_monoid_family
from observation_ambiguity import observe
from plot_current_results import esc


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


Observation = Hashable


def transition_stats(env: Environment, mode: str) -> dict[str, float]:
    outcomes: dict[tuple[Observation, str], set[tuple[int, ...]]] = defaultdict(set)
    weights: dict[tuple[Observation, str], int] = defaultdict(int)

    for state in env.states:
        obs = observe(state, mode)
        for action in env.actions:
            if not env.is_valid(state, action):
                continue
            next_state = env.step(state, action)
            key = (obs, action)
            outcomes[key].add(next_state)
            weights[key] += 1

    if not outcomes:
        return {
            "mean_next_candidates": float("nan"),
            "max_next_candidates": float("nan"),
            "ambiguous_fraction": float("nan"),
            "weighted_ambiguous_fraction": float("nan"),
        }

    total_weight = sum(weights.values())
    mean_next = sum(len(outcomes[key]) * weights[key] for key in outcomes) / total_weight
    max_next = max(len(values) for values in outcomes.values())
    ambiguous_keys = [key for key, values in outcomes.items() if len(values) > 1]
    ambiguous_fraction = len(ambiguous_keys) / len(outcomes)
    weighted_ambiguous = sum(weights[key] for key in ambiguous_keys) / total_weight
    return {
        "mean_next_candidates": mean_next,
        "max_next_candidates": float(max_next),
        "ambiguous_fraction": ambiguous_fraction,
        "weighted_ambiguous_fraction": weighted_ambiguous,
    }


def collect() -> tuple[list[Environment], list[str], dict[str, dict[str, dict[str, float]]]]:
    envs = trace_monoid_family(max_depth=8)
    modes = ["full", "counts", "depth_last", "depth"]
    results: dict[str, dict[str, dict[str, float]]] = {}
    for env in envs:
        results[env.name] = {}
        for mode in modes:
            results[env.name][mode] = transition_stats(env, mode)
    return envs, modes, results


def write_csv(envs: list[Environment], modes: list[str], results: dict[str, dict[str, dict[str, float]]]) -> None:
    path = FIGURES / "transition_ambiguity.csv"
    fields = [
        "environment",
        "observation_mode",
        "mean_next_candidates",
        "max_next_candidates",
        "ambiguous_fraction",
        "weighted_ambiguous_fraction",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for env in envs:
            for mode in modes:
                row = {"environment": env.name, "observation_mode": mode}
                row.update({key: f"{value:.6f}" for key, value in results[env.name][mode].items()})
                writer.writerow(row)


def heatmap(envs: list[Environment], modes: list[str], results: dict[str, dict[str, dict[str, float]]]) -> str:
    width, height = 820, 430
    margin_left, margin_right, margin_top, margin_bottom = 180, 50, 80, 70
    cell_w = (width - margin_left - margin_right) / len(modes)
    cell_h = (height - margin_top - margin_bottom) / len(envs)

    def color(value: float) -> str:
        low = (255, 247, 237)
        high = (194, 65, 12)
        r = round(low[0] + (high[0] - low[0]) * value)
        g = round(low[1] + (high[1] - low[1]) * value)
        b = round(low[2] + (high[2] - low[2]) * value)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:19px}.tick{font-size:17px;fill:#52606d}.cell{font-size:18px;font-weight:700}</style>',
        '<text class="title" x="80" y="38">Observation transition ambiguity</text>',
        '<text class="label" x="80" y="60">Cell = weighted fraction of observation-action pairs with multiple latent next states</text>',
    ]
    for row, env in enumerate(envs):
        y0 = margin_top + row * cell_h
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 14}" y="{y0 + cell_h / 2 + 5:.1f}">{esc(env.name.replace("_", " "))}</text>')
        for col, mode in enumerate(modes):
            value = results[env.name][mode]["weighted_ambiguous_fraction"]
            x0 = margin_left + col * cell_w
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{color(value)}" stroke="#ffffff" stroke-width="2"/>')
            text_color = "#ffffff" if value > 0.55 else "#1f2933"
            parts.append(f'<text class="cell" text-anchor="middle" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 5:.1f}" fill="{text_color}">{value:.2f}</text>')
    for col, mode in enumerate(modes):
        x0 = margin_left + col * cell_w + cell_w / 2
        parts.append(f'<text class="tick" text-anchor="middle" x="{x0:.1f}" y="{height - 35}">{esc(mode)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    envs, modes, results = collect()
    write_csv(envs, modes, results)
    path = FIGURES / "transition_ambiguity_heatmap.svg"
    path.write_text(heatmap(envs, modes, results), encoding="utf-8")
    print("Generated:")
    print(path.relative_to(ROOT))
    print((FIGURES / "transition_ambiguity.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()

