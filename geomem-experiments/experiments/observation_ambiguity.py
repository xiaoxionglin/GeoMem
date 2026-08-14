"""Measure observation-induced state ambiguity for trace-monoid environments."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from math import log2
from pathlib import Path
from typing import Hashable

from commutativity_probe import Environment, trace_monoid_family
from plot_current_results import esc


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


Observation = Hashable


def action_counts(state: tuple[int, ...], n_actions: int = 3) -> tuple[int, ...]:
    counts = [0] * n_actions
    for action in state:
        counts[action] += 1
    return tuple(counts)


def observe(state: tuple[int, ...], mode: str) -> Observation:
    if mode == "full":
        return state
    if mode == "counts":
        return action_counts(state)
    if mode == "depth_last":
        return (len(state), state[-1] if state else -1)
    if mode == "depth":
        return len(state)
    raise ValueError(f"unknown observation mode: {mode}")


def ambiguity_stats(env: Environment, mode: str) -> dict[str, float]:
    groups: dict[Observation, list[tuple[int, ...]]] = defaultdict(list)
    for state in env.states:
        groups[observe(state, mode)].append(state)

    n_states = len(env.states)
    weighted_sizes = [len(groups[observe(state, mode)]) for state in env.states]
    mean_candidates = sum(weighted_sizes) / n_states
    max_candidates = max(weighted_sizes)
    normalized = 0.0 if n_states <= 1 else (mean_candidates - 1) / (n_states - 1)
    obs_counts = Counter(observe(state, mode) for state in env.states)
    entropy = -sum((count / n_states) * log2(count / n_states) for count in obs_counts.values())
    max_entropy = log2(n_states) if n_states > 1 else 0.0
    retained_fraction = entropy / max_entropy if max_entropy else 1.0
    return {
        "n_states": float(n_states),
        "n_observations": float(len(groups)),
        "mean_candidates": mean_candidates,
        "max_candidates": float(max_candidates),
        "normalized_ambiguity": normalized,
        "retained_entropy_fraction": retained_fraction,
    }


def collect() -> tuple[list[Environment], list[str], dict[str, dict[str, dict[str, float]]]]:
    envs = trace_monoid_family(max_depth=8)
    modes = ["full", "counts", "depth_last", "depth"]
    results: dict[str, dict[str, dict[str, float]]] = {}
    for env in envs:
        results[env.name] = {}
        for mode in modes:
            results[env.name][mode] = ambiguity_stats(env, mode)
    return envs, modes, results


def write_csv(envs: list[Environment], modes: list[str], results: dict[str, dict[str, dict[str, float]]]) -> None:
    path = FIGURES / "observation_ambiguity.csv"
    fields = [
        "environment",
        "observation_mode",
        "n_states",
        "n_observations",
        "mean_candidates",
        "max_candidates",
        "normalized_ambiguity",
        "retained_entropy_fraction",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for env in envs:
            for mode in modes:
                row = {"environment": env.name, "observation_mode": mode}
                row.update({key: f"{value:.6f}" for key, value in results[env.name][mode].items()})
                writer.writerow(row)


def heatmap(
    envs: list[Environment],
    modes: list[str],
    results: dict[str, dict[str, dict[str, float]]],
    *,
    metric: str,
    title: str,
    subtitle: str,
) -> str:
    width, height = 820, 430
    margin_left, margin_right, margin_top, margin_bottom = 180, 50, 80, 70
    cell_w = (width - margin_left - margin_right) / len(modes)
    cell_h = (height - margin_top - margin_bottom) / len(envs)

    def color(value: float) -> str:
        low = (240, 249, 244)
        high = (21, 128, 61)
        r = round(low[0] + (high[0] - low[0]) * value)
        g = round(low[1] + (high[1] - low[1]) * value)
        b = round(low[2] + (high[2] - low[2]) * value)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:19px}.tick{font-size:17px;fill:#52606d}.cell{font-size:18px;font-weight:700}</style>',
        f'<text class="title" x="80" y="38">{esc(title)}</text>',
        f'<text class="label" x="80" y="60">{esc(subtitle)}</text>',
    ]
    for row, env in enumerate(envs):
        y0 = margin_top + row * cell_h
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 14}" y="{y0 + cell_h / 2 + 5:.1f}">{esc(env.name.replace("_", " "))}</text>')
        for col, mode in enumerate(modes):
            value = results[env.name][mode][metric]
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
    ambiguity_path = FIGURES / "observation_ambiguity_heatmap.svg"
    ambiguity_path.write_text(
        heatmap(
            envs,
            modes,
            results,
            metric="normalized_ambiguity",
            title="Observation ambiguity in trace environments",
            subtitle="Cell = normalized candidate-set size; higher means more latent states per observation",
        ),
        encoding="utf-8",
    )
    information_path = FIGURES / "observation_information_heatmap.svg"
    information_path.write_text(
        heatmap(
            envs,
            modes,
            results,
            metric="retained_entropy_fraction",
            title="Observation information retained",
            subtitle="Cell = observation entropy / latent-state entropy; higher means less aliasing",
        ),
        encoding="utf-8",
    )
    print("Generated:")
    print(ambiguity_path.relative_to(ROOT))
    print(information_path.relative_to(ROOT))
    print((FIGURES / "observation_ambiguity.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()
