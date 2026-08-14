"""Goal-conditioned policy ambiguity for trace-monoid environments."""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path
from typing import Hashable

from commutativity_probe import Environment, trace_monoid_family
from observation_ambiguity import observe
from plot_current_results import esc


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

Observation = Hashable
Policy = frozenset[str]


def navigation_graph(env: Environment) -> dict[tuple[int, ...], list[tuple[str, tuple[int, ...]]]]:
    """Treat valid trace transitions as reversible navigation edges."""
    graph: dict[tuple[int, ...], list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    for state in env.states:
        for action in env.actions:
            if not env.is_valid(state, action):
                continue
            next_state = env.step(state, action)
            graph[state].append((action, next_state))
            graph[next_state].append((f"undo_{action}", state))
    return graph


def goal_state(env: Environment) -> tuple[int, ...]:
    target = ("A", "B", "C", "A", "B", "C", "A", "B")
    state, valid = env.rollout((), target)
    if not valid:
        raise RuntimeError(f"goal sequence is invalid for {env.name}")
    return state


def distances_to_goal(env: Environment, goal: tuple[int, ...]) -> dict[tuple[int, ...], int]:
    graph = navigation_graph(env)
    distances = {goal: 0}
    queue: deque[tuple[int, ...]] = deque([goal])
    while queue:
        state = queue.popleft()
        for _, neighbor in graph[state]:
            if neighbor not in distances:
                distances[neighbor] = distances[state] + 1
                queue.append(neighbor)
    return distances


def optimal_policy_sets(env: Environment, goal: tuple[int, ...]) -> dict[tuple[int, ...], Policy]:
    graph = navigation_graph(env)
    distances = distances_to_goal(env, goal)
    policies: dict[tuple[int, ...], Policy] = {}
    for state, distance in distances.items():
        if distance == 0:
            continue
        optimal = frozenset(
            action for action, neighbor in graph[state]
            if neighbor in distances and distances[neighbor] == distance - 1
        )
        if optimal:
            policies[state] = optimal
    return policies


def policy_ambiguity_stats(env: Environment, mode: str) -> dict[str, float]:
    goal = goal_state(env)
    policies = optimal_policy_sets(env, goal)
    by_obs: dict[Observation, list[Policy]] = defaultdict(list)
    for state, policy in policies.items():
        by_obs[observe(state, mode)].append(policy)

    if not policies:
        return {
            "goal_depth": float(len(goal)),
            "n_policy_states": 0.0,
            "policy_ambiguous_observation_fraction": float("nan"),
            "weighted_policy_ambiguity": float("nan"),
            "mean_policy_set_size": float("nan"),
        }

    ambiguous_obs = {
        obs for obs, policy_list in by_obs.items()
        if len(set(policy_list)) > 1
    }
    weighted_ambiguous = sum(
        len(policy_list) for obs, policy_list in by_obs.items()
        if obs in ambiguous_obs
    ) / len(policies)
    ambiguous_fraction = len(ambiguous_obs) / len(by_obs)
    mean_policy_size = sum(len(policy) for policy in policies.values()) / len(policies)
    return {
        "goal_depth": float(len(goal)),
        "n_policy_states": float(len(policies)),
        "policy_ambiguous_observation_fraction": ambiguous_fraction,
        "weighted_policy_ambiguity": weighted_ambiguous,
        "mean_policy_set_size": mean_policy_size,
    }


def collect() -> tuple[list[Environment], list[str], dict[str, dict[str, dict[str, float]]]]:
    envs = trace_monoid_family(max_depth=8)
    modes = ["full", "counts", "depth_last", "depth"]
    results: dict[str, dict[str, dict[str, float]]] = {}
    for env in envs:
        results[env.name] = {}
        for mode in modes:
            results[env.name][mode] = policy_ambiguity_stats(env, mode)
    return envs, modes, results


def write_csv(envs: list[Environment], modes: list[str], results: dict[str, dict[str, dict[str, float]]]) -> None:
    path = FIGURES / "policy_ambiguity.csv"
    fields = [
        "environment",
        "observation_mode",
        "goal_depth",
        "n_policy_states",
        "policy_ambiguous_observation_fraction",
        "weighted_policy_ambiguity",
        "mean_policy_set_size",
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
        low = (239, 246, 255)
        high = (29, 78, 216)
        r = round(low[0] + (high[0] - low[0]) * value)
        g = round(low[1] + (high[1] - low[1]) * value)
        b = round(low[2] + (high[2] - low[2]) * value)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:19px}.tick{font-size:17px;fill:#52606d}.cell{font-size:18px;font-weight:700}</style>',
        '<text class="title" x="80" y="38">Goal-conditioned policy ambiguity</text>',
        '<text class="label" x="80" y="60">Cell = fraction of states whose observation aliases different shortest-path policies</text>',
    ]
    for row, env in enumerate(envs):
        y0 = margin_top + row * cell_h
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 14}" y="{y0 + cell_h / 2 + 5:.1f}">{esc(env.name.replace("_", " "))}</text>')
        for col, mode in enumerate(modes):
            value = results[env.name][mode]["weighted_policy_ambiguity"]
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
    path = FIGURES / "policy_ambiguity_heatmap.svg"
    path.write_text(heatmap(envs, modes, results), encoding="utf-8")
    print("Generated:")
    print(path.relative_to(ROOT))
    print((FIGURES / "policy_ambiguity.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()

