"""Factorial formal sweep for GeoMem trace geometry and memory sufficiency."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from random import Random
from typing import Hashable

from commutativity_probe import Environment, effective_commutativity, trace_monoid_general
from plot_current_results import esc


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DEPTHS = (4, 6)
ACTION_COUNTS = (3, 4)
HISTORY_LENGTHS = (1, 2, 4, 8)
OBSERVATION_MODES = ("full", "counts", "noisy_counts", "depth_last", "depth")
THRESHOLD = 0.95

State = tuple[int, ...]
Observation = Hashable


@dataclass(frozen=True)
class TraceSpec:
    action_count: int
    depth: int
    relation_count: int
    relation_id: int
    commuting_pairs: frozenset[tuple[str, str]]

    @property
    def relation_density(self) -> float:
        total = self.action_count * (self.action_count - 1) // 2
        return self.relation_count / total if total else 0.0


@dataclass(frozen=True)
class Candidate:
    name: str
    architecture: str
    history_len: int
    cost: int


def action_names(action_count: int) -> tuple[str, ...]:
    return tuple(chr(ord("A") + idx) for idx in range(action_count))


def relation_specs(action_count: int, depth: int) -> list[TraceSpec]:
    actions = action_names(action_count)
    all_pairs = tuple(combinations(actions, 2))
    relation_counts = (0, len(all_pairs) // 3, math.ceil(2 * len(all_pairs) / 3), len(all_pairs))
    specs: list[TraceSpec] = []
    for relation_count in sorted(set(relation_counts)):
        relations = list(combinations(all_pairs, relation_count))
        if len(relations) > 3:
            indices = sorted({0, len(relations) // 2, len(relations) - 1})
            relations = [relations[idx] for idx in indices]
        for relation_id, relation in enumerate(relations):
            specs.append(TraceSpec(action_count, depth, relation_count, relation_id, frozenset(relation)))
    return specs


def trace_environment(spec: TraceSpec) -> Environment:
    return trace_monoid_general(
        spec.action_count,
        spec.commuting_pairs,
        spec.depth,
        relation_id=spec.relation_id,
    )


def observe(state: State, mode: str, action_count: int) -> Observation:
    counts = tuple(state.count(action_idx) for action_idx in range(action_count))
    if mode == "full":
        return state
    if mode == "counts":
        return counts
    if mode == "noisy_counts":
        return tuple(count // 2 for count in counts)
    if mode == "depth_last":
        return (len(state), state[-1] if state else -1)
    if mode == "depth":
        return len(state)
    raise ValueError(f"unknown observation mode: {mode}")


def prefix_history(state: State, mode: str, history_len: int, action_count: int) -> tuple[Observation, ...]:
    prefixes = [state[:idx] for idx in range(len(state) + 1)]
    return tuple(observe(prefix, mode, action_count) for prefix in prefixes[-history_len:])


def observation_stats(env: Environment, action_count: int, mode: str) -> dict[str, float]:
    groups: dict[Observation, list[State]] = defaultdict(list)
    for state in env.states:
        groups[observe(state, mode, action_count)].append(state)
    weighted_sizes = [len(groups[observe(state, mode, action_count)]) for state in env.states]
    n_states = len(env.states)
    counts = Counter(observe(state, mode, action_count) for state in env.states)
    entropy = -sum((count / n_states) * math.log2(count / n_states) for count in counts.values())
    max_entropy = math.log2(n_states) if n_states > 1 else 0.0
    return {
        "mean_candidates": sum(weighted_sizes) / n_states,
        "normalized_observation_ambiguity": ((sum(weighted_sizes) / n_states) - 1) / max(1, n_states - 1),
        "retained_entropy_fraction": entropy / max_entropy if max_entropy else 1.0,
    }


def transition_stats(env: Environment, action_count: int, mode: str) -> dict[str, float]:
    outcomes: dict[tuple[Observation, str], set[State]] = defaultdict(set)
    weights: Counter[tuple[Observation, str]] = Counter()
    for state in env.states:
        obs = observe(state, mode, action_count)
        for action in env.actions:
            if env.is_valid(state, action):
                key = (obs, action)
                outcomes[key].add(env.step(state, action))
                weights[key] += 1
    total = sum(weights.values())
    ambiguous = [key for key, values in outcomes.items() if len(values) > 1]
    return {
        "mean_next_candidates": sum(len(outcomes[key]) * weights[key] for key in outcomes) / total if total else float("nan"),
        "weighted_transition_ambiguity": sum(weights[key] for key in ambiguous) / total if total else float("nan"),
    }


def navigation_graph(env: Environment) -> dict[State, list[tuple[str, State]]]:
    graph: dict[State, list[tuple[str, State]]] = defaultdict(list)
    for state in env.states:
        for action in env.actions:
            if env.is_valid(state, action):
                target = env.step(state, action)
                graph[state].append((action, target))
                graph[target].append((f"undo_{action}", state))
    return graph


def policy_sets(env: Environment, goal: State) -> dict[State, frozenset[str]]:
    graph = navigation_graph(env)
    distances = {goal: 0}
    queue: deque[State] = deque([goal])
    while queue:
        state = queue.popleft()
        for _, neighbor in graph[state]:
            if neighbor not in distances:
                distances[neighbor] = distances[state] + 1
                queue.append(neighbor)
    policies = {}
    for state, distance in distances.items():
        if distance:
            options = frozenset(
                action for action, neighbor in graph[state]
                if distances.get(neighbor) == distance - 1
            )
            if options:
                policies[state] = options
    return policies


def selected_goals(env: Environment, seed: int) -> list[State]:
    deepest = [state for state in env.states if len(state) == env.max_depth]
    rng = Random(seed)
    rng.shuffle(deepest)
    return sorted(deepest[: min(4, len(deepest))])


def policy_rows(env: Environment, goals: list[State]) -> list[dict[str, object]]:
    rows = []
    for goal_idx, goal in enumerate(goals):
        for state, policy in policy_sets(env, goal).items():
            rows.append({
                "goal_idx": goal_idx,
                "state": state,
                "target": sorted(policy)[0],
                "policy": policy,
            })
    return rows


def policy_ambiguity(rows: list[dict[str, object]], mode: str, action_count: int) -> float:
    by_key: dict[tuple[int, Observation], set[frozenset[str]]] = defaultdict(set)
    weights: Counter[tuple[int, Observation]] = Counter()
    for row in rows:
        state = row["state"]
        assert isinstance(state, tuple)
        key = (int(row["goal_idx"]), observe(state, mode, action_count))
        policy = row["policy"]
        assert isinstance(policy, frozenset)
        by_key[key].add(policy)
        weights[key] += 1
    total = sum(weights.values())
    return sum(weights[key] for key, values in by_key.items() if len(values) > 1) / total if total else float("nan")


def candidates(action_count: int) -> list[Candidate]:
    result = [Candidate("memoryless_counts", "vector_counts", 0, action_count)]
    for history_len in HISTORY_LENGTHS:
        result.append(Candidate(f"history_counts_h{history_len}", "sequence_counts_history", history_len, action_count * history_len))
        result.append(Candidate(f"hybrid_counts_depth_last_h{history_len}", "hybrid_vector_route_history", history_len, action_count + 2 * history_len))
    return result


def architecture_key(row: dict[str, object], candidate: Candidate, action_count: int) -> Hashable:
    state = row["state"]
    assert isinstance(state, tuple)
    goal_idx = int(row["goal_idx"])
    if candidate.architecture == "vector_counts":
        return (goal_idx, observe(state, "counts", action_count))
    if candidate.architecture == "sequence_counts_history":
        return (goal_idx, prefix_history(state, "counts", candidate.history_len, action_count))
    if candidate.architecture == "hybrid_vector_route_history":
        return (
            goal_idx,
            observe(state, "counts", action_count),
            prefix_history(state, "depth_last", candidate.history_len, action_count),
        )
    raise ValueError(candidate.architecture)


def lookup_accuracy(rows: list[dict[str, object]], candidate: Candidate, action_count: int) -> float:
    by_key: dict[Hashable, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_key[architecture_key(row, candidate, action_count)][str(row["target"])] += 1
    table = {key: sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] for key, counts in by_key.items()}
    return sum(
        int(table[architecture_key(row, candidate, action_count)] in row["policy"])
        for row in rows
    ) / len(rows)


def collect() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    architecture_rows: list[dict[str, object]] = []
    for action_count in ACTION_COUNTS:
        for depth in DEPTHS:
            for spec in relation_specs(action_count, depth):
                env = trace_environment(spec)
                commute = effective_commutativity(env, length=min(4, depth), n_sequences=160, seed=depth + spec.relation_id, valid_only=True)
                rows = policy_rows(env, selected_goals(env, seed=1000 + depth + spec.relation_id))
                counts_obs = observation_stats(env, action_count, "counts")
                counts_transition = transition_stats(env, action_count, "counts")
                metric_rows.append({
                    "environment": env.name,
                    "action_count": action_count,
                    "depth": depth,
                    "relation_count": spec.relation_count,
                    "relation_density": spec.relation_density,
                    "relation_id": spec.relation_id,
                    "n_states": len(env.states),
                    "n_goals": len(selected_goals(env, seed=1000 + depth + spec.relation_id)),
                    "effective_commutativity": commute,
                    **counts_obs,
                    **counts_transition,
                    "weighted_policy_ambiguity": policy_ambiguity(rows, "counts", action_count),
                })
                candidate_rows = []
                for candidate in candidates(action_count):
                    candidate_rows.append({
                        "environment": env.name,
                        "action_count": action_count,
                        "depth": depth,
                        "relation_density": spec.relation_density,
                        "relation_id": spec.relation_id,
                        "effective_commutativity": commute,
                        "model": candidate.name,
                        "architecture": candidate.architecture,
                        "history_len": candidate.history_len,
                        "memory_cost": candidate.cost,
                        "optimal_action_accuracy": lookup_accuracy(rows, candidate, action_count),
                    })
                sufficient = [row for row in candidate_rows if float(row["optimal_action_accuracy"]) >= THRESHOLD]
                selected = min(sufficient, key=lambda row: (int(row["memory_cost"]), -float(row["optimal_action_accuracy"]))) if sufficient else None
                for row in candidate_rows:
                    row["selected_at_095"] = bool(selected is row)
                    row["selected_memory_cost"] = int(selected["memory_cost"]) if selected is not None else -1
                    row["selected_model"] = str(selected["model"]) if selected is not None else "none"
                architecture_rows.extend(candidate_rows)
    return metric_rows, architecture_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(metric_rows: list[dict[str, object]], architecture_rows: list[dict[str, object]], path: Path) -> None:
    cost_by_env = {
        str(row["environment"]): int(row["selected_memory_cost"])
        for row in architecture_rows
        if bool(row["selected_at_095"])
    }
    panels = (
        ("weighted_transition_ambiguity", "Transition ambiguity", "#c2410c", 1.0),
        ("weighted_policy_ambiguity", "Policy ambiguity", "#2563eb", 1.0),
        ("selected_memory_cost", "Cheapest sufficient cost", "#15803d", max(cost_by_env.values(), default=1)),
    )
    width, height = 1120, 452
    margin_left, margin_right, margin_top, margin_bottom = 72, 35, 112, 58
    panel_w = (width - margin_left - margin_right) / len(panels)
    plot_h = height - margin_top - margin_bottom
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:31px;font-weight:700}.label{font-size:18px}.tick{font-size:16px;fill:#52606d}</style>',
        '<text class="title" x="56" y="36">Systematic trace geometry sweep</text>',
        '<text class="label" x="56" y="60">Counts observation; marker shape by action alphabet, repeated relation graphs and trace depths</text>',
    ]
    for panel_idx, (metric, title, color, max_value) in enumerate(panels):
        x0 = margin_left + panel_idx * panel_w
        parts.append(f'<text class="label" x="{x0 + 8:.1f}" y="{margin_top - 14}">{esc(title)}</text>')
        parts.append(f'<line x1="{x0}" y1="{margin_top + plot_h}" x2="{x0 + panel_w - 28}" y2="{margin_top + plot_h}" stroke="#9aa5b1"/>')
        parts.append(f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{margin_top + plot_h}" stroke="#9aa5b1"/>')
        for tick in (0.0, 0.5, 1.0):
            x = x0 + tick * (panel_w - 28)
            parts.append(f'<text class="tick" text-anchor="middle" x="{x:.1f}" y="{height - 28}">{tick:.1f}</text>')
        for row in metric_rows:
            env = str(row["environment"])
            value = cost_by_env.get(env, -1) if metric == "selected_memory_cost" else float(row[metric])
            if value < 0:
                continue
            x = x0 + float(row["effective_commutativity"]) * (panel_w - 28)
            y = margin_top + plot_h * (1 - min(1.0, float(value) / max(1e-8, max_value)))
            radius = 4 if int(row["action_count"]) == 3 else 6
            opacity = 0.55 if int(row["depth"]) == 4 else 0.95
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" fill-opacity="{opacity}" stroke="#ffffff"/>')
    parts.append(f'<text class="label" text-anchor="middle" x="{width / 2:.1f}" y="{height - 7}">effective commutativity</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metric_rows, architecture_rows = collect()
    write_csv(FIGURES / "systematic_formal_sweep.csv", metric_rows)
    write_csv(FIGURES / "systematic_formal_architectures.csv", architecture_rows)
    plot(metric_rows, architecture_rows, FIGURES / "systematic_formal_sweep.svg")
    print("Generated:")
    print((FIGURES / "systematic_formal_sweep.csv").relative_to(ROOT))
    print((FIGURES / "systematic_formal_architectures.csv").relative_to(ROOT))
    print((FIGURES / "systematic_formal_sweep.svg").relative_to(ROOT))


if __name__ == "__main__":
    main()
