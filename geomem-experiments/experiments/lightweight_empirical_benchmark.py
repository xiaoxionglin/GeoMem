"""Lightweight multi-goal empirical benchmark for GeoMem.

This benchmark stays deliberately small: it uses shortest-path policy labels,
table-lookup sufficiency diagnostics, and CPU decision-tree classifiers. It is
intended to test whether the existing trace-environment diagnostics survive
goal variation, sparse/degraded observations, and finite memory limits.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean, pstdev
from typing import Hashable

try:
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    DecisionTreeClassifier = None

    def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
        return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)

from commutativity_probe import Environment, trace_monoid_family
from observation_ambiguity import action_counts, observe
from policy_ambiguity import navigation_graph, optimal_policy_sets
from plot_current_results import esc


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

State = tuple[int, ...]
Observation = Hashable
Key = Hashable
ActionLabel = str


OBSERVATION_MODES = ("counts", "noisy_counts", "depth_last", "depth")
HISTORY_LENGTHS = (1, 2, 4, 8)
ENV_NAMES = ("trace_commute_000", "trace_commute_033", "trace_commute_067", "trace_commute_100")
LEARNED_ESTIMATOR = "decision_tree" if DecisionTreeClassifier is not None else "feature_lookup"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    architecture: str
    observation_mode: str
    history_len: int


def canonical_label(policy: frozenset[str]) -> str:
    """Use a deterministic single target when several actions are optimal."""
    return sorted(policy)[0]


def observe_light(state: State, mode: str) -> Observation:
    """Observation modes used by the lightweight benchmark."""
    if mode == "noisy_counts":
        return tuple(count // 2 for count in action_counts(state))
    return observe(state, mode)


def prefix_history(state: State, mode: str, history_len: int) -> tuple[Observation, ...]:
    """Observation history along the canonical path from root to state."""
    if history_len <= 0:
        return tuple()
    prefixes = [state[:idx] for idx in range(len(state) + 1)]
    observations = [observe_light(prefix, mode) for prefix in prefixes]
    return tuple(observations[-history_len:])


def selected_goals(env: Environment, *, n_sampled: int = 8, seed: int = 23) -> list[State]:
    """Fixed goal plus deterministic sampled goals from depths 5-8."""
    fixed, valid = env.rollout((), ("A", "B", "C", "A", "B", "C", "A", "B"))
    if not valid:
        raise RuntimeError(f"fixed goal invalid for {env.name}")

    candidates = [state for state in env.states if 5 <= len(state) <= 8 and state != fixed]
    candidates = sorted(candidates)
    rng = Random(seed)
    rng.shuffle(candidates)
    sampled = sorted(candidates[:n_sampled])
    return [fixed, *sampled]


def goal_id(goal: State) -> str:
    return "".join("ABC"[idx] for idx in goal) or "root"


def example_rows(env: Environment, goal: State) -> list[dict[str, object]]:
    policies = optimal_policy_sets(env, goal)
    graph = navigation_graph(env)
    rows: list[dict[str, object]] = []
    for state, policy in sorted(policies.items()):
        if not policy:
            continue
        rows.append({
            "state": state,
            "target": canonical_label(policy),
            "optimal_set": policy,
            "degree": len(graph[state]),
        })
    return rows


def model_specs() -> list[ModelSpec]:
    specs = [ModelSpec("full_state", "full_state", "full", 0)]
    for mode in OBSERVATION_MODES:
        specs.append(ModelSpec(f"memoryless_{mode}", "memoryless", mode, 0))
        for history_len in HISTORY_LENGTHS:
            specs.append(ModelSpec(f"history_{mode}_h{history_len}", "history", mode, history_len))
    for history_len in HISTORY_LENGTHS:
        specs.append(ModelSpec(
            f"hybrid_counts_depth_last_h{history_len}",
            "hybrid_counts_depth_last",
            "depth_last",
            history_len,
        ))
    return specs


def model_key(row: dict[str, object], spec: ModelSpec) -> Key:
    state = row["state"]
    assert isinstance(state, tuple)
    if spec.architecture == "full_state":
        return state
    if spec.architecture == "memoryless":
        return observe_light(state, spec.observation_mode)
    if spec.architecture == "history":
        return prefix_history(state, spec.observation_mode, spec.history_len)
    if spec.architecture == "hybrid_counts_depth_last":
        return (observe_light(state, "counts"), prefix_history(state, "depth_last", spec.history_len))
    raise ValueError(f"unknown architecture: {spec.architecture}")


def observation_features(obs: Observation) -> list[float]:
    if isinstance(obs, tuple):
        values: list[float] = []
        for item in obs:
            if isinstance(item, tuple):
                values.extend(observation_features(item))
            else:
                values.append(float(item))
        return values
    return [float(obs)]


def pad(values: list[float], size: int) -> list[float]:
    return values[:size] + [0.0] * max(0, size - len(values))


def feature_width(spec: ModelSpec) -> int:
    if spec.architecture == "full_state":
        return 8
    if spec.architecture == "memoryless":
        return {"counts": 3, "noisy_counts": 3, "depth_last": 2, "depth": 1}[spec.observation_mode]
    if spec.architecture == "history":
        width = {"counts": 3, "noisy_counts": 3, "depth_last": 2, "depth": 1}[spec.observation_mode]
        return width * spec.history_len
    if spec.architecture == "hybrid_counts_depth_last":
        return 3 + 2 * spec.history_len
    raise ValueError(f"unknown architecture: {spec.architecture}")


def tree_features(row: dict[str, object], spec: ModelSpec) -> list[float]:
    state = row["state"]
    assert isinstance(state, tuple)
    if spec.architecture == "full_state":
        return pad([float(action + 1) for action in state], feature_width(spec))
    if spec.architecture == "memoryless":
        return pad(observation_features(observe_light(state, spec.observation_mode)), feature_width(spec))
    if spec.architecture == "history":
        values: list[float] = []
        for obs in prefix_history(state, spec.observation_mode, spec.history_len):
            values.extend(observation_features(obs))
        return pad(values, feature_width(spec))
    if spec.architecture == "hybrid_counts_depth_last":
        values = observation_features(observe_light(state, "counts"))
        for obs in prefix_history(state, "depth_last", spec.history_len):
            values.extend(observation_features(obs))
        return pad(values, feature_width(spec))
    raise ValueError(f"unknown architecture: {spec.architecture}")


def train_majority(rows: list[dict[str, object]], spec: ModelSpec) -> tuple[dict[Key, ActionLabel], ActionLabel]:
    by_key: dict[Key, Counter[ActionLabel]] = defaultdict(Counter)
    global_counts: Counter[ActionLabel] = Counter()
    for row in rows:
        target = row["target"]
        assert isinstance(target, str)
        by_key[model_key(row, spec)][target] += 1
        global_counts[target] += 1
    table = {
        key: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for key, counter in by_key.items()
    }
    fallback = sorted(global_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return table, fallback


def evaluate_lookup(rows: list[dict[str, object]], spec: ModelSpec) -> dict[str, float]:
    table, fallback = train_majority(rows, spec)
    return evaluate_table(rows, table, fallback, spec)


def evaluate_table(
    rows: list[dict[str, object]],
    table: dict[Key, ActionLabel],
    fallback: ActionLabel,
    spec: ModelSpec,
) -> dict[str, float]:
    exact = 0
    optimal = 0
    covered = 0
    for row in rows:
        key = model_key(row, spec)
        if key in table:
            covered += 1
        prediction = table.get(key, fallback)
        target = row["target"]
        optimal_set = row["optimal_set"]
        assert isinstance(target, str)
        assert isinstance(optimal_set, frozenset)
        exact += int(prediction == target)
        optimal += int(prediction in optimal_set)
    n = len(rows)
    return {
        "exact_accuracy": exact / n if n else float("nan"),
        "optimal_action_accuracy": optimal / n if n else float("nan"),
        "coverage": covered / n if n else float("nan"),
    }


def random_split(
    rows: list[dict[str, object]],
    *,
    seed: int,
    train_fraction: float = 0.7,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    shuffled = list(rows)
    Random(seed).shuffle(shuffled)
    cut = int(round(train_fraction * len(shuffled)))
    return shuffled[:cut], shuffled[cut:]


def evaluate_decision_tree(train: list[dict[str, object]], test: list[dict[str, object]], spec: ModelSpec) -> dict[str, float]:
    x_train = [tree_features(row, spec) for row in train]
    y_train = [row["target"] for row in train]
    x_test = [tree_features(row, spec) for row in test]
    y_test = [row["target"] for row in test]
    if DecisionTreeClassifier is None:
        by_feature: dict[tuple[float, ...], Counter[ActionLabel]] = defaultdict(Counter)
        fallback_counts: Counter[ActionLabel] = Counter()
        for features, target in zip(x_train, y_train):
            assert isinstance(target, str)
            by_feature[tuple(features)][target] += 1
            fallback_counts[target] += 1
        table = {
            features: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
            for features, counter in by_feature.items()
        }
        fallback = sorted(fallback_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        predictions = [table.get(tuple(features), fallback) for features in x_test]
    else:
        clf = DecisionTreeClassifier(random_state=0, min_samples_leaf=2)
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
    exact = accuracy_score(y_test, predictions)
    optimal = sum(
        int(prediction in row["optimal_set"])
        for prediction, row in zip(predictions, test)
    ) / len(test)
    return {
        "exact_accuracy": float(exact),
        "optimal_action_accuracy": float(optimal),
        "coverage": 1.0,
    }


def collect_results() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    specs = model_specs()
    for env in trace_monoid_family(max_depth=8):
        goals = selected_goals(env)
        for goal_index, goal in enumerate(goals):
            rows = example_rows(env, goal)
            if not rows:
                continue
            train, test = random_split(rows, seed=100 + goal_index)
            for spec in specs:
                lookup_metrics = evaluate_lookup(rows, spec)
                results.append({
                    "environment": env.name,
                    "goal_index": goal_index,
                    "goal_id": goal_id(goal),
                    "goal_depth": len(goal),
                    "split": "all",
                    "estimator": "lookup",
                    "model": spec.name,
                    "architecture": spec.architecture,
                    "observation_mode": spec.observation_mode,
                    "history_len": spec.history_len,
                    "n_train": len(rows),
                    "n_test": len(rows),
                    **lookup_metrics,
                })
                tree_metrics = evaluate_decision_tree(train, test, spec)
                results.append({
                    "environment": env.name,
                    "goal_index": goal_index,
                    "goal_id": goal_id(goal),
                    "goal_depth": len(goal),
                    "split": "random70",
                    "estimator": LEARNED_ESTIMATOR,
                    "model": spec.name,
                    "architecture": spec.architecture,
                    "observation_mode": spec.observation_mode,
                    "history_len": spec.history_len,
                    "n_train": len(train),
                    "n_test": len(test),
                    **tree_metrics,
                })
    return results


def write_csv(rows: list[dict[str, object]]) -> None:
    path = FIGURES / "lightweight_empirical_benchmark.csv"
    fields = [
        "environment",
        "goal_index",
        "goal_id",
        "goal_depth",
        "split",
        "estimator",
        "model",
        "architecture",
        "observation_mode",
        "history_len",
        "n_train",
        "n_test",
        "exact_accuracy",
        "optimal_action_accuracy",
        "coverage",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def grouped_metric(
    rows: list[dict[str, object]],
    *,
    split: str,
    estimator: str,
    metric: str,
) -> dict[tuple[str, str], tuple[float, float, int]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["split"] == split and row["estimator"] == estimator:
            grouped[(str(row["environment"]), str(row["model"]))].append(float(row[metric]))
    return {
        key: (mean(values), pstdev(values) if len(values) > 1 else 0.0, len(values))
        for key, values in grouped.items()
    }


def heatmap(
    rows: list[dict[str, object]],
    *,
    path: Path,
    split: str,
    estimator: str,
    models: list[str],
    title: str,
    subtitle: str,
    metric: str = "optimal_action_accuracy",
) -> None:
    values = grouped_metric(rows, split=split, estimator=estimator, metric=metric)
    width, height = 1050, 640
    margin_left, margin_right, margin_top, margin_bottom = 250, 50, 92, 198
    cell_w = (width - margin_left - margin_right) / len(ENV_NAMES)
    cell_h = (height - margin_top - margin_bottom) / len(models)

    def color(value: float) -> str:
        low = (254, 242, 242)
        high = (22, 163, 74)
        r = round(low[0] + (high[0] - low[0]) * value)
        g = round(low[1] + (high[1] - low[1]) * value)
        b = round(low[2] + (high[2] - low[2]) * value)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:19px}.tick{font-size:17px;fill:#52606d}.cell{font-size:18px;font-weight:700}</style>',
        f'<text class="title" x="70" y="38">{esc(title)}</text>',
        f'<text class="label" x="70" y="62">{esc(subtitle)}</text>',
    ]
    for row_idx, model in enumerate(models):
        y0 = margin_top + row_idx * cell_h
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 14}" y="{y0 + cell_h / 2 + 5:.1f}">{esc(model.replace("_", " "))}</text>')
        for col_idx, env in enumerate(ENV_NAMES):
            value, sd, n = values.get((env, model), (float("nan"), float("nan"), 0))
            x0 = margin_left + col_idx * cell_w
            fill = "#f5f7fa" if value != value else color(value)
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            text_color = "#ffffff" if value == value and value > 0.55 else "#1f2933"
            label = "NA" if value != value else f"{value:.2f}"
            parts.append(f'<text class="cell" text-anchor="middle" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2:.1f}" fill="{text_color}">{label}</text>')
            if n:
                parts.append(f'<text class="tick" text-anchor="middle" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 17:.1f}" fill="{text_color}">sd {sd:.2f}</text>')
    for col_idx, env in enumerate(ENV_NAMES):
        x0 = margin_left + col_idx * cell_w + cell_w / 2
        parts.append(f'<text class="tick" text-anchor="end" x="{x0 + 22:.1f}" y="{height - 102}" transform="rotate(-30 {x0 + 22:.1f} {height - 102})">{esc(env.replace("_", " "))}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def memory_capacity_curve(rows: list[dict[str, object]], path: Path) -> None:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["split"] == "all"
            and row["estimator"] == "lookup"
            and row["architecture"] == "history"
            and row["observation_mode"] == "depth_last"
        ):
            grouped[(str(row["environment"]), int(row["history_len"]))].append(float(row["optimal_action_accuracy"]))

    width, height = 920, 470
    margin_left, margin_right, margin_top, margin_bottom = 76, 170, 64, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    x_positions = {length: margin_left + idx * plot_w / (len(HISTORY_LENGTHS) - 1) for idx, length in enumerate(HISTORY_LENGTHS)}

    def y_pos(value: float) -> float:
        return margin_top + (1.0 - value) * plot_h

    colors = {
        "trace_commute_000": "#b91c1c",
        "trace_commute_033": "#c2410c",
        "trace_commute_067": "#2563eb",
        "trace_commute_100": "#15803d",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:19px}.tick{font-size:17px;fill:#52606d}</style>',
        '<text class="title" x="70" y="36">Memory-capacity curve</text>',
        '<text class="label" x="70" y="58">History over sparse depth+last observation; mean across goals</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#9aa5b1"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#9aa5b1"/>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_pos(tick)
        parts.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" stroke="#e4e7eb"/>')
        parts.append(f'<text class="tick" text-anchor="end" x="{margin_left - 10}" y="{y + 4:.1f}">{tick:.2f}</text>')
    for length, x in x_positions.items():
        parts.append(f'<text class="tick" text-anchor="middle" x="{x:.1f}" y="{height - 32}">{length}</text>')
    parts.append(f'<text class="label" text-anchor="middle" x="{margin_left + plot_w / 2:.1f}" y="{height - 10}">history length</text>')

    for env_idx, env in enumerate(ENV_NAMES):
        points: list[tuple[float, float]] = []
        for length in HISTORY_LENGTHS:
            vals = grouped.get((env, length), [])
            if vals:
                points.append((x_positions[length], y_pos(mean(vals))))
        if not points:
            continue
        path_data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        color = colors[env]
        parts.append(f'<polyline points="{path_data}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        legend_y = margin_top + env_idx * 26 + 14
        parts.append(f'<line x1="{width - margin_right + 22}" y1="{legend_y}" x2="{width - margin_right + 48}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text class="tick" x="{width - margin_right + 56}" y="{legend_y + 4}">{esc(env.replace("_", " "))}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def architecture_gap_summary(rows: list[dict[str, object]]) -> list[str]:
    values = grouped_metric(rows, split="all", estimator="lookup", metric="optimal_action_accuracy")
    lines = []
    for env in ENV_NAMES:
        counts = values[(env, "memoryless_counts")][0]
        h4 = values[(env, "history_depth_last_h4")][0]
        hybrid = values[(env, "hybrid_counts_depth_last_h4")][0]
        lines.append(f"{env}: counts={counts:.3f}, history_depth_last_h4={h4:.3f}, hybrid_h4={hybrid:.3f}")
    return lines


def write_figures(rows: list[dict[str, object]]) -> None:
    heatmap(
        rows,
        path=FIGURES / "lightweight_sparsity_commutativity.svg",
        split="all",
        estimator="lookup",
        models=["memoryless_counts", "memoryless_noisy_counts", "memoryless_depth_last", "memoryless_depth"],
        title="Sparsity x commutativity benchmark",
        subtitle="Cell = multi-goal optimal-action accuracy, lookup sufficiency diagnostic",
    )
    heatmap(
        rows,
        path=FIGURES / "lightweight_multigoal_robustness.svg",
        split="all",
        estimator="lookup",
        models=[
            "full_state",
            "memoryless_counts",
            "history_counts_h4",
            "history_depth_last_h4",
            "hybrid_counts_depth_last_h4",
        ],
        title="Multi-goal robustness",
        subtitle="Cell = mean accuracy across fixed and sampled goals; sd shown below",
    )
    heatmap(
        rows,
        path=FIGURES / "lightweight_architecture_comparison.svg",
        split="random70",
        estimator=LEARNED_ESTIMATOR,
        models=[
            "full_state",
            "memoryless_counts",
            "memoryless_noisy_counts",
            "history_depth_last_h4",
            "hybrid_counts_depth_last_h4",
        ],
        title="Lightweight learned architecture comparison",
        subtitle=f"{LEARNED_ESTIMATOR.replace('_', ' ')}, random 70/30 split, mean across goals",
    )
    memory_capacity_curve(rows, FIGURES / "lightweight_memory_capacity.svg")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = collect_results()
    write_csv(rows)
    write_figures(rows)
    print("Generated:")
    print((FIGURES / "lightweight_empirical_benchmark.csv").relative_to(ROOT))
    print((FIGURES / "lightweight_sparsity_commutativity.svg").relative_to(ROOT))
    print((FIGURES / "lightweight_multigoal_robustness.svg").relative_to(ROOT))
    print((FIGURES / "lightweight_architecture_comparison.svg").relative_to(ROOT))
    print((FIGURES / "lightweight_memory_capacity.svg").relative_to(ROOT))
    print()
    print("All/lookup architecture gap summary:")
    for line in architecture_gap_summary(rows):
        print(line)


if __name__ == "__main__":
    main()
