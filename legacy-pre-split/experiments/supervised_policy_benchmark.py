"""Dependency-free supervised policy benchmark for trace environments.

The benchmark uses shortest-path policies from ``policy_ambiguity.py`` as labels.
Models are table-based classifiers that differ only in the information available
to their key: current observation, full latent state, finite observation history,
or a hybrid current-count plus history key.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict, deque
from pathlib import Path
from random import Random
from typing import Hashable

try:
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    DecisionTreeClassifier = None

    def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
        return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)

from commutativity_probe import Environment, trace_monoid_family
from observation_ambiguity import observe
from policy_ambiguity import goal_state, navigation_graph, optimal_policy_sets
from plot_current_results import esc


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
LEARNED_ESTIMATOR = "decision_tree" if DecisionTreeClassifier is not None else "feature_lookup"

ActionLabel = str
Key = Hashable


def canonical_label(policy: frozenset[str]) -> str:
    """Use a deterministic single target when multiple optimal actions exist."""
    return sorted(policy)[0]


def prefix_history(state: tuple[int, ...], mode: str, history_len: int) -> tuple[Hashable, ...]:
    """Observation history along the canonical path from root to the state."""
    prefixes = [state[:idx] for idx in range(len(state) + 1)]
    observations = [observe(prefix, mode) for prefix in prefixes]
    if history_len <= 0:
        return tuple()
    return tuple(observations[-history_len:])


def example_rows(env: Environment) -> list[dict[str, object]]:
    goal = goal_state(env)
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


def model_key(row: dict[str, object], model: str, history_len: int) -> Key:
    state = row["state"]
    assert isinstance(state, tuple)
    if model == "full_state":
        return state
    if model == "memoryless_counts":
        return observe(state, "counts")
    if model == "memoryless_depth_last":
        return observe(state, "depth_last")
    if model == "history_counts":
        return prefix_history(state, "counts", history_len)
    if model == "history_depth_last":
        return prefix_history(state, "depth_last", history_len)
    if model == "hybrid_counts_depth_last_history":
        return (observe(state, "counts"), prefix_history(state, "depth_last", history_len))
    raise ValueError(f"unknown model: {model}")


def pad(values: list[float], size: int) -> list[float]:
    return values[:size] + [0.0] * max(0, size - len(values))


def observation_features(obs: Hashable) -> list[float]:
    if isinstance(obs, tuple):
        return [float(value) for value in obs]
    return [float(obs)]


def tree_features(row: dict[str, object], model: str, history_len: int) -> list[float]:
    state = row["state"]
    assert isinstance(state, tuple)
    max_depth = 8
    n_actions = 3
    if model == "full_state":
        return pad([float(action + 1) for action in state], max_depth)
    if model == "memoryless_counts":
        return observation_features(observe(state, "counts"))
    if model == "memoryless_depth_last":
        return observation_features(observe(state, "depth_last"))
    if model == "history_counts":
        values: list[float] = []
        for obs in prefix_history(state, "counts", history_len):
            values.extend(observation_features(obs))
        return pad(values, history_len * n_actions)
    if model == "history_depth_last":
        values = []
        for obs in prefix_history(state, "depth_last", history_len):
            values.extend(observation_features(obs))
        return pad(values, history_len * 2)
    if model == "hybrid_counts_depth_last_history":
        values = observation_features(observe(state, "counts"))
        for obs in prefix_history(state, "depth_last", history_len):
            values.extend(observation_features(obs))
        return pad(values, n_actions + history_len * 2)
    raise ValueError(f"unknown model: {model}")


def train_majority(rows: list[dict[str, object]], model: str, history_len: int) -> tuple[dict[Key, ActionLabel], ActionLabel]:
    by_key: dict[Key, Counter[ActionLabel]] = defaultdict(Counter)
    global_counts: Counter[ActionLabel] = Counter()
    for row in rows:
        target = row["target"]
        assert isinstance(target, str)
        by_key[model_key(row, model, history_len)][target] += 1
        global_counts[target] += 1
    table = {
        key: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for key, counter in by_key.items()
    }
    fallback = sorted(global_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return table, fallback


def evaluate(rows: list[dict[str, object]], table: dict[Key, ActionLabel], fallback: ActionLabel, model: str, history_len: int) -> dict[str, float]:
    exact = 0
    optimal = 0
    covered = 0
    for row in rows:
        key = model_key(row, model, history_len)
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


def split_rows(rows: list[dict[str, object]], split: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Deterministic splits testing interpolation and extrapolation over depth."""
    if split == "all":
        return rows, rows
    if split == "even_odd_depth":
        train = [row for row in rows if len(row["state"]) % 2 == 0]
        test = [row for row in rows if len(row["state"]) % 2 == 1]
        return train, test
    if split == "shallow_deep":
        train = [row for row in rows if len(row["state"]) <= 5]
        test = [row for row in rows if len(row["state"]) > 5]
        return train, test
    raise ValueError(f"unknown split: {split}")


def random_split(rows: list[dict[str, object]], *, seed: int, train_fraction: float = 0.7) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    shuffled = list(rows)
    Random(seed).shuffle(shuffled)
    cut = int(round(train_fraction * len(shuffled)))
    return shuffled[:cut], shuffled[cut:]


def evaluate_decision_tree(train: list[dict[str, object]], test: list[dict[str, object]], model: str, history_len: int) -> dict[str, float]:
    x_train = [tree_features(row, model, history_len) for row in train]
    y_train = [row["target"] for row in train]
    x_test = [tree_features(row, model, history_len) for row in test]
    y_test = [row["target"] for row in test]
    if DecisionTreeClassifier is None:
        by_feature: dict[tuple[float, ...], Counter[ActionLabel]] = defaultdict(Counter)
        fallback_counts: Counter[ActionLabel] = Counter()
        for features, target in zip(x_train, y_train):
            assert isinstance(target, str)
            by_feature[tuple(features)][target] += 1
            fallback_counts[target] += 1
        table = {
            key: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
            for key, counter in by_feature.items()
        }
        fallback = sorted(fallback_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        predictions = [table.get(tuple(features), fallback) for features in x_test]
    else:
        clf = DecisionTreeClassifier(random_state=0)
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
    models = [
        "full_state",
        "memoryless_counts",
        "memoryless_depth_last",
        "history_counts",
        "history_depth_last",
        "hybrid_counts_depth_last_history",
    ]
    splits = ["all", "random70", "even_odd_depth", "shallow_deep"]
    history_len = 4
    result_rows: list[dict[str, object]] = []
    for env in trace_monoid_family(max_depth=8):
        rows = example_rows(env)
        for split in splits:
            train, test = random_split(rows, seed=11) if split == "random70" else split_rows(rows, split)
            if not train or not test:
                continue
            for model in models:
                table, fallback = train_majority(train, model, history_len)
                metrics = evaluate(test, table, fallback, model, history_len)
                result_rows.append({
                    "environment": env.name,
                    "split": split,
                    "estimator": "lookup",
                    "model": model,
                    "history_len": history_len,
                    "n_train": len(train),
                    "n_test": len(test),
                    **metrics,
                })
                if split != "all":
                    tree_metrics = evaluate_decision_tree(train, test, model, history_len)
                    result_rows.append({
                        "environment": env.name,
                        "split": split,
                        "estimator": LEARNED_ESTIMATOR,
                        "model": model,
                        "history_len": history_len,
                        "n_train": len(train),
                        "n_test": len(test),
                        **tree_metrics,
                    })
    return result_rows


def write_csv(rows: list[dict[str, object]]) -> None:
    path = FIGURES / "supervised_policy_benchmark.csv"
    fields = [
        "environment",
        "split",
        "estimator",
        "model",
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


def heatmap(rows: list[dict[str, object]], *, split: str, estimator: str, metric: str) -> str:
    envs = ["trace_commute_000", "trace_commute_033", "trace_commute_067", "trace_commute_100"]
    models = [
        "full_state",
        "memoryless_counts",
        "memoryless_depth_last",
        "history_counts",
        "history_depth_last",
        "hybrid_counts_depth_last_history",
    ]
    values = {
        (row["environment"], row["model"]): float(row[metric])
        for row in rows
        if row["split"] == split and row["estimator"] == estimator
    }
    width, height = 1040, 600
    margin_left, margin_right, margin_top, margin_bottom = 305, 50, 90, 190
    cell_w = (width - margin_left - margin_right) / len(envs)
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
        f'<text class="title" x="80" y="38">Supervised policy benchmark: {esc(split)} / {esc(estimator)}</text>',
        f'<text class="label" x="80" y="60">Cell = {esc(metric.replace("_", " "))}; history length 4 where applicable</text>',
    ]
    for row_idx, model in enumerate(models):
        y0 = margin_top + row_idx * cell_h
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 14}" y="{y0 + cell_h / 2 + 5:.1f}">{esc(model.replace("_", " "))}</text>')
        for col_idx, env in enumerate(envs):
            value = values.get((env, model), float("nan"))
            x0 = margin_left + col_idx * cell_w
            fill = "#f5f7fa" if value != value else color(value)
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            text_color = "#ffffff" if value == value and value > 0.55 else "#1f2933"
            text = "NA" if value != value else f"{value:.2f}"
            parts.append(f'<text class="cell" text-anchor="middle" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 5:.1f}" fill="{text_color}">{text}</text>')
    for col_idx, env in enumerate(envs):
        x0 = margin_left + col_idx * cell_w + cell_w / 2
        parts.append(f'<text class="tick" text-anchor="end" x="{x0 + 22:.1f}" y="{height - 102}" transform="rotate(-30 {x0 + 22:.1f} {height - 102})">{esc(env.replace("_", " "))}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = collect_results()
    write_csv(rows)
    plot_specs = [
        ("all", "lookup"),
        ("random70", "lookup"),
        ("random70", LEARNED_ESTIMATOR),
        ("even_odd_depth", LEARNED_ESTIMATOR),
        ("shallow_deep", LEARNED_ESTIMATOR),
    ]
    for split, estimator in plot_specs:
        for metric in ["exact_accuracy", "optimal_action_accuracy"]:
            path = FIGURES / f"supervised_policy_{split}_{estimator}_{metric}.svg"
            path.write_text(heatmap(rows, split=split, estimator=estimator, metric=metric), encoding="utf-8")
    print("Generated:")
    print((FIGURES / "supervised_policy_benchmark.csv").relative_to(ROOT))
    for split, estimator in plot_specs:
        print((FIGURES / f"supervised_policy_{split}_{estimator}_exact_accuracy.svg").relative_to(ROOT))
        print((FIGURES / f"supervised_policy_{split}_{estimator}_optimal_action_accuracy.svg").relative_to(ROOT))


if __name__ == "__main__":
    main()
