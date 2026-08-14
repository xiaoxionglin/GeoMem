"""Factorized supervised navigation architecture sweep for GeoMem."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from commutativity_probe import Environment, room_graph_environment, topology_edges
from plot_current_results import esc
from rnn_navigation_benchmark import (
    Dataset,
    NumpyRNN,
    TaskSpec,
    accuracy_score,
    build_dataset,
    fit_linear_predictor,
    lin_block_sequence_states,
    pad_history_features,
    pad_last_features,
    reservoir_states,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
ROOM_SIZE = 3
DATASET_SEEDS = (11, 17)
TEMPORAL_MODES = ("temporal_full", "temporal_cue_then_blank")
OBSERVATION_MODES = ("gaussian_sensory", "vector", "sparse_aliased")
HISTORY_LENGTH = 8
RNN_HIDDEN_DIM = 32

State = tuple[int, ...]


def task_specs() -> list[TaskSpec]:
    return [
        TaskSpec(
            name=f"room_graph_{topology}",
            env=room_graph_environment(topology),
            root=(0, ROOM_SIZE // 2, ROOM_SIZE // 2),
            geometry=topology,
            behavior_family="room_graph_navigation",
            max_examples=420,
            n_goals=3,
        )
        for topology in ("loop_rich", "bottleneck", "tree_like")
    ]


def labels_by_last_frame(dataset: Dataset) -> dict[bytes, set[int]]:
    labels: dict[bytes, set[int]] = defaultdict(set)
    for xs, ys in ((dataset.x_train, dataset.y_train), (dataset.x_test, dataset.y_test)):
        for x, y in zip(xs, ys):
            labels[np.asarray(x[-1], dtype=np.float32).tobytes()].add(int(y))
    return labels


def decision_indices(dataset: Dataset) -> list[int]:
    labels = labels_by_last_frame(dataset)
    indices = [
        idx for idx, x in enumerate(dataset.x_test)
        if len(labels[np.asarray(x[-1], dtype=np.float32).tobytes()]) > 1
    ]
    return indices or list(range(len(dataset.y_test)))


def index_accuracy(targets: list[int], predictions: np.ndarray, indices: list[int]) -> float:
    return accuracy_score([targets[idx] for idx in indices], [int(predictions[idx]) for idx in indices])


def hybrid_features(vector_dataset: Dataset, observed_dataset: Dataset, history_len: int) -> tuple[np.ndarray, np.ndarray]:
    current_vector_train = pad_last_features(vector_dataset.x_train)
    current_vector_test = pad_last_features(vector_dataset.x_test)
    observed_history_train = pad_history_features(observed_dataset.x_train, history_len)
    observed_history_test = pad_history_features(observed_dataset.x_test, history_len)
    return (
        np.concatenate([current_vector_train, observed_history_train], axis=1),
        np.concatenate([current_vector_test, observed_history_test], axis=1),
    )


def row(
    dataset: Dataset,
    topology: str,
    model: str,
    seed: int,
    predictions: np.ndarray,
    state_dim: int,
    decision: list[int],
) -> dict[str, object]:
    return {
        "task": dataset.task.name,
        "topology": topology,
        "aliasing": "low" if dataset.obs_mode == "vector" else "high",
        "obs_mode": dataset.obs_mode,
        "cue_visibility": "persistent" if dataset.temporal_mode == "temporal_full" else "cue_gap",
        "temporal_mode": dataset.temporal_mode,
        "dataset_seed": seed,
        "model": model,
        "state_dim": state_dim,
        "n_train": len(dataset.y_train),
        "n_test": len(dataset.y_test),
        "decision_fraction": len(decision) / len(dataset.y_test),
        "action_accuracy": accuracy_score(dataset.y_test, list(predictions)),
        "decision_accuracy": index_accuracy(dataset.y_test, predictions, decision),
    }


def evaluate_dataset(dataset: Dataset, vector_full_dataset: Dataset, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    decision = decision_indices(dataset)
    last_train = pad_last_features(dataset.x_train)
    last_test = pad_last_features(dataset.x_test)
    memoryless = fit_linear_predictor(last_train, dataset.y_train)
    rows.append(row(dataset, dataset.task.geometry, "memoryless_linear", seed, memoryless(last_test), last_train.shape[1], decision))

    history_train = pad_history_features(dataset.x_train, HISTORY_LENGTH)
    history_test = pad_history_features(dataset.x_test, HISTORY_LENGTH)
    history = fit_linear_predictor(history_train, dataset.y_train)
    rows.append(row(dataset, dataset.task.geometry, f"history_h{HISTORY_LENGTH}", seed, history(history_test), history_train.shape[1], decision))

    hybrid_train, hybrid_test = hybrid_features(vector_full_dataset, dataset, HISTORY_LENGTH)
    hybrid = fit_linear_predictor(hybrid_train, dataset.y_train)
    rows.append(row(dataset, dataset.task.geometry, f"hybrid_vector_history_h{HISTORY_LENGTH}", seed, hybrid(hybrid_test), hybrid_train.shape[1], decision))

    rnn = NumpyRNN(dataset.feature_dim, hidden_dim=RNN_HIDDEN_DIM, output_dim=len(dataset.action_labels), seed=seed)
    rnn.fit(dataset.x_train, dataset.y_train, epochs=7, lr=0.015, seed=seed)
    rows.append(row(dataset, dataset.task.geometry, f"numpy_rnn_h{RNN_HIDDEN_DIM}", seed, rnn.predict(dataset.x_test), RNN_HIDDEN_DIM, decision))

    for kind in ("diagonal", "nilpotent"):
        train_states, _ = reservoir_states(dataset.x_train, kind, hidden_dim=RNN_HIDDEN_DIM, input_dim=dataset.feature_dim, seed=seed + 31)
        test_states, _ = reservoir_states(dataset.x_test, kind, hidden_dim=RNN_HIDDEN_DIM, input_dim=dataset.feature_dim, seed=seed + 31)
        predictor = fit_linear_predictor(train_states, dataset.y_train)
        rows.append(row(dataset, dataset.task.geometry, f"reservoir_{kind}_h{RNN_HIDDEN_DIM}", seed, predictor(test_states), RNN_HIDDEN_DIM, decision))

    lin_states_train, _ = lin_block_sequence_states(dataset.x_train, 3, 16)
    lin_states_test, _ = lin_block_sequence_states(dataset.x_test, 3, 16)
    lin = fit_linear_predictor(lin_states_train, dataset.y_train)
    rows.append(row(dataset, dataset.task.geometry, "lin_block_r3_l16", seed, lin(lin_states_test), lin_states_train.shape[1], decision))
    return rows


def collect() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_idx, task in enumerate(task_specs()):
        for temporal_idx, temporal_mode in enumerate(TEMPORAL_MODES):
            for seed in DATASET_SEEDS:
                vector_full = build_dataset(task, "vector", temporal_mode="temporal_full", seed=seed + task_idx)
                for obs_idx, obs_mode in enumerate(OBSERVATION_MODES):
                    dataset = build_dataset(task, obs_mode, temporal_mode=temporal_mode, seed=seed + task_idx)
                    rows.extend(evaluate_dataset(dataset, vector_full, 3000 + seed + 100 * obs_idx + 1000 * temporal_idx))
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for record in rows:
        grouped[(str(record["topology"]), str(record["aliasing"]), str(record["cue_visibility"]), str(record["model"]))].append(record)
    summary = []
    for (topology, aliasing, cue_visibility, model), group in sorted(grouped.items()):
        accuracy = np.asarray([float(record["action_accuracy"]) for record in group], dtype=float)
        decision = np.asarray([float(record["decision_accuracy"]) for record in group], dtype=float)
        summary.append({
            "topology": topology,
            "aliasing": aliasing,
            "cue_visibility": cue_visibility,
            "model": model,
            "state_dim": group[0]["state_dim"],
            "n_runs": len(group),
            "action_accuracy_mean": float(np.mean(accuracy)),
            "action_accuracy_sem": float(np.std(accuracy, ddof=1) / math.sqrt(len(accuracy))) if len(accuracy) > 1 else 0.0,
            "decision_accuracy_mean": float(np.mean(decision)),
            "decision_accuracy_sem": float(np.std(decision, ddof=1) / math.sqrt(len(decision))) if len(decision) > 1 else 0.0,
            "decision_fraction_mean": float(np.mean([float(record["decision_fraction"]) for record in group])),
        })
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary: list[dict[str, object]], path: Path) -> None:
    models = (
        "memoryless_linear",
        f"history_h{HISTORY_LENGTH}",
        f"hybrid_vector_history_h{HISTORY_LENGTH}",
        f"numpy_rnn_h{RNN_HIDDEN_DIM}",
        f"reservoir_diagonal_h{RNN_HIDDEN_DIM}",
        f"reservoir_nilpotent_h{RNN_HIDDEN_DIM}",
        "lin_block_r3_l16",
    )
    columns = [
        (topology, aliasing, visibility)
        for topology in ("loop_rich", "bottleneck", "tree_like")
        for aliasing in ("low", "high")
        for visibility in ("persistent", "cue_gap")
    ]
    values = {
        (str(row["model"]), str(row["topology"]), str(row["aliasing"]), str(row["cue_visibility"])): float(row["decision_accuracy_mean"])
        for row in summary
    }
    width, height = 1320, 620
    margin_left, margin_right, margin_top, margin_bottom = 260, 36, 95, 165
    cell_w = (width - margin_left - margin_right) / len(columns)
    cell_h = (height - margin_top - margin_bottom) / len(models)

    def color(value: float) -> str:
        low = (254, 242, 242)
        high = (22, 101, 52)
        return "rgb({},{},{})".format(*[
            round(low[idx] + (high[idx] - low[idx]) * value) for idx in range(3)
        ])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:18px}.cell{font-size:17px;font-weight:700}.tick{font-size:16px;fill:#52606d}</style>',
        '<text class="title" x="54" y="38">Systematic navigation architecture sweep</text>',
        '<text class="label" x="54" y="64">Cell = policy accuracy on observation-aliased test decisions</text>',
    ]
    for row_idx, model in enumerate(models):
        y0 = margin_top + row_idx * cell_h
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 14}" y="{y0 + cell_h / 2 + 4:.1f}">{esc(model.replace("_", " "))}</text>')
        for col_idx, column in enumerate(columns):
            x0 = margin_left + col_idx * cell_w
            value = values.get((model, *column), float("nan"))
            fill = "#f5f7fa" if value != value else color(value)
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            label = "NA" if value != value else f"{value:.2f}"
            text_color = "#ffffff" if value == value and value > 0.60 else "#1f2933"
            parts.append(f'<text class="cell" text-anchor="middle" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 4:.1f}" fill="{text_color}">{label}</text>')
    for col_idx, (topology, aliasing, visibility) in enumerate(columns):
        x = margin_left + (col_idx + 0.5) * cell_w
        label = f"{topology} / {aliasing} alias / {visibility}"
        parts.append(f'<text class="tick" text-anchor="end" x="{x + 40:.1f}" y="{height - 165}" transform="rotate(-42 {x + 40:.1f} {height - 165})">{esc(label.replace("_", " "))}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = collect()
    summary = summarize(rows)
    write_csv(FIGURES / "systematic_navigation_sweep.csv", rows)
    write_csv(FIGURES / "systematic_navigation_summary.csv", summary)
    plot(summary, FIGURES / "systematic_navigation_sweep.svg")
    print("Generated:")
    print((FIGURES / "systematic_navigation_sweep.csv").relative_to(ROOT))
    print((FIGURES / "systematic_navigation_summary.csv").relative_to(ROOT))
    print((FIGURES / "systematic_navigation_sweep.svg").relative_to(ROOT))


if __name__ == "__main__":
    main()
