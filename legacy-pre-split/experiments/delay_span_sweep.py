"""Delay-by-memory-span sweep for supervised sparse cue navigation."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from plot_current_results import esc
from rnn_navigation_benchmark import (
    Dataset,
    NumpyRNN,
    TaskSpec,
    accuracy_score,
    aliased_t_maze,
    build_dataset,
    fit_linear_predictor,
    lin_block_sequence_states,
    pad_history_features,
    pad_last_features,
    reservoir_states,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DELAYS = (4, 16, 32, 64, 80)
SPANS = (4, 16, 32, 64, 80)
DATASET_SEEDS = (11, 17)
REGIMES = (
    ("dense", "temporal_full", "visible_state"),
    ("sparse_aliased", "temporal_cue_then_blank", "cue_gap"),
)


def task_spec(delay: int) -> TaskSpec:
    return TaskSpec(
        "long_aliased_t_maze",
        aliased_t_maze(corridor_length=delay),
        (),
        f"delay_{delay}",
        "long_delayed_landmark_choice",
        max_examples=2 * delay + 64,
        n_goals=2,
    )


def add_row(
    rows: list[dict[str, object]],
    dataset: Dataset,
    delay: int,
    seed: int,
    model: str,
    state_dim: int,
    predictions: np.ndarray,
    memory_span: int | str = "",
) -> None:
    rows.append({
        "delay": delay,
        "cue_visibility": "visible_state" if dataset.obs_mode == "dense" else "cue_gap",
        "obs_mode": dataset.obs_mode,
        "temporal_mode": dataset.temporal_mode,
        "dataset_seed": seed,
        "model": model,
        "memory_span": memory_span,
        "state_dim": state_dim,
        "n_train": len(dataset.y_train),
        "n_test": len(dataset.y_test),
        "action_accuracy": accuracy_score(dataset.y_test, list(predictions)),
    })


def evaluate_dataset(dataset: Dataset, delay: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    last_train = pad_last_features(dataset.x_train)
    last_test = pad_last_features(dataset.x_test)
    memoryless = fit_linear_predictor(last_train, dataset.y_train)
    add_row(rows, dataset, delay, seed, "memoryless_linear", last_train.shape[1], memoryless(last_test))
    if dataset.obs_mode == "dense":
        return rows

    for span in SPANS:
        history_train = pad_history_features(dataset.x_train, span)
        history_test = pad_history_features(dataset.x_test, span)
        history = fit_linear_predictor(history_train, dataset.y_train)
        add_row(rows, dataset, delay, seed, f"history_h{span}", history_train.shape[1], history(history_test), span)

        lin_train, _ = lin_block_sequence_states(dataset.x_train, 3, span)
        lin_test, _ = lin_block_sequence_states(dataset.x_test, 3, span)
        lin = fit_linear_predictor(lin_train, dataset.y_train)
        add_row(rows, dataset, delay, seed, f"lin_block_r3_l{span}", lin_train.shape[1], lin(lin_test), span)

    for hidden_dim in (32, 128):
        rnn = NumpyRNN(dataset.feature_dim, hidden_dim=hidden_dim, output_dim=len(dataset.action_labels), seed=seed + hidden_dim)
        rnn.fit(dataset.x_train, dataset.y_train, epochs=7, lr=0.015, seed=seed + hidden_dim)
        add_row(rows, dataset, delay, seed, f"numpy_rnn_h{hidden_dim}", hidden_dim, rnn.predict(dataset.x_test))

        train_states, _ = reservoir_states(
            dataset.x_train,
            "nilpotent",
            hidden_dim=hidden_dim,
            input_dim=dataset.feature_dim,
            seed=seed + hidden_dim,
        )
        test_states, _ = reservoir_states(
            dataset.x_test,
            "nilpotent",
            hidden_dim=hidden_dim,
            input_dim=dataset.feature_dim,
            seed=seed + hidden_dim,
        )
        reservoir = fit_linear_predictor(train_states, dataset.y_train)
        add_row(rows, dataset, delay, seed, f"reservoir_nilpotent_h{hidden_dim}", hidden_dim, reservoir(test_states))
    return rows


def collect() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for delay_idx, delay in enumerate(DELAYS):
        task = task_spec(delay)
        for regime_idx, (obs_mode, temporal_mode, _) in enumerate(REGIMES):
            for seed in DATASET_SEEDS:
                dataset = build_dataset(task, obs_mode, temporal_mode=temporal_mode, seed=seed + delay_idx)
                rows.extend(evaluate_dataset(dataset, delay, 7000 + seed + 1000 * regime_idx + delay))
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str, object], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["delay"]), str(row["cue_visibility"]), str(row["model"]), row["memory_span"])].append(row)
    summary = []
    for (delay, cue_visibility, model, memory_span), group in sorted(grouped.items()):
        values = np.asarray([float(row["action_accuracy"]) for row in group], dtype=float)
        summary.append({
            "delay": delay,
            "cue_visibility": cue_visibility,
            "model": model,
            "memory_span": memory_span,
            "state_dim": group[0]["state_dim"],
            "n_runs": len(group),
            "action_accuracy_mean": float(np.mean(values)),
            "action_accuracy_sem": float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
        })
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def panel(
    parts: list[str],
    summary: list[dict[str, object]],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    family_prefix: str,
    title: str,
) -> None:
    values = {
        (int(row["delay"]), int(row["memory_span"])): float(row["action_accuracy_mean"])
        for row in summary
        if row["cue_visibility"] == "cue_gap" and str(row["model"]).startswith(family_prefix)
    }
    cell_w = width / len(SPANS)
    cell_h = height / len(DELAYS)

    def color(value: float) -> str:
        low = (254, 242, 242)
        high = (22, 101, 52)
        return "rgb({},{},{})".format(*[
            round(low[idx] + (high[idx] - low[idx]) * value) for idx in range(3)
        ])

    parts.append(f'<text class="label" x="{x0:.1f}" y="{y0 - 16:.1f}">{esc(title)}</text>')
    for row_idx, delay in enumerate(DELAYS):
        y = y0 + row_idx * cell_h
        parts.append(f'<text class="tick" text-anchor="end" x="{x0 - 12:.1f}" y="{y + cell_h / 2 + 4:.1f}">D {delay}</text>')
        for col_idx, span in enumerate(SPANS):
            x = x0 + col_idx * cell_w
            value = values.get((delay, span), float("nan"))
            fill = "#f5f7fa" if value != value else color(value)
            label = "NA" if value != value else f"{value:.2f}"
            text_color = "#ffffff" if value == value and value > 0.60 else "#1f2933"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            parts.append(f'<text class="cell" text-anchor="middle" x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 4:.1f}" fill="{text_color}">{label}</text>')
    for col_idx, span in enumerate(SPANS):
        x = x0 + (col_idx + 0.5) * cell_w
        parts.append(f'<text class="tick" text-anchor="middle" x="{x:.1f}" y="{y0 + height + 24:.1f}">span {span}</text>')


def plot(summary: list[dict[str, object]], path: Path) -> None:
    width, height = 1160, 570
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:19px}.cell{font-size:17px;font-weight:700}.tick{font-size:17px;fill:#52606d}</style>',
        '<text class="title" x="54" y="38">Delay x memory span sweep</text>',
        '<text class="label" x="54" y="64">Sparse cue-gap T-maze; cell = mean supervised action accuracy</text>',
    ]
    panel(parts, summary, x0=115, y0=128, width=400, height=320, family_prefix="history_h", title="Explicit observed history")
    panel(parts, summary, x0=670, y0=128, width=400, height=320, family_prefix="lin_block_r3_l", title="Independent Lin blocks")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = collect()
    summary = summarize(rows)
    write_csv(FIGURES / "delay_span_sweep.csv", rows)
    write_csv(FIGURES / "delay_span_summary.csv", summary)
    plot(summary, FIGURES / "delay_span_sweep.svg")
    print("Generated:")
    print((FIGURES / "delay_span_sweep.csv").relative_to(ROOT))
    print((FIGURES / "delay_span_summary.csv").relative_to(ROOT))
    print((FIGURES / "delay_span_sweep.svg").relative_to(ROOT))


if __name__ == "__main__":
    main()
