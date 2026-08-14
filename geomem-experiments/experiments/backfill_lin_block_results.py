"""Backfill independent Lin-block rows into existing navigation benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rnn_navigation_benchmark import (
    ACTION_EMBEDDING_DIM,
    ACTION_INPUT_MODES,
    DATASET_SEEDS,
    FIGURES,
    LIN_BLOCK_CONFIGS,
    TEMPORAL_DROPOUT_P,
    TEMPORAL_MODES,
    accuracy_score,
    action_ablation_plot,
    build_dataset,
    context_readout_accuracy,
    fit_linear_predictor,
    heatmap,
    lin_block_sequence_states,
    lin_sequence_grid_plot,
    lin_sequence_injection_summary,
    lin_sequence_summary,
    memory_advantage_plot,
    memory_advantage_summary,
    output_path,
    parse_csv_arg,
    pad_last_features,
    result_row,
    selected_task_specs,
    summarize_rows,
    temporal_reservoir_plot,
    temporal_sparsity_advantage,
    temporal_sparsity_summary,
    write_csv,
)


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def is_lin_block(row: dict[str, object]) -> bool:
    return str(row["model"]).startswith("lin_block_")


def collect_lin_block_rows(
    *,
    task_suite: str,
    obs_modes: tuple[str, ...],
    temporal_modes: tuple[str, ...],
    action_inputs: tuple[str, ...],
    action_embedding_dim: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tasks = selected_task_specs(task_suite)
    for task_idx, task in enumerate(tasks):
        for action_idx, action_input in enumerate(action_inputs):
            current_embedding_dim = action_embedding_dim if action_input == "embedding" else 0
            for obs_mode in obs_modes:
                for temporal_idx, temporal_mode in enumerate(temporal_modes):
                    for dataset_seed in DATASET_SEEDS:
                        dataset = build_dataset(
                            task,
                            obs_mode,
                            temporal_mode=temporal_mode,
                            temporal_dropout_p=TEMPORAL_DROPOUT_P,
                            seed=dataset_seed + task_idx,
                            action_input=action_input,
                            action_embedding_dim=current_embedding_dim,
                        )
                        seed = 100 + dataset_seed + task_idx + 100 * action_idx + 1000 * temporal_idx
                        for sequence_r, sequence_l in LIN_BLOCK_CONFIGS:
                            train_states, _ = lin_block_sequence_states(dataset.x_train, sequence_r, sequence_l)
                            test_states, _ = lin_block_sequence_states(dataset.x_test, sequence_r, sequence_l)
                            predictor = fit_linear_predictor(train_states, dataset.y_train)

                            rows.append(result_row(
                                dataset,
                                f"lin_block_r{sequence_r}_l{sequence_l}",
                                dataset_seed,
                                seed + 900 + 17 * sequence_r + 31 * sequence_l,
                                accuracy_score(dataset.y_test, list(predictor(test_states))),
                                float("nan"),
                                sequence_r=sequence_r,
                                sequence_l=sequence_l,
                                input_injection="independent_blocks",
                                context_accuracy=context_readout_accuracy(
                                    train_states,
                                    test_states,
                                    None,
                                ),
                            ))
    return rows


def write_merged_outputs(
    *,
    output_prefix: str,
    task_suite: str,
    obs_modes: tuple[str, ...],
    action_inputs: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    summary_rows = summarize_rows(rows)
    advantage_rows = memory_advantage_summary(summary_rows)
    temporal_summary_rows = temporal_sparsity_summary(summary_rows)
    temporal_advantage_rows = temporal_sparsity_advantage(advantage_rows)
    lin_rows = lin_sequence_summary(summary_rows)
    lin_injection_rows = lin_sequence_injection_summary(summary_rows)
    tasks = selected_task_specs(task_suite)

    write_csv(output_path(output_prefix, "benchmark"), rows)
    write_csv(output_path(output_prefix, "summary"), summary_rows)
    write_csv(output_path(output_prefix, "memory_advantage"), advantage_rows)
    write_csv(output_path(output_prefix, "temporal_sparsity_summary"), temporal_summary_rows)
    write_csv(output_path(output_prefix, "temporal_sparsity_advantage"), temporal_advantage_rows)
    write_csv(output_path(output_prefix, "lin_sequence_summary"), lin_rows)
    write_csv(output_path(output_prefix, "lin_sequence_injection"), lin_injection_rows)

    for obs_mode in obs_modes:
        heatmap(rows, obs_mode=obs_mode, temporal_mode="temporal_full", metric="action_accuracy", path=FIGURES / f"{output_prefix}_{obs_mode}_accuracy.svg", tasks=tasks)
        heatmap(rows, obs_mode=obs_mode, temporal_mode="temporal_full", metric="greedy_success", path=FIGURES / f"{output_prefix}_{obs_mode}_success.svg", tasks=tasks)
    if "gaussian_sensory" in obs_modes and len(action_inputs) > 1:
        action_ablation_plot(summary_rows, FIGURES / f"{output_prefix}_action_ablation.svg")
    memory_advantage_plot(advantage_rows, output_path(output_prefix, "memory_advantage_svg"))
    temporal_reservoir_plot(summary_rows, output_path(output_prefix, "temporal_sparsity_reservoirs_svg"))
    lin_sequence_grid_plot(summary_rows, output_path(output_prefix, "lin_sequence_grid_svg"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-modes", required=True)
    parser.add_argument("--task-suite", default="updated_navigation", choices=("legacy", "updated_navigation"))
    parser.add_argument("--action-input", default="one_hot")
    parser.add_argument("--action-embedding-dim", type=int, default=ACTION_EMBEDDING_DIM)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--temporal-modes", default=",".join(TEMPORAL_MODES))
    args = parser.parse_args()

    obs_modes = parse_csv_arg(args.obs_modes, ("gaussian_sensory", "vector", "sparse_aliased", "egocentric_landmark", "hybrid"))
    temporal_modes = parse_csv_arg(args.temporal_modes, TEMPORAL_MODES)
    action_inputs = parse_csv_arg(args.action_input, ACTION_INPUT_MODES, allow_all=True)
    benchmark_path = output_path(args.output_prefix, "benchmark")
    existing_rows = read_csv(benchmark_path)
    non_lin_rows = [row for row in existing_rows if not is_lin_block(row)]
    lin_rows = collect_lin_block_rows(
        task_suite=args.task_suite,
        obs_modes=obs_modes,
        temporal_modes=temporal_modes,
        action_inputs=action_inputs,
        action_embedding_dim=args.action_embedding_dim,
    )
    merged_rows = non_lin_rows + lin_rows
    write_merged_outputs(
        output_prefix=args.output_prefix,
        task_suite=args.task_suite,
        obs_modes=obs_modes,
        action_inputs=action_inputs,
        rows=merged_rows,
    )
    print(f"merged rows: {len(merged_rows)}")
    print(f"lin_block rows: {len(lin_rows)}")


if __name__ == "__main__":
    main()
