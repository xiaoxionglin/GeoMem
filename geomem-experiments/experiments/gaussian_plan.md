# Gaussian Navigation Experiment Refresh

## Summary

- First produce a focused Gaussian-sensory run across the updated non-trace default environments: lattice, rooms, tree/shortcut-tree, room-graph, aliased navigation, cue-gap, and route-choice tasks.
- Add an action-input ablation: no previous action, current one-hot previous action, and deterministic dense previous-action embeddings.
- After the focused run passes, regenerate the broader benchmark outputs so existing figures stay comparable.

## Key Changes

- Update `experiments/rnn_navigation_benchmark.py` so task construction can use `geomem_env.default_environments()` instead of the stale hardcoded subset.
- Add CLI/config switches for:
  - `--obs-modes gaussian_sensory` for focused Stage 1.
  - `--task-suite updated_navigation` for non-trace default environments.
  - `--action-input none|one_hot|embedding|all`.
  - `--output-prefix rnn_gaussian_updated` for non-destructive focused outputs.
- Keep Gaussian observations environment-pure: use `gaussian_sensory_channel()` unchanged, with no valid-action bits or action history inside the channel.
- Implement model-side previous-action handling:
  - `none`: concatenate no previous-action features.
  - `one_hot`: current behavior.
  - `embedding`: deterministic seeded dense vectors per action plus start token, default dimension `8`.
- Include `action_input` and `action_embedding_dim` columns in result and summary CSVs.

## Execution

- Stage 1 focused run:
  - Run Gaussian sensory only.
  - Run all action-input ablations.
  - Write distinct outputs under `figures/`, e.g. `rnn_gaussian_updated_benchmark.csv`, `rnn_gaussian_updated_summary.csv`, and Gaussian/action-ablation SVGs.
- Stage 2 full refresh:
  - Run the existing full RNN benchmark with the updated task suite only after Stage 1 compiles and produces sane summaries.
  - Regenerate canonical `rnn_navigation_*`, `rnn_memory_*`, `rnn_lin_*`, and `rnn_weight_analysis.csv` outputs.

## Test Plan

- Static check: `python3 -B -m py_compile experiments/rnn_navigation_benchmark.py experiments/systematic_navigation_sweep.py`.
- Dataset smoke check: instantiate every updated navigation task with `gaussian_sensory` and all three action-input modes; verify non-empty train/test sets and consistent feature dimensions.
- Focused acceptance checks:
  - CSV contains all updated non-trace default environments.
  - Summary has rows for `none`, `one_hot`, and `embedding`.
  - Gaussian focused SVGs render from the produced summary.
- Full acceptance check: full benchmark completes and canonical CSV/SVG outputs are regenerated without missing rows.

## Assumptions

- Stage 1 excludes trace-monoid environments because the request focuses on the new noisy navigation environments and the full-depth trace defaults are much larger.
- Dense action embeddings are deterministic fixed features, not trainable parameters, to keep the NumPy benchmark lightweight and comparable.
- Existing generated figures may be overwritten only in Stage 2; Stage 1 writes prefixed focused outputs.
