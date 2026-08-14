# Experiment Scaffold

This folder starts the computational implementation of GeoMem.

The first target is intentionally small: construct controlled graph/action environments and compute an effective commutativity index. This makes "task geometry" operational before adding learning agents.

## Current Script

```bash
python3 commutativity_probe.py
```

The script compares simple environment families:

- torus lattice: fully commuting positive control;
- clamped lattice: bounded lattice where boundary clamping breaks raw commutativity;
- tree: low commutativity;
- bottleneck rooms: locally metric but globally graph-like;
- shortcut trees: controlled tree variants with lateral same-depth shortcuts.

The script reports two scores:

- `raw_commutativity`: includes boundary and invalid-action effects as part of task dynamics.
- `valid_only_commutativity`: filters out action-order pairs where either trajectory hits an invalid transition.

Use the raw score to study the full finite task, and the valid-only score to isolate local action algebra.

## Current Plots

Generate overview plots with:

```bash
python3 experiments/plot_current_results.py
```

Outputs:

- `../figures/commutativity_bar.svg`
- `../figures/commutativity_by_length.svg`
- `../figures/commutativity_heatmap.svg`
- `../figures/current_commutativity_results.csv`
- `../figures/pairwise_commutator_matrices.csv`
- `../figures/commutator_matrix_*_raw.svg`
- `../figures/commutator_matrix_*_valid.svg`

Current interpretation:

- `torus_lattice` is exactly commutative in both raw and valid-only modes.
- `clamped_lattice` is not fully commutative in raw mode because boundary clamping makes inverse action orders differ.
- `clamped_lattice` and `bottleneck_rooms` become locally commutative under valid-only filtering, showing that their raw noncommutativity is mostly boundary/door constraint.
- `tree` remains low under valid-only filtering.
- Lateral shortcuts do not automatically make a tree more commutative; they add actions that can still fail to commute with up/down/child moves. This motivates a future quotient or partial-commutation environment where shortcuts create explicit path equivalences.

Next steps:

- add sensory sparsity and aliasing;
- add quotient or trace-monoid environments where shortcut density creates true path equivalence;
- plug in sequence-generator, LSTM, vector, graph, and hybrid agents;
- evaluate sample efficiency, generalization, detours, and memory cost.

## Observation Ambiguity

Run:

```bash
python3 experiments/observation_ambiguity.py
```

This measures how many latent trace states remain possible under different observation modes:

- `full`: full canonical state;
- `counts`: action counts only;
- `depth_last`: sequence depth and last action;
- `depth`: sequence depth only.

## Transition Ambiguity

Run:

```bash
python3 experiments/transition_ambiguity.py
```

This measures whether an observation is Markov: for each observation-action pair, it counts whether multiple latent next states are possible.

## Policy Ambiguity

Run:

```bash
python3 experiments/policy_ambiguity.py
```

This treats each trace environment as a reversible navigation graph, defines a goal state, computes shortest-path optimal actions, and measures whether the same observation aliases states requiring different optimal actions.

## Supervised Policy Benchmark

Run:

```bash
python3 experiments/supervised_policy_benchmark.py
```

This trains table-lookup and decision-tree classifiers on shortest-path policy labels. It compares memoryless count/vector observations, full-state observations, finite observation history, and hybrid count-plus-history representations.

## Lightweight Empirical Benchmark

Run:

```bash
python3 experiments/lightweight_empirical_benchmark.py
```

This extends the supervised benchmark across multiple goals, degraded observations, finite memory lengths, and lightweight decision-tree classifiers. It is the current small-scale empirical test bed before larger recurrent or hippocampus-inspired agent training.

## Architecture Selection

Run:

```bash
python3 experiments/architecture_selection.py
```

This reads `figures/lightweight_empirical_benchmark.csv` and selects the cheapest representation that reaches a target multi-goal policy accuracy. It is the current direct test of the claim that the best memory architecture depends on task geometry.

## RNN And Reservoir Navigation Benchmark

Run:

```bash
python3 experiments/rnn_navigation_benchmark.py
python3 experiments/recurrent_matrix_analysis.py
```

These scripts convert the generated geometries into supervised goal-conditioned navigation tasks, train a lightweight NumPy RNN, compare fixed reservoir matrices with trained readouts, and analyze recurrent matrix structure. The benchmark also includes delayed behavioral choices, lightweight egocentric-landmark observations, and Lin-style fixed sequence generators parameterized by `R`, `L`, and `ell = R + L - 1`. A dedicated long bridge uses independent block-nilpotent input chains with `L` up to `64` on a 72-step delayed-cue task and compares them on the same task against memoryless controls, explicit observed histories, the trained RNN, every fixed reservoir family, and the earlier shared Lin buffers. This is supervised control from shortest-path or directed-choice labels, not full RL.

## Systematic Paper Sweeps

Run:

```bash
python3 experiments/systematic_formal_sweep.py
python3 experiments/systematic_navigation_sweep.py
python3 experiments/delay_span_sweep.py
python3 experiments/temporal_basis_benchmark.py
```

These paper-first sweeps consolidate the episodic diagnostics above. The formal sweep varies trace alphabet size, depth, and repeated commutation relation graphs; the navigation sweep varies room-graph topology, aliasing, and cue visibility against a fixed primary architecture set; the delay-span sweep maps explicit history and independent Lin-block span against delayed sparse cue navigation with a dense visible-state ceiling.

`temporal_basis_benchmark.py` is a theory-control benchmark. It compares nilpotent delay lines, damped oscillatory banks, thresholded oscillatory banks, and a matched phase oscillator on delayed-cue and phase-reconstruction tasks.
