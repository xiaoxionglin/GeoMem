# Figures

Generated overview plots for the current GeoMem commutativity probe.

- `commutativity_bar.svg`: bar overview for action-sequence length 4.
- `commutativity_by_length.svg`: valid-only sensitivity of commutativity to action-sequence length.
- `commutativity_heatmap.svg`: compact valid-only environment-by-length overview.
- `formal_environment_gallery.svg`: formal geometry prototypes, including trace quotients from ordered histories to action counts.
- `navigation_environment_gallery.svg`: latent graphs for the named behavior-facing navigation tasks.
- `room_graph_environment_gallery.svg`: local metric rooms under the factorized global room topologies.
- `all_environment_atlas.svg`: complete atlas of all default `geomem-env` environments with complexity and residual-history badges.
- `current_commutativity_results.csv`: source values used by the plots.
- `pairwise_commutator_matrices.csv`: action-pair commutator scores.
- `commutator_matrix_*_raw.svg`: pairwise raw commutator matrix for each environment.
- `commutator_matrix_*_valid.svg`: pairwise boundary-filtered commutator matrix for each environment.
- `observation_ambiguity.csv`: latent-state ambiguity under different observation modes.
- `observation_ambiguity_heatmap.svg`: normalized candidate-set size by trace environment and observation mode.
- `observation_information_heatmap.svg`: fraction of latent-state entropy retained by each observation mode.
- `transition_ambiguity.csv`: non-Markov transition ambiguity for each observation mode.
- `transition_ambiguity_heatmap.svg`: weighted fraction of observation-action pairs with multiple latent next states.
- `policy_ambiguity.csv`: goal-conditioned shortest-path policy ambiguity for each observation mode.
- `policy_ambiguity_heatmap.svg`: fraction of states whose observation aliases different optimal policies.
- `supervised_policy_benchmark.csv`: supervised shortest-path policy benchmark results.
- `supervised_policy_*_*.svg`: lookup and decision-tree policy benchmark heatmaps.
- `lightweight_empirical_benchmark.csv`: multi-goal lightweight benchmark with sparsity, memory-capacity, and decision-tree results.
- `lightweight_sparsity_commutativity.svg`: memoryless observation accuracy across commutativity and input degradation.
- `lightweight_multigoal_robustness.svg`: multi-goal lookup robustness for representative architectures.
- `lightweight_architecture_comparison.svg`: decision-tree architecture comparison across trace geometries.
- `lightweight_memory_capacity.svg`: finite-history capacity curve.
- `architecture_selection.csv`: cost-aware cheapest-sufficient architecture analysis.
- `architecture_selection_summary.md`: readable table of selected architectures by threshold.
- `architecture_selection.svg`: figure showing memory cost of the cheapest sufficient architecture.
- `rnn_navigation_benchmark.csv`: supervised goal-conditioned RNN/reservoir navigation benchmark.
- `rnn_navigation_summary.csv`: mean and SEM summary of the RNN/reservoir navigation benchmark.
- `rnn_memory_advantage.csv`: best recurrent or structured-sequence model minus best memoryless model by task and observation regime.
- `rnn_temporal_sparsity_summary.csv`: temporal input sparsity summary by observation mode, temporal mode, and model.
- `rnn_temporal_sparsity_advantage.csv`: recurrent advantage summary under full, dropout, and cue-then-blank temporal input.
- `rnn_lin_sequence_summary.csv`: Lin-style sequence-generator summaries with explicit `R`, `L`, `ell`, input injection, and behavioral context readout.
- `rnn_lin_sequence_injection.csv`: front-band vs all-unit injection comparison for the `R=3`, `L=4` reference sequence.
- `rnn_lin_long_sequence_benchmark.csv`: long delayed-cue supervised architecture benchmark with explicit history, prior recurrent controls, shared Lin buffers, and independent block-nilpotent Lin sequence chains.
- `rnn_lin_long_sequence_summary.csv`: detailed long delayed-cue summary across every compared architecture and long Lin block configuration.
- `rnn_lin_long_architecture_families.csv`: best long delayed-cue model per architecture family and observation regime.
- `rnn_weight_analysis.csv`: recurrent matrix metrics for trained RNNs and fixed reservoirs in the navigation benchmark.
- `rnn_navigation_*_accuracy.svg`: action-accuracy heatmaps by observation regime.
- `rnn_navigation_*_success.svg`: greedy rollout-success heatmaps by observation regime.
- `rnn_memory_advantage.svg`: direct visualization of recurrent memory advantage in sparse and hybrid observation regimes.
- `rnn_temporal_sparsity_reservoirs.svg`: sparse/aliased reservoir accuracy across temporal input regimes.
- `rnn_lin_sequence_grid.svg`: sparse cue-then-blank behavioral performance across Lin-style `R x L` sequence configurations.
- `rnn_lin_long_sequence_grid.svg`: long delayed-cue performance for independent Lin blocks across `R x L`.
- `rnn_lin_long_architecture_comparison.svg`: long delayed-cue heatmap comparing previous memory architectures against explicit history and independent Lin blocks.
- `reservoir_matrix_analysis.csv`: standalone metrics for fixed reservoir matrix families.
- `reservoir_matrix_analysis.svg`: visual comparison of reservoir matrix diagnostics.
- `systematic_formal_sweep.csv`: repeated trace-geometry sweep over alphabet size, depth, and commutation relation graphs.
- `systematic_formal_architectures.csv`: finite-memory architecture readouts and cheapest sufficient model in the formal trace sweep.
- `systematic_formal_sweep.svg`: consolidated trace sweep linking commutativity to ambiguity and memory cost.
- `systematic_navigation_sweep.csv`: factorized room-graph navigation architecture runs.
- `systematic_navigation_summary.csv`: decision-level room-graph navigation summary by topology, aliasing, and cue visibility.
- `systematic_navigation_sweep.svg`: primary room-graph architecture heatmap for aliased policy decisions.
- `delay_span_sweep.csv`: delayed T-maze runs over task delay and memory span.
- `delay_span_summary.csv`: delay-span summary with dense visible-state ceiling and sparse cue-gap memory controls.
- `delay_span_sweep.svg`: delay-by-span heatmaps for explicit history and independent Lin blocks.
- `temporal_basis_delayed_cue.csv`: nilpotent and oscillatory basis benchmark for delayed sparse cue recall with distractors.
- `temporal_basis_phase_task.csv`: phase-reconstruction benchmark for nilpotent and oscillatory temporal bases.
- `temporal_basis_forgetting_curves.svg`: analytic exponential damping versus nilpotent hard cutoff.
- `temporal_basis_lag_similarity.svg`: lag-code similarity matrices for nilpotent slots and oscillatory phase codes.
- `temporal_basis_delayed_cue_accuracy.svg`: delayed cue accuracy by temporal basis and delay.
- `temporal_basis_phase_task.svg`: modulo-phase reconstruction by temporal basis.

The current figures separate raw finite-task effects from valid-only local algebra. This matters because bounded lattices are not fully commutative under clamped boundary dynamics, even though the corresponding torus lattice is exactly commutative.

Regenerate from the project root with:

```bash
python3 experiments/plot_current_results.py
PYTHONPATH=../geomem-env/src:experiments python3 experiments/environment_gallery.py
```
