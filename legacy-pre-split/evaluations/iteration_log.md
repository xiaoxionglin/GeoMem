# Iteration Log

## Iteration 1-5: Literature-Grounded Reframing

Decision: shift from sequence-first framing to geometry-dependent memory architecture.

## Iteration 6-10: Compression And Commutativity

Decision: use history compression and effective commutativity as the main bridge between path integration and sequence memory.

## Professor Evaluation: Narrowing

Decision: pitch the paper first as a theory of when hippocampal sequence generators are useful. Treat geometry-dependent memory architecture as the broader implication.

Remaining risk: project may become too broad if all modules are treated equally in the manuscript.

## Iteration 001: Corrected Commutativity Probe

Date: 2026-05-20

Changed modules: `experiments`, `figures`, `manuscript`, `modules/experimental_plan.md`.

Reason: the first commutativity probe confounded theoretical action commutativity with finite-boundary artifacts. Tree and shortcut-tree were also not controlled variants.

Evidence: `torus_lattice` scores 1.000 in raw and valid-only modes; `clamped_lattice` scores below 1.000 raw but returns to 1.000 valid-only; `tree` remains low under valid-only filtering.

Decision: keep raw and valid-only metrics. Treat shortcut trees as a diagnostic, not as the main interpolation. Add quotient or trace-monoid environments next.

Remaining risks: valid-only filtering removes biologically relevant boundary structure if used alone; raw scores are task-relevant but mix local algebra with finite constraints.

## Iteration 002: Trace-Monoid Environment

Date: 2026-05-20

Changed modules: `experiments/commutativity_probe.py`, `figures`, `manuscript`, `modules/experimental_plan.md`.

Reason: shortcut trees did not produce a clean interpolation because adding lateral actions is not equivalent to imposing path equivalence.

Evidence: valid-only commutativity increases monotonically across trace environments: `0.000 -> 0.264 -> 0.534 -> 1.000`.

Decision: use trace/partial-commutation environments as the formal backbone. Keep shortcut trees as a diagnostic and Dyck as a limiting case.

Remaining risks: the trace environment is abstract. The next step must add observations, aliasing, and task rewards to connect it to navigation and agent memory.

## Iteration 003: Observation Ambiguity

Date: 2026-05-20

Changed modules: `experiments/observation_ambiguity.py`, `figures`, `manuscript`, `modules/experimental_plan.md`.

Reason: the theory requires a second axis beyond geometry: input density and sensory aliasing.

Evidence: count-like observations have high latent ambiguity in noncommutative trace environments but become complete in the fully commuting environment.

Decision: use observation ambiguity as a pre-agent memory-demand metric.

Remaining risks: state ambiguity alone is not enough. The next step must measure whether aliased latent states require different actions.

## Iteration 004: Transition Ambiguity

Date: 2026-05-20

Changed modules: `experiments/transition_ambiguity.py`, `figures`, `manuscript`, `modules/experimental_plan.md`.

Reason: observation ambiguity shows latent aliasing, but transition ambiguity tests whether an observation is Markov.

Evidence: count observations have weighted transition ambiguity near 1.0 in noncommutative trace environments and 0.0 in the fully commutative environment.

Decision: use transition ambiguity as the main pre-agent memory-demand metric.

Remaining risks: transition ambiguity still does not prove control failure. The next step is policy ambiguity under goals or rewards.

## Synthesis 001: Current Results

Date: 2026-05-20

Decision at that stage: the evidence supported representational and transition-level memory demand, but not yet learned control differences. Later iterations added policy, supervised, multi-goal, and architecture-selection diagnostics.

Supported claim: count/vector memory is sufficient when task geometry makes action order irrelevant; otherwise, compressed observations can alias distinct latent states and transition futures.

Missing claim: whether aliased states require different optimal actions under a concrete reward or goal distribution.

## Iteration 005: Goal-Conditioned Policy Ambiguity

Date: 2026-05-20

Changed modules: `experiments/policy_ambiguity.py`, `figures`, `manuscript`, `modules/experimental_plan.md`.

Reason: transition ambiguity does not alone prove that memory matters for control.

Evidence: count observations have weighted policy ambiguity near 1.0 in noncommutative trace environments and 0.0 in the fully commutative trace environment.

Decision: use policy ambiguity as the strongest pre-agent memory-demand metric.

Remaining risks: this is still a graph-search diagnostic, not trained neural-agent performance.

## Iteration 006: Supervised Policy Benchmark

Date: 2026-05-20

Changed modules: `experiments/supervised_policy_benchmark.py`, `figures`, `manuscript`, `modules/experimental_plan.md`.

Reason: the paper needed a learned architecture comparison, not only ambiguity diagnostics.

Evidence: finite-history and hybrid representations outperform memoryless count/vector representations in noncommutative trace environments. Count lookup is sufficient in the fully commuting environment.

Decision: include this as a preliminary model-comparison result. Distinguish sufficiency diagnostics from finite-data learned performance.

Remaining risks: the benchmark is still symbolic/supervised and not yet a hippocampus-inspired actor-critic agent.

## Iteration 007: Lightweight Empirical Benchmark

Date: 2026-05-20

Changed modules: `experiments/lightweight_empirical_benchmark.py`, `figures`, `reports`, `experiments/README.md`, `figures/README.md`.

Reason: the paper needed lightweight empirical coverage of multi-goal robustness, input degradation, finite memory capacity, and simple learned performance before larger-scale recurrent training.

Evidence: across nine goals per environment, memoryless count accuracy increases with commutativity: `0.517 -> 0.573 -> 0.823 -> 1.000`. Hybrid count-plus-history stays near ceiling across environments.

Decision: use this as the current small-scale empirical benchmark. Present it as theory/diagnostic support, not final neural-agent evidence.

Remaining risks: `depth_last` is artificially strong in low-commutativity shortest-path tasks because it often identifies the route-reversal action. A later navigation-like task should test sensory sparsity without this cue.

## Iteration 008: Architecture Selection And Paper Completion

Date: 2026-05-21

Changed modules: `experiments/architecture_selection.py`, `modules/memory_architecture.md`, `modules/biological_predictions.md`, `manuscript/paper_draft.md`, `figures`, `reports`.

Reason: the paper needed an explicit memory-architecture theory, a cost-aware architecture result, biological predictions, and a structured manuscript draft.

Evidence: the cheapest sufficient representation at the 0.95 threshold shifts from history/hybrid memory in noncommutative environments to memoryless counts in the fully commutative environment.

Decision: frame the current paper as a theory/diagnostic paper about architecture selection. Do not claim final biological-agent superiority until recurrent or hippocampus-inspired training is added.

Remaining risks: trace tasks remain abstract, and the `depth_last` cue can act as a route-reversal shortcut. The next empirical extension should use a navigation-like aliased maze.

## Iteration 009: RNN And Reservoir Navigation Benchmark

Date: 2026-05-21

Changed modules: `experiments/rnn_navigation_benchmark.py`, `experiments/recurrent_matrix_analysis.py`, `figures`, `manuscript/paper_draft.md`, `reports`.

Reason: the paper needed a learned recurrent-control benchmark and more navigation-like aliased tasks.

Evidence: under sparse/aliased observations, mean action accuracy improved from `0.641` for the memoryless linear policy and `0.552` for the memoryless MLP to `0.736` for the NumPy RNN. Fixed reservoirs also improved over memoryless baselines on specific tasks, with diagonal reservoirs reaching `0.678` mean sparse/aliased accuracy. The largest sparse/hybrid recurrent advantages occurred in `trace_commute_100`, `aliased_loop_rooms`, `aliased_rooms`, `trace_commute_067`, and `torus_lattice`.

Update: temporal sparsity is now an explicit benchmark axis. Under sparse/aliased observations, RNN accuracy remains above memoryless linear under temporal dropout (`0.664` vs `0.599`) and cue-then-blank (`0.635` vs `0.531`). Nilpotent reservoirs gain selected finite-delay wins, but diagonal reservoirs remain the strongest average fixed reservoir under sparse/aliased temporal degradation.

Decision: include this as supervised learned-control evidence. It supports the memory-architecture claim but should not be described as full RL.

Remaining risks: greedy rollout is noisier than action accuracy because rollouts create off-distribution histories. Full actor-critic training remains future work.

## Iteration 010: Supervised Lin Bridge

Date: 2026-05-21

Changed modules: `experiments/rnn_navigation_benchmark.py`, `figures`, `manuscript/paper_draft.md`, `reports`.

Reason: the supervised benchmark needed richer behavioral examples and explicit Lin-style sequence parameters before moving to RL.

Evidence: the benchmark now reports `R`, `L`, and `ell = R + L - 1` for structured sequence readouts. On sparse cue-then-blank behavioral tasks, the `L=8` Lin-style variants lead the `R x L` grid (`0.772` to `0.784` mean action accuracy) and solve the delayed T-maze, landmark-gap detour, and shortcut-route choice better than the best memoryless baselines.

Decision: use this as the supervised bridge to Lin/Yiu/Leibold 2026. Keep full actor-critic egocentric visual navigation and a direct trained sequence-generator reproduction as the next stage.

Remaining risks: the new egocentric-landmark features are lightweight symbolic proxies, behavioral choice datasets are small, and fixed sequence readouts are not the full trained Lin architecture.

## Iteration 011: Long Independent Lin Blocks

Date: 2026-05-21

Changed modules: `experiments/rnn_navigation_benchmark.py`, `figures`, `manuscript/paper_draft.md`, `reports`.

Reason: the first Lin bridge used short `L <= 8` mixed buffers, whereas the long Lin regime requires independent sequence chains and an environment where long delay matters.

Evidence: the dedicated 72-step sparse delayed-cue bridge tests independent block-nilpotent chains with `L in {4,16,64}`. Memoryless linear, NumPy RNN, and all short `L=4/16` block variants stay at `0.506` sparse cue-then-blank action accuracy, while `L=64` reaches `0.694` for `R=1`, `0.714` for `R=3`, and `0.729` for `R=5`.

Decision: treat the long independent-block result as the faithful supervised sequence-span bridge. Keep actor-critic visual navigation as the remaining direct Lin comparison.

Remaining risks: the long bridge is still supervised and uses constructed delayed cues rather than full visual sparse actor-critic exploration.

## Iteration 012: Long Architecture Comparison

Date: 2026-05-21

Changed modules: `experiments/rnn_navigation_benchmark.py`, `figures`, `manuscript/paper_draft.md`, `reports`.

Reason: the first long bridge did not compare its independent Lin blocks against the previous fixed reservoirs, short shared Lin buffers, memoryless MLP, or explicit history controls on the same delayed task.

Evidence: in sparse cue-then-blank long navigation, memoryless linear, memoryless MLP, NumPy RNN, and the best previous shared Lin buffer stay at `0.506`; the best 32-unit fixed reservoir reaches `0.505`. Explicit history reveals the horizon boundary: `H4/H16` stay at `0.506`, `H64` reaches `0.690`, and `H80` reaches `1.000`. Independent `L=64` blocks reach `0.694`, `0.714`, and `0.729` for `R=1`, `R=3`, and `R=5`.

Decision: use the long architecture comparison as the current supervised horizon control. Interpret the independent block result relative to explicit finite history, not as a blanket win over arbitrary capacity-matched recurrent agents.

Remaining risks: the old recurrent controls are 32-unit lightweight baselines; a direct trained Lin-style actor-critic or capacity/horizon-matched learned recurrent control remains future work.

## Iteration 013: Systematic Paper Sweeps

Date: 2026-05-22

Changed modules: `experiments/systematic_formal_sweep.py`, `experiments/systematic_navigation_sweep.py`, `experiments/delay_span_sweep.py`, `figures`, `manuscript/paper_draft.md`, `reports`.

Reason: the trace diagnostics, supervised room examples, reservoir controls, and Lin bridge had become a sequence of targeted experiments instead of a small set of main factorial figures.

Evidence: the formal sweep repeats trace commutation across `K in {3,4}`, depths `4/6`, and relation-graph replicates; the room-graph sweep measures topology by aliasing by cue visibility on held-out policy decisions; the delay-span sweep gives an explicit `D x H/L` boundary with a dense visible-state ceiling.

Decision: use these three sweeps as the paper-facing evidence hierarchy. Keep older named behavioral tasks and exhaustive reservoir tables as supporting controls.

Remaining risks: the room-graph sweep is still supervised and symbolic. Actor-critic learning, continuous visual input, and broader held-out graph generalization remain future model-paper work.
