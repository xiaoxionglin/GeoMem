---
status: draft
priority: high
confidence: medium
depends_on:
  - ../00_core_thesis.md
  - formal_model.md
evaluated: true
last_reviewed: 2026-05-20
---

# Experimental Plan Module

## First Experimental Question

Does task geometry modulate the sequence advantage, and is that advantage driven by intrinsically sparse observations or by DG-style sparsification of a dense noisy stream?

## Current Implementation Scaffold

`../experiments/commutativity_probe.py` implements the first measurable geometry variable: effective commutativity. It samples action strings, swaps action order, and measures whether the final state is preserved.

Current sanity-check result:

```text
environment,raw_commutativity,valid_only_commutativity
torus_lattice,1.000,1.000
clamped_lattice,0.786,1.000
bottleneck_rooms,0.632,1.000
tree,0.458,0.330
shortcut_tree_025,0.669,0.394
shortcut_tree_050,0.617,0.366
shortcut_tree_100,0.543,0.353
```

The torus result verifies the positive control. The clamped-lattice result shows why bounded environments are not fully commutative in raw mode. The tree remains low under valid-only filtering. The shortcut-tree variants show that lateral shortcuts alone are not the same as true path equivalence, so the next implementation should add quotient or trace-monoid environments.

## Iteration 001 Evaluation

The corrected probe should be treated as a geometry diagnostic, not as evidence about learned memory architectures. Its main contribution is conceptual hygiene:

- raw commutativity captures the full finite task, including walls, boundaries, and doors;
- valid-only commutativity isolates local action algebra;
- bounded metric environments can look noncommutative because of boundary constraints;
- shortcuts are not equivalent to partial commutation unless they impose path equivalence.

The next missing piece is an environment family where the level of action equivalence is controlled directly.

## Iteration 002 Evaluation

The trace-monoid environment implements that missing family. States are canonicalized action histories under selected commutation relations. Valid-only commutativity now increases monotonically as action pairs are allowed to commute:

```text
trace_commute_000,0.000
trace_commute_033,0.264
trace_commute_067,0.534
trace_commute_100,1.000
```

This is a better formal backbone than shortcut trees for testing the core claim. Shortcut trees manipulate graph topology; trace environments manipulate path equivalence directly.

## Baselines

- Lin/Yiu/Leibold 2026 sequence-generator actor-critic agent.
- LSTM actor-critic baseline.
- Vector/path-integration memory agent.
- Graph-memory or clone-state agent for aliased environments.
- Hybrid sequence plus vector-memory agent.

## Environment Families

- Local Euclidean arenas: high loop closure, high commutativity.
- Lattices: high commutativity with discrete actions.
- Trees: low commutativity, strong history dependence.
- Shortcut-augmented trees: lateral-action diagnostic, not yet a true lattice interpolation.
- Quotient or trace-monoid environments: tunable transition from noncommuting tree-like histories to commuting coordinate-like histories.
- Bottleneck mazes: local metric patches connected by graph structure.
- Aliased POMDP mazes: same observation, different latent state.
- Dyck-like symbolic navigation: stack-memory limiting case only.

## Independent Variables

- Input density: intrinsic sparsity versus dense/noisy sensory input.
- Front-end sparsification: DG-like denoising or decorrelation before sequence memory.
- Effective commutativity index.
- Sensory aliasing level.
- Loop closure reliability.
- Branching and bottleneck structure.

## Dependent Variables

- Sample efficiency.
- Final task performance.
- Generalization to new layouts.
- Shortcut and detour behavior.
- Robustness to input dropout or geometry mismatch.
- Memory cost and model size.
- Emergent place fields, remapping, distance kernels, and sequence structure.

## Main Prediction

Sequence generators should win in sparse, noncommutative, aliased, or bottlenecked environments. Vector/path-integration agents should win in dense, locally metric, high-commutativity environments. Hybrids should dominate mixed settings.

## Broad Geometry Diagnostics

The trace-monoid tasks provide exact control over action-history equivalence. For broader environments, geometry should be measured by a diagnostic family rather than by exporting one trace commutativity scalar unchanged:

- local action reorderability: estimate whether swapping short valid action blocks preserves latent or task state;
- path-compression gap: estimate transition or policy ambiguity after a coordinate-like path summary is provided;
- route residual: estimate the remaining control-relevant history needed after local metric or integrator state is available.

Grid worlds and room graphs can use exact transition graphs. DeepMind Lab-style simulators should use privileged simulator state and task outcomes for the geometry diagnostic, then evaluate sensory emissions separately. Web-agent benchmarks need a task-relevant equivalence relation over browser states or action values before action-history geometry is measured.

The broad diagnostic is a planned contribution, not yet established by the current trace results.

## Cross-Environment Portfolio Hypothesis

The current claim is regime-dependent selection: specialized vector-like, sequence-like, or hybrid memories should be selected according to geometry, observations, goals, and cost. A stronger mammalian claim should be tested separately:

> Under a fixed memory budget and a task distribution that mixes locally metric, route-dependent, and mixed environments, a hybrid memory minimizes regret across the portfolio.

Testing this needs a declared environment distribution, a shared resource budget, architecture-specific costs, and comparison against specialized systems that may remain better on their preferred subsets.

## Next Implementation Target

Implement an observation-emission layer on top of the quotient-string environment:

- sparse observations expose only a subset of state features;
- dense observations expose the full canonical state;
- aliased observations map multiple latent states to the same sensory symbol;
- DG-like emissions sparsify a dense or noisy code before the memory core;
- memory demand can be estimated before training agents by measuring posterior state ambiguity.

This turns the formal geometry into the planned two-dimensional hypothesis space: task geometry by input density.

## Iteration 003 Evaluation

The observation ambiguity diagnostic adds the input-density axis. It shows that count-like observations become sufficient only when the trace geometry is fully commutative:

```text
environment         mean_candidates(counts)   retained_entropy(counts)
trace_commute_000   251.11                    0.447
trace_commute_033    93.94                    0.504
trace_commute_067    23.74                    0.619
trace_commute_100     1.00                    1.000
```

This supports the core claim: coordinate-like summaries are only complete when action order is irrelevant.

Next implementation target: add a goal or reward structure and compute action ambiguity under each observation mode.

## Iteration 004 Evaluation

Transition ambiguity measures whether an observation is Markov. Under count observations:

```text
environment         mean_next_candidates(counts)   weighted_ambiguous_fraction(counts)
trace_commute_000   95.55                           0.993
trace_commute_033   41.17                           0.973
trace_commute_067   12.90                           0.873
trace_commute_100    1.00                           0.000
```

This is the strongest pre-agent evidence so far. Count/vector observations are non-Markov when action order matters, but complete when all actions commute.

Next implementation target: add goals/rewards and compute policy ambiguity.

## Iteration 005 Evaluation

Goal-conditioned policy ambiguity now tests whether aliased observations require different optimal actions. Under count observations:

```text
environment         weighted_policy_ambiguity(counts)
trace_commute_000   0.997
trace_commute_033   0.987
trace_commute_067   0.920
trace_commute_100   0.000
```

This closes the pre-agent diagnostic chain:

- geometry controls commutativity;
- observations control aliasing;
- transition ambiguity shows whether observations are Markov;
- policy ambiguity shows whether aliased states require different actions.

Next implementation target: train simple models to predict shortest-path actions from observation histories.

## Iteration 006 Evaluation

The supervised policy benchmark compares memoryless, finite-history, and hybrid keys on shortest-path action labels.

Policy-sufficiency diagnostic (`all/lookup`, optimal-action accuracy):

```text
environment         memoryless_counts   history_counts   hybrid
trace_commute_000   0.517               1.000            1.000
trace_commute_033   0.571               0.990            0.996
trace_commute_067   0.822               1.000            1.000
trace_commute_100   1.000               1.000            1.000
```

Finite-data learning diagnostic (`random70/decision_tree`, optimal-action accuracy):

```text
environment         memoryless_counts   history_counts   hybrid
trace_commute_000   0.491               0.993            0.997
trace_commute_033   0.527               0.975            0.986
trace_commute_067   0.753               0.944            0.987
trace_commute_100   0.755               0.857            1.000
```

Interpretation: history and hybrid representations rescue performance when count/vector state is insufficient. The decision-tree result should be presented as a preliminary learning benchmark because it also depends on finite data coverage.

## Iteration 007 Evaluation

The lightweight empirical benchmark extends the supervised policy benchmark across multiple goals, degraded observation modes, and finite memory lengths.

Multi-goal policy-sufficiency diagnostic (`all/lookup`, optimal-action accuracy):

```text
environment         memoryless_counts   history_depth_last_h4   hybrid_h4
trace_commute_000   0.517               1.000                   1.000
trace_commute_033   0.573               0.998                   0.996
trace_commute_067   0.823               0.925                   1.000
trace_commute_100   1.000               0.920                   1.000
```

Interpretation: the count/vector result is robust across sampled goals. Count memory becomes sufficient as commutativity increases, while finite history and hybrid memory rescue noncommutative environments.

## Iteration 008 Evaluation

The architecture-selection analysis asks which representation is cheapest while reaching a target policy accuracy.

At the 0.95 threshold:

```text
environment         conservative_counts   cost   hybrid_allowed               cost
trace_commute_000   history_counts_h2      6      hybrid_counts_depth_last_h1  5
trace_commute_033   history_counts_h4     12      hybrid_counts_depth_last_h2  7
trace_commute_067   history_counts_h2      6      hybrid_counts_depth_last_h1  5
trace_commute_100   memoryless_counts      3      memoryless_counts            3
```

Interpretation: this is the current direct test of the memory-architecture claim. Fully commutative geometry selects vector/count memory. Noncommutative geometry selects finite history or hybrid memory.

Next implementation target: add a navigation-like aliased maze where route-reversal cues are unavailable, then compare the same architecture-selection diagnostics against a small recurrent or hippocampus-inspired agent.
