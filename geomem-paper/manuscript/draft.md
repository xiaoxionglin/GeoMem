# GeoMem: When Are Hippocampal Sequence Generators Useful?

## Abstract

Navigation requires memory, but no single memory architecture is optimal across all environments. We propose that memory architecture should be matched to task geometry and input statistics. Hippocampal sequence dynamics are advantageous when sparse observations and noncommutative task geometry prevent compression of action history into a low-dimensional metric state. In contrast, path-integration and vector memory should dominate when local metric structure is reliable and action histories commute. This framing extends recent work showing that hippocampus-inspired sequence generators support egocentric visual navigation after DG-like sparsification of the sensory stream. We propose that this advantage should be modulated by task geometry: sequence generators should dominate in aliased, bottlenecked, tree-like, or graph-like environments, whereas vector/path-integration systems should dominate in dense, locally Euclidean, high-commutativity environments. The resulting framework explains why path integration, cognitive graphs, predictive maps, structural generalization, and sequence-centric hippocampal theories each capture part of the problem, while predicting when each should fail.

## 1. Introduction

The hippocampal-entorhinal system is implicated in spatial navigation, episodic memory, replay, relational inference, and abstract structure learning. This breadth has produced several successful but partially competing theories. Path-integration and grid-cell theories emphasize compact metric updating. Cognitive graph accounts emphasize route structure and local metric labels. Successor representation models emphasize predictive state occupancy. The Tolman-Eichenbaum Machine emphasizes structural generalization. Sequence-centric theories propose that spatial maps can emerge from latent higher-order sequence learning.

We argue that these theories are best understood as describing different regimes of a common computational problem: how much of the past must be preserved to act well now? In some environments, action history can be compressed into position, heading, displacement, or a low-dimensional vector state. In others, current input is ambiguous and the order of previous actions remains necessary. The relevant question is therefore not whether navigation is fundamentally metric or sequential, but when trajectory history can be compressed without losing task-relevant information.

Our starting point is recent work from Lin, Yiu, and Leibold showing that a minimal hippocampus-inspired sequence generator can support actor-critic egocentric visual navigation after a DG-like sparsifying front end. GeoMem extends this result by asking when that advantage should appear. We propose that input sparsity, DG-style code sparsification, and task geometry are separate axes that jointly determine the useful memory architecture.

## 2. Core Hypothesis

The core hypothesis is a two-dimensional phase space:

```text
                          Input density
                    sparse             dense
geometry
metric              sequence may help   path integration/vector state
tree-like           sequence            sequence or graph
mixed               hybrid              hybrid or graph
```

This table is not intended as a final taxonomy. It is a testable scaffold. The central prediction is that sequence generators should provide the greatest advantage when observations are sparse and the task is noncommutative, aliased, bottlenecked, or graph-like.

## 3. Task Geometry As History Compressibility

Task geometry determines whether histories can be compressed. In locally Euclidean settings, many action sequences are equivalent. Walking north then east can often be treated as equivalent to walking east then north, up to small local metric errors. In such settings, a coordinate-like state or vector displacement is an efficient summary.

In tree-like or route-dependent settings, action order matters. Entering a corridor from one branch and then taking a turn is not equivalent to taking those actions in the opposite order. In aliased settings, the same observation can correspond to different latent states depending on history. In these regimes, compressing away order destroys information required for action.

We therefore use effective commutativity as a practical measure. Sample pairs of action strings with the same action multiset and ask whether they lead to the same state, nearby states, or task-equivalent states. High commutativity predicts that vector or path-integration memory should be efficient. Low commutativity predicts that sequence, graph, clone-state, or stack-like memory should be useful.

A first implementation scaffold is provided in `../../geomem-experiments/experiments/commutativity_probe.py`. It now separates ideal local commutativity from finite-task constraints by comparing a torus lattice, a clamped lattice, bottleneck rooms, trees, and shortcut-augmented trees. The torus lattice is exactly commutative, while the clamped lattice is not fully commutative in raw mode because boundary clamping makes inverse action orders differ. Boundary-filtered scores recover local commutativity for lattice-like environments but remain low for trees.

The shortcut-tree diagnostic is also informative: adding lateral same-depth actions does not automatically make a tree more commutative, because those new actions still fail to commute with up/down/child moves. A stronger interpolation should add explicit path equivalences, for example through a quotient graph or trace-monoid environment.

## 4. Relation To Existing Theories

Path integration explains how an agent can update position from self-motion cues and is efficient when local metric structure is reliable. Cognitive graph theories explain how navigators can use route networks and local metric labels without requiring a globally consistent Euclidean map. Successor representation explains predictive occupancy under a policy. TEM explains structural generalization across domains. Clone-structured cognitive graphs and latent-sequence theories explain how hidden state and spatial structure can emerge from higher-order sequence learning.

GeoMem is complementary to these accounts. It does not replace their representational objectives. Instead, it asks which memory substrate should be favored under which task geometry and input regime.

## 5. Leibold Lab Extension

This project is a direct extension of hippocampal sequence work in the Leibold lab. Prior work has modeled biological constraints on sequence memory, temporal compression, intrinsic sequence reservoirs, theta correlations, MEC influences on replay, sparse place coding constraints, and grid-field deformation by environmental geometry.

The immediate technical predecessor is Lin, Yiu, and Leibold 2026. That work shows that a hippocampus-inspired sequence generator can act as a temporal buffer after DG-like sparsification and can produce spatial representations during navigation. GeoMem asks whether the advantage of that architecture is predictable from task geometry. The proposed extension is:

> from sparse inputs favor hippocampal sequence generators to task geometry, intrinsic observation sparsity, and DG-like sparsification jointly determine optimal memory architecture.

## 6. Pilot Geometry Probe

Before comparing learning agents, we need to make "task geometry" measurable. We therefore implemented a small diagnostic probe that estimates effective commutativity: the probability that reordering two actions in a trajectory preserves the final state. The probe reports both raw commutativity, which includes boundaries and invalid moves as part of the finite task, and valid-only commutativity, which filters out trajectories that hit invalid transitions.

The current results are:

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

The torus lattice is the positive control and is exactly commutative. The clamped lattice is not fully commutative in raw mode because boundary clamping makes inverse action orders differ, but valid-only filtering recovers perfect local commutativity. Bottleneck rooms behave similarly: their local action algebra is metric, but walls and doors create finite-task constraints. Trees remain low under valid-only filtering, as expected.

The shortcut-tree result is important because it is not a clean interpolation from tree to lattice. Adding lateral same-depth actions does not by itself create path equivalence. These actions still fail to commute with up/down/child actions. This negative result sharpens the formal model: if the theory is about when history can be compressed, the environment must manipulate action equivalence directly, not merely add graph shortcuts.

We therefore implemented a second formal probe based on trace-monoid action histories. States are canonicalized action strings under selected commutation relations. With no commuting action pairs, ordered histories remain distinct. With all action pairs commuting, histories collapse to action counts. With partial commutation, some dimensions of history compress while others remain ordered. This gives the intended interpolation:

```text
environment,raw_commutativity,valid_only_commutativity
trace_commute_000,0.790,0.000
trace_commute_033,0.771,0.264
trace_commute_067,0.716,0.534
trace_commute_100,0.722,1.000
```

The valid-only score increases monotonically with the number of commuting action pairs. This is the cleanest current operationalization of history compressibility. It also clarifies why raw finite-task scores should be interpreted cautiously: finite depth creates boundary/self-loop effects, whereas valid-only scores isolate the local algebra of action order.

We then added an observation diagnostic to connect geometry to input density. In trace environments, a count-like observation reports only how often each action has occurred, ignoring order. This is a coordinate-like summary. It is sufficient in the fully commuting environment, but ambiguous when action order matters:

```text
environment         mean_candidates(counts)   retained_entropy(counts)
trace_commute_000   251.11                    0.447
trace_commute_033    93.94                    0.504
trace_commute_067    23.74                    0.619
trace_commute_100     1.00                    1.000
```

This gives a pre-agent estimate of memory demand. If observations collapse distinct latent histories that remain behaviorally relevant, then a memory system must preserve sequence information. If task geometry makes those histories equivalent, count/vector observations are sufficient.

A stronger diagnostic is transition ambiguity. For each observation-action pair, we ask whether the same observation and same action can lead to multiple latent next states. If so, the observation is not Markov and some memory of hidden history is required to predict future state. Under count observations:

```text
environment         mean_next_candidates(counts)   weighted_ambiguous_fraction(counts)
trace_commute_000   95.55                           0.993
trace_commute_033   41.17                           0.973
trace_commute_067   12.90                           0.873
trace_commute_100    1.00                           0.000
```

Thus, count/vector observations are not merely lossy in noncommutative environments; they are non-Markov. In the fully commuting environment, the same count observation is complete. This is the clearest current bridge from geometry to memory demand.

Finally, we added a goal-conditioned policy ambiguity diagnostic. Each trace environment is treated as a reversible navigation graph, and a fixed goal state is defined by the target action sequence `ABCABCAB`. For each latent state, shortest-path optimal actions to the goal are computed. Policy ambiguity measures whether states sharing the same observation require different optimal action sets. Under count observations:

```text
environment         weighted_policy_ambiguity(counts)
trace_commute_000   0.997
trace_commute_033   0.987
trace_commute_067   0.920
trace_commute_100   0.000
```

This closes the pre-agent loop. Count observations are sufficient for control only when the geometry is fully commutative. When action order matters, the same count observation can alias states that require different actions.

## 7. Formal Definitions

Let latent states be histories of actions modulo a set of equivalence relations. A memory representation is sufficient when it preserves all history distinctions needed to predict task-relevant futures. The current implementation measures three related quantities.

Effective commutativity is the probability that reordering actions preserves latent state:

$$
C = \Pr[f_s(ab) = f_s(ba)]
$$

where `f_s` is the transition from start state `s`. Raw commutativity includes finite task constraints, while valid-only commutativity excludes trajectories that hit invalid transitions.

Observation ambiguity is the number of latent states compatible with the same observation:

$$
A(o) = |\{s : O(s) = o\}|
$$

Transition ambiguity asks whether an observation is Markov:

$$
T(o,a) = |\{s' : \exists s \text{ with } O(s)=o \text{ and } P(s' \mid s,a)>0\}|
$$

If `T(o,a) > 1`, then observation `o` and action `a` do not determine a unique latent future. This is a pre-agent indicator that memory may be required.

Policy ambiguity asks whether aliased states require different optimal action sets under a goal:

$$
P(o) = |\{\pi^*(s) : O(s)=o\}|
$$

where `pi*(s)` is the set of shortest-path optimal actions from latent state `s` to the goal. If `P(o) > 1`, then the observation is insufficient for selecting an optimal action.

## 8. Proposed Experiments

The first full experiment should compare a sequence-generator actor-critic agent, an LSTM actor-critic agent, a vector/path-integration agent, a graph or clone-state agent, and a hybrid agent across controlled environment families.

Environment families should include torus lattices, clamped lattices, bottleneck mazes, trees, quotient or trace-monoid environments, aliased POMDP mazes, and Dyck-like symbolic navigation. Dyck should be treated only as a limiting case for stack memory. Shortcut-augmented trees should be retained as a diagnostic graph family, but not as the main interpolation between tree-like and Euclidean-like geometry.

The next environment family should build on the trace-monoid scaffold by adding rewards. Increasing the number of commuting action pairs should gradually move the task from ordered sequence memory toward count/vector memory. Sparse or aliased observations already reveal whether the agent can infer the compressed state directly or needs sequence memory to predict transitions. Policy ambiguity then asks whether aliased states require different optimal actions.

The current implementation should not yet be interpreted as showing hippocampus-inspired neural-agent failure, but it now goes beyond representational insufficiency. It shows that count/vector observations fail to define a Markov state and can also be insufficient for goal-directed policy selection when action order remains relevant. The remaining empirical gap is not basic control but ecological and biological validation: a navigation-like aliased task, recurrent or hippocampus-inspired agents, and perturbation-style comparisons of sequence memory with path-integration memory.

We implemented this first supervised benchmark using shortest-path policy labels. A table-majority lookup on all states acts as a policy-sufficiency diagnostic, while a decision-tree classifier trained on a random 70% split gives a first finite-data learning test. Under the lookup diagnostic, count observations are sufficient only in the fully commuting trace environment, while finite-history and hybrid keys recover high performance in noncommutative environments:

```text
environment         memoryless_counts   history_counts   hybrid
trace_commute_000   0.517               1.000            1.000
trace_commute_033   0.571               0.990            0.996
trace_commute_067   0.822               1.000            1.000
trace_commute_100   1.000               1.000            1.000
```

The learned decision-tree benchmark follows the same qualitative pattern, although it also reflects finite data coverage and function-generalization limits:

```text
environment         memoryless_counts   history_counts   hybrid
trace_commute_000   0.491               0.993            0.997
trace_commute_033   0.527               0.975            0.986
trace_commute_067   0.753               0.944            0.987
trace_commute_100   0.755               0.857            1.000
```

Thus, the current evidence distinguishes representational sufficiency, policy sufficiency, and learned finite-data performance. The benchmark is not yet a hippocampus-inspired agent model, but it is the first direct architecture comparison.

We then implemented a lightweight multi-goal empirical benchmark to test whether the result was tied to the single target sequence `ABCABCAB`. The benchmark evaluates nine goals per environment, degraded observations, finite memory lengths, lookup sufficiency, and random 70/30 decision-tree performance. Under the all-state lookup diagnostic, the count/vector representation again becomes sufficient only as commutativity increases:

```text
environment         memoryless_counts   history_depth_last_h4   hybrid_h4
trace_commute_000   0.517               1.000                   1.000
trace_commute_033   0.573               0.998                   0.996
trace_commute_067   0.823               0.925                   1.000
trace_commute_100   1.000               0.920                   1.000
```

The lightweight learned benchmark preserves the same broad architecture ranking:

```text
environment         full_state   memoryless_counts   history_counts_h4   hybrid_h4
trace_commute_000   0.998        0.503               0.998               0.998
trace_commute_033   0.985        0.551               0.974               0.988
trace_commute_067   0.968        0.792               0.962               0.978
trace_commute_100   0.943        0.762               0.864               0.968
```

This strengthens the controlled empirical claim: across multiple goals, coordinate/count memory becomes policy-sufficient as action commutativity increases, while finite history and hybrid representations rescue performance when action order remains behaviorally relevant. A caveat is that the `depth_last` observation is unusually strong in low-commutativity shortest-path tasks because it often identifies the route-reversal action. It should therefore be interpreted as a route cue, not as a generic sparse sensory input.

Independent variables should include input density, effective commutativity, sensory aliasing, loop closure reliability, and bottleneck or branching structure. Dependent variables should include sample efficiency, final performance, generalization, shortcut use, detour recovery, robustness to input dropout, memory cost, emergent place fields, remapping, and sequence structure.

The main prediction is that sequence generators win in sparse, noncommutative, aliased, or bottlenecked environments. Vector/path-integration systems win in dense, locally metric, high-commutativity environments. Hybrid systems win in mixed settings.

## 9. Biological Predictions

If the theory is correct, hippocampal sequence-like dynamics should be most behaviorally useful when current observations underdetermine state or when future choices depend on route history. Entorhinal/path-integration signals should be most useful when local metric updates are reliable and sufficient. Mixed environments should produce switching or cooperation between metric and sequence-like codes.

This is a computational interpretation, not a strict anatomical mapping. The claim is not that one region implements one memory type in isolation, but that hippocampal-entorhinal interactions should reflect the geometry and input statistics of the task.

## 10. Discussion

GeoMem reframes navigation and abstract memory as a problem of history compression. Sequence memory and path integration are not rival explanations. They are different solutions to different geometry regimes. Mammalian navigation is hybrid because natural environments are locally metric but globally graph-like, bottlenecked, and often aliased.

The theory should be judged by whether it predicts architecture-dependent performance across controlled tasks. Its first target is computational: explain when hippocampal sequence generators are useful. Its broader implication is that biological and artificial agents should select memory architectures according to task geometry.

## References To Integrate

See `../../geomem-theory/literature/reading-list.md` for the canonical working bibliography. The manuscript should prioritize Lin/Yiu/Leibold 2026, Leibold 2020, Yiu/Leibold 2023, Chenani et al. 2019, Gao/Ganguli 2015, McNaughton et al. 2006, Peer et al. 2021, Chrastil/Warren 2014, Stachenfeld et al. 2017, Whittington et al. 2020, George et al. 2021/2024, and Banino et al. 2018.
