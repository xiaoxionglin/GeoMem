---
status: draft
priority: high
confidence: medium
depends_on: []
evaluated: true
last_reviewed: 2026-05-20
---

# Core Thesis

Navigation memory demand depends on task geometry, observations, and goals.

The project asks which action-observation histories can be compressed into local metric state and which residual route, cue, branch, or latent-context distinctions must remain available for control. Hippocampal sequence dynamics are not universally optimal memory mechanisms. They should become more useful when sparse or aliased observations leave behaviorally relevant history distinctions unresolved by local metric state. Path-integration-like and vector memories should be favored when local metric structure and available observations already provide a task-sufficient current state. Hybrid hippocampal-entorhinal memory is a plausible interpretation for tasks that mix local metric structure with global route or context dependence.

## Key Mechanism: History Compression

Navigation and abstract reasoning unfold as action histories. A memory system is efficient when it preserves only the history variables that remain task-relevant.

- In locally integrable and well-observed settings, many action histories collapse into equivalent task state. History can be compressed into position, displacement, heading, or another metric variable.
- In route-dependent, bottlenecked, hierarchical, or aliased settings, local summaries can leave behaviorally relevant residual history distinctions unresolved. Sequence, graph, clone-state, belief-like, or stack-like memory may be appropriate for those distinctions.
- In mixed settings, local history can be compressed while route or latent-context state remains explicit.

Action histories play a specific role in this thesis: they expose the geometry of control paths and the path equivalences induced by the task. They are not a substitute for sensory input in realistic navigation. A path-integration-like representation is one possible compression of action and self-motion history into local metric state; a visual, symbolic, noisy, aliased, or DG-sparsified observation channel supplies separate evidence about latent state.

## Primary Experimental Axes

Use a small set of operational variables:

- Input sparsity: how much reliable sensory evidence is available at each step.
- Action commutativity: whether action strings such as AB and BA lead to equivalent states.
- Sensory aliasing: whether the same observation corresponds to multiple latent states.
- Loop closure reliability: whether returning paths close accurately under local metric integration.
- Bottleneck and branching structure: whether future choices depend on route history.

Graph hyperbolicity, local dimensionality, and curvature can be used as secondary analytic measures.

The geometry diagnostics should broaden from exact trace commutativity to measures usable on transition graphs and simulators: local action reorderability, the ambiguity left after a coordinate-like path summary, and the policy-relevant route residual after that summary is available.

## Central Prediction

The advantage of sequence generators should increase when sparse or aliased observations preserve policy-relevant route or context distinctions after local metric compression. Vector/path-integration systems should be favored when observations are dense, local metric structure is reliable, and the task quotient makes action history compressible.

A stronger portfolio hypothesis is separate: under a fixed resource budget and a varied task distribution spanning locally metric, route-dependent, and mixed environments, a hybrid memory may minimize cross-task regret even when specialized memories win individual regimes. That hypothesis requires an explicit task distribution and cost model.
