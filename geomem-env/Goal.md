# Goal

The broader GeoMem project plan lives in `../geomem-paper/`. This package focuses on the environment side of that project.

`geomem-env` should define a shared taxonomy of task environments and geometry diagnostics for studying when action-observation history can be compressed into compact state, and when residual route, cue, branch, or latent-context information must remain explicit for control.

The package's role is to put diverse environments into a common interface so the rest of GeoMem can compare memory architectures under matched task assumptions. In the current project, "memory architecture" mainly means reservoir, sequence, vector/count, finite-history, and hybrid representations; the goal should remain general enough to support richer agents later.

## Responsibilities

- Provide reusable environment primitives: states, actions, transition maps, validity, rollout, and task metadata.
- Implement controlled environment families spanning locally metric, tree-like, bottlenecked, shortcut, trace-monoid, aliased, and mixed local/global structure.
- Extend those families toward task structures that matter in realistic navigation and cognition, while preserving enough control to isolate geometry, observation, and goal effects.
- Expose diagnostics that quantify history compressibility, including effective commutativity, observation ambiguity, transition ambiguity, and policy-relevant residual history.
- Develop measures that place richer, more realistic environments and simplified diagnostic environments on the same comparative axes.
- Keep environment definitions independent of model and experiment code so `geomem-models` and `geomem-experiments` can depend on this package without circular coupling.

## Non-Goals

- Do not train models or own benchmark scripts here.
- Do not own generated figures, CSVs, or paper-facing artifacts.
- Do not encode conclusions about which memory architecture is best; this package supplies the environments and measurements used to test those claims.

## Near-Term Direction

- Consolidate duplicated environment constructors currently living in experiment scripts into stable package APIs.
- Add enough metadata to compare environment families along the paper's main axes: commutativity, aliasing, bottlenecks, branching, loop closure, and cue visibility.
- Identify which simplified diagnostics remain meaningful when applied to richer environment families, and where new measures are needed.
- Keep backward-compatible names where possible so existing experiments can migrate incrementally.

## Structured Plan

### Phase 1: Stabilize the Core Interface

Define the minimal environment contract that all task families must satisfy: state space, action set, transition function, validity mask, rollout behavior, start/goal conventions, and optional metadata. Keep the interface small enough for formal toy environments, but expressive enough for navigation tasks with observations, goals, and partial observability.

Deliverables:

- A stable `Environment` API and constructor conventions, with gymnasium standard.
- Shared helpers for transition validation, rollouts, graph extraction, and environment summaries.
- Compatibility shims for current experiment scripts that still import legacy constructors.

### Phase 2: Build the Baseline Taxonomy

Organize environment families by the geometry and memory pressure they are meant to isolate. The first taxonomy should cover locally metric spaces, tree-like structures, bottleneck and room graphs, trace-monoid action-history spaces, aliased observation tasks, cue-gap tasks, and mixed local/global navigation tasks.

Deliverables:

- A documented registry of environment families and variants.
- Metadata fields for the main comparison axes: commutativity, aliasing, bottlenecks, branching, loop closure, cue visibility, and goal dependence.
- Small default parameter sets for fast diagnostics and larger parameter sets for experiments.

### Phase 3: Unify Diagnostics

Turn the current paper diagnostics into reusable package functions. Diagnostics should measure not only raw graph properties, but whether a proposed state summary is sufficient for transition prediction and policy selection.

Deliverables:

- Effective commutativity and pairwise action reorderability.
- Observation ambiguity and transition ambiguity.
- Goal-conditioned policy ambiguity.
- Residual history measures after coordinate-like, count-like, finite-history, or hybrid summaries.

### Phase 4: Bridge Simplified and Realistic Environments

Extend the taxonomy toward richer environments without losing comparability. The goal is not immediate realism for its own sake, but controlled increases in sensory richness, geometry complexity, and latent context so simplified diagnostics can be tested against more realistic tasks.

Deliverables:

- Richer navigation families with continuous or high-dimensional observations where possible.
- Measures that map realistic environments onto the same axes as simplified environments.
- A report identifying which diagnostics transfer cleanly, which need modification, and which fail outside symbolic settings.

### Phase 5: Migrate Experiments onto `geomem-env`

Move reusable constructors and diagnostics out of `geomem-experiments` and into this package. Experiments should become consumers of stable environment APIs rather than owners of environment definitions.

Deliverables:

- Replaced duplicated constructors in experiment scripts with imports from `geomem_env`.
- Smoke tests that verify old experiment-facing names still behave the same during migration.
- Clear separation between package-owned environment logic and experiment-owned benchmark logic.

### Phase 6: Validate Against the GeoMem Claim

Use the environment taxonomy to test whether memory architecture choice follows from measurable task properties. The environment package should support this by producing comparable tasks and diagnostics, not by training or selecting models itself.

Deliverables:

- A standard diagnostic table for each environment family.
- Acceptance checks showing when vector/count, finite-history, sequence, reservoir, or hybrid representations should be expected to succeed or fail.
- Paper-facing summaries generated by experiments, with environment definitions remaining in this package.
