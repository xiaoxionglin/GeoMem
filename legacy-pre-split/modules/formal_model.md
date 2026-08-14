---
status: draft
priority: high
confidence: medium
depends_on:
  - ../00_core_thesis.md
evaluated: true
last_reviewed: 2026-05-22
---

# Formal Model Module

## Default Backbone

Use partial commutation as the formal scaffold. In plain terms, ask whether doing action A then B is equivalent to doing B then A.

- High commutativity: action history can collapse into coordinates or vectors.
- Low commutativity: action order can remain state-relevant, so order-discarding memory must be checked against observations and goals.
- Partial commutativity: hybrid geometry, where some local histories compress and others must remain explicit.

Trace monoids or RAAG-like action spaces can provide the formal language, but the manuscript should foreground the intuition and experiments.

`theoretical_effort.md` carries the broader proof agenda. `formal_theory.md` gives the current deterministic shortest-path statements, and `../manuscript/theory_appendix.md` gives the full proofs. The trace scaffold should instantiate history compression and path quotients, not replace a task-level definition of Markov and policy sufficiency.

## Effective Commutativity Index

For an environment, sample pairs of action strings with the same action multiset. Compare their final states using exact equality, metric distance, or task-equivalence.

High index means order often does not matter. Low index means order often matters.

## Alternative Models

- Dyck language: limiting case for stack memory and hierarchical long-range dependency.
- Controlled graph families: trees, lattices, shortcut-augmented trees, small-world graphs, hyperbolic graphs.
- POMDP / clone-structured graph: limiting case for aliased observations and latent-state inference.

## Implementation Decision

The first simulation should not start with full algebraic machinery. It should implement graph families with measurable commutativity, aliasing, and input sparsity, then interpret those families through partial commutation.

The theory layer should later state which of those measurements witnesses state aliasing, Markov failure, policy failure, or only a learned-model performance difference.
