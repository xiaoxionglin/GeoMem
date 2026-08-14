# Diagnostic Transfer Report

This note records how the current `geomem-env` diagnostics transfer from simplified formal environments to richer navigation-like environments.

## Shared Axes

The package now compares all standard environments along the same axes:

- action reorderability: `effective_commutativity()` and `pairwise_commutator_matrix()`;
- observation aliasing: `observation_ambiguity()`;
- Markov sufficiency: `transition_ambiguity()`;
- control sufficiency: `policy_ambiguity()`;
- residual history demand: `residual_history_diagnostics()`;
- memory-architecture expectations: `architecture_expectations()`.

These functions accept arbitrary observation functions, so the same diagnostic logic can be applied to trace states, local room coordinates, sparse symbols, cue-gap observations, and local affordance vectors.

## What Transfers Cleanly

- Observation ambiguity transfers directly from trace monoids to room graphs, repeated-room tasks, cue-gap tasks, and route-choice tasks.
- Transition ambiguity transfers directly whenever the environment has deterministic latent transitions and a candidate observation summary.
- Policy ambiguity transfers directly for finite transition graphs by treating valid moves as reversible navigation edges and comparing shortest-path policy sets.
- Residual history demand transfers as a common summary of whether an observation leaves transition-relevant or policy-relevant latent distinctions.

## What Needs Care

- Effective commutativity is cleanest for action-history and local movement systems. In cue-choice tasks, low commutativity can reflect task protocol actions rather than spatial geometry alone.
- Local view vectors are richer than symbolic sparse observations, but they are still controlled feature summaries, not rendered continuous sensory streams.
- Policy ambiguity depends on the selected goal. The default goal is deterministic and useful for diagnostics, but experiments should still choose paper-specific goals explicitly.

## Current Boundary

The package now bridges simplified and richer environments through shared finite-state diagnostics and higher-dimensional local observation vectors. It does not yet implement continuous rendered observations, physics, or full Gymnasium environment classes with mutable episode state. Those remain future extensions once the finite diagnostic substrate is stable.
