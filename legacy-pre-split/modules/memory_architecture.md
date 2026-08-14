---
status: draft
priority: high
confidence: medium
depends_on:
  - ../00_core_thesis.md
  - formal_model.md
  - experimental_plan.md
evaluated: true
last_reviewed: 2026-05-22
---

# Memory Architecture Module

## Role

This module turns the project from a statement about representational sufficiency into a navigation-facing theory of memory demand.

The central rule is:

> Select the lowest-cost memory representation that preserves the task-relevant history quotient under the geometry, observation regime, and goal family.

This rule separates **memory content** from **memory mechanism**. The current benchmarks test content-level sufficiency using explicit representations and lightweight cost proxies; later recurrent or hippocampus-inspired agents should test mechanism-level learnability.

## Architecture Classes

| Architecture | Computational content | Favored regime | Failure mode |
|---|---|---|---|
| Vector / path-integration memory | Current coordinate, displacement, action counts, heading, or local metric state | Locally integrable metric structure and observations that expose a task-sufficient current state | Fails when the same local summary aliases residual route or context distinctions with different futures |
| Sequence / recurrent memory | Ordered recent history or recurrent hidden state | Sparse or aliased observations with policy-relevant delayed route, cue, or context dependence | Can waste capacity when order has already collapsed to a task-sufficient local state |
| Graph / clone-state memory | Latent state identity under observation aliasing | Repeated sensory observations, bottlenecks, context-dependent routes | Requires state splitting and can overfit task-specific topology |
| Stack / hierarchical memory | Nested unresolved dependencies | Dyck-like or hierarchical tasks | Too specialized for ordinary metric navigation |
| Hybrid memory | Compressible local dimensions stored as metric state; residual distinctions kept as sequence, graph, clone-state, or belief-like context | Mixed local metric and global route/context regimes | Requires routing between memory systems |

## Architecture-Selection Prediction

The useful memory content should change with history compressibility and the observation regime:

```text
metric quotient + dense evidence      vector/path-integration memory
metric quotient + sparse cue gap      vector memory + cue-span memory
residual route/context ambiguity      sequence, graph, clone-state, or belief-like memory
mixed local metric/global route       hybrid metric + residual memory
hierarchical order                    stack-like memory
```

Input statistics modulate the same rule. Dense observations can sometimes identify latent state directly, reducing the need for internal sequence memory. Sparse or aliased observations increase the value of history, clone-state inference, or hybrid memory. A separate DG-like front end can also make sequence memory more useful by turning a dense noisy stream into a sparse, stable code before control.

## Current Empirical Status

The current lightweight benchmark supports this architecture-selection view at the information level:

- count/vector memory becomes sufficient in the tested trace conditions as action commutativity increases;
- finite history rescues tested order-sensitive trace conditions when count observations alias policy-relevant histories;
- hybrid representations are robust across mixed regimes;
- a cost-based selection analysis can choose the lowest-width sufficient representation among tested representations.

The current benchmark does **not** yet show that a biological recurrent system learns these memories efficiently. That is a later mechanism-level experiment.

## Biological Mapping

The proposed mapping is computational, not one-to-one anatomical:

- entorhinal/grid/path-integration systems are candidate mechanisms for efficient local metric compression;
- hippocampal sequence dynamics are candidate mechanisms for preserving residual route or cue context when observations are sparse or aliased, or when a DG-like front end has already sparsified a dense stream into a sequence-friendly code;
- hippocampal relational or clone-like representations split aliased states into context-specific latent states;
- hippocampal-entorhinal interaction is a plausible hybrid mechanism when local metric state and global route/context state must both remain available.

## Manuscript Use

This should become a central theory section before the empirical results. The paper should not merely say that "history helps when counts fail." It should argue that memory systems should allocate capacity according to the invariants preserved by the task:

- locally compressible dimensions should be compressed;
- residual route or context distinctions should remain represented when they affect prediction or control;
- mixed local-metric/global-route tasks should motivate hybrid memory content.
