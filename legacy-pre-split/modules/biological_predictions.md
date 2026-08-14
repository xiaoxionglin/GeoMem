---
status: draft
priority: high
confidence: medium
depends_on:
  - memory_architecture.md
  - biological_mapping.md
evaluated: true
last_reviewed: 2026-05-22
---

# Biological Predictions

## Prediction Table

| Task manipulation | Memory demand predicted by GeoMem | Hippocampal perturbation prediction | Entorhinal/path-integration perturbation prediction |
|---|---|---|---|
| Dense local metric arena with reliable self-motion | Vector/path-integration memory should be sufficient | Mild impairment unless sequence replay is required for planning | Strong impairment in metric updating, homing, and shortcut behavior |
| Dense/noisy sensory stream with DG-like sparsification | DG-style sparsified code plus sequence buffer | Impairment should grow if sparsification is weakened or the code is no longer clean | Less specific unless the metric channel is also degraded |
| Sparse landmarks in a locally metric arena | Hybrid memory: metric state plus sequence buffer for missing input | Impairment grows as landmarks become sparse or intermittent | Impairment remains if self-motion is needed to bridge gaps |
| Bottleneck maze with locally metric rooms | Hybrid memory: local metric compression plus graph/route context | Impaired route disambiguation and replay across bottlenecks | Impaired local metric estimates inside rooms and at door transitions |
| Tree-like or route-dependent environment | Residual route/context memory when current cues do not identify branch history | Strong impairment when branch history remains behaviorally relevant | Weaker impairment unless local displacement is required |
| Aliased corridors or repeated sensory states | Clone-state or episodic sequence memory | Strong impairment in distinguishing visually identical states by history | Mild-to-moderate impairment depending on metric cue reliability |
| Fully commutative symbolic/metric task | Count/vector memory | Limited impairment after learning if current vector/count state is available | Stronger impairment if vector state must be updated from motion |
| Partial-commutation task | Hybrid memory | Selective impairment on noncommuting route distinctions | Selective impairment on commuting metric dimensions |

## Experimental Implication

The clean biological prediction is not that hippocampus always beats path integration. It is that hippocampal contribution should increase with:

- observation aliasing;
- intrinsic sensory sparsity;
- DG-style sparsification of a noisy stream;
- route dependence;
- bottleneck or graph-like topology;
- route/context distinctions not resolved by local metric state.

Entorhinal/path-integration contribution should increase with:

- reliable self-motion;
- local metric continuity;
- history compression into local metric state;
- low sensory aliasing;
- tasks requiring vector shortcuts or homing.

## Link To Leibold-Lab Work

GeoMem naturally extends Lin/Yiu/Leibold 2026 by predicting when their sequence-generator advantage should be large. The extension has two separate cases: the sequence advantage should be largest when intrinsically sparse or aliased observations leave policy-relevant route or context distinctions unresolved by local metric state, and it should also appear when a DG-like front end sparsifies and denoises a dense stream before the sequence generator. The advantage should be smallest when dense observations already provide a policy-sufficient current state.
