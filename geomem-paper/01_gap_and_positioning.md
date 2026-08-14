---
status: draft
priority: high
confidence: medium
depends_on:
  - 00_core_thesis.md
evaluated: true
last_reviewed: 2026-05-20
---

# Gap And Positioning

## Field-Level Gap

Existing theories already describe several representational formats:

- Path integration and grid-cell accounts explain compact metric updating.
- Cognitive map and cognitive graph accounts explain metric, graph-like, and route-based navigation.
- Successor representation explains predictive state occupancy under a policy.
- TEM explains structural generalization across spatial and relational domains.
- CSCG and latent-sequence theories explain context-dependent sequence learning and aliased observations.

The missing piece is a normative and testable account of when each memory architecture is efficient. GeoMem makes task geometry and input sparsity the independent variables.

## Lab-Specific Gap

The Leibold lab has strong mechanistic models of hippocampal sequences, replay/preplay, sparse place coding, grid/place geometry, and sequence-based navigation. Lin, Yiu, and Leibold 2026 shows that a hippocampus-inspired sequence generator helps egocentric visual navigation when input is sparse.

GeoMem generalizes this from:

> Sparse inputs favor hippocampal sequence generators.

to:

> Task geometry and input sparsity jointly determine whether sequence, vector, graph, or hybrid memory is optimal.

## Relationship To Nearby Theories

- Successor representation asks what future states are expected under a policy.
- TEM asks how structural knowledge generalizes across tasks.
- Cognitive graph work asks whether spatial knowledge is metric or graph-like.
- Sequence-centric hippocampal theories ask whether spatial maps emerge from sequence learning.
- GeoMem asks which memory architecture is efficient for a given geometry and input regime.

## Main Reviewer Risk

The phrase "task geometry" must not remain metaphorical. Each experiment should report measurable geometry variables, memory cost, sample efficiency, generalization, robustness, and failure modes.

