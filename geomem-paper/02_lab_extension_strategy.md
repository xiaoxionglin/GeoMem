---
status: draft
priority: high
confidence: medium
depends_on:
  - 00_core_thesis.md
  - 01_gap_and_positioning.md
evaluated: true
last_reviewed: 2026-05-20
---

# Leibold Lab Extension Strategy

## Direct Anchor

Lin, Yiu, and Leibold 2026 provides the main technical predecessor. It shows that a minimal hippocampus-inspired sequence generator can support egocentric visual navigation and outperform LSTM cores after a DG-like sparsifying front end, while producing hippocampus-like spatial representations.

## Extension

GeoMem asks when this effect should appear. The proposed answer is that input sparsity alone is not sufficient; the structure of the task also matters, and the sensory-emission case should be kept separate from the DG-sparsification case. Sequence generators should be most useful when sparse observations are paired with noncommutative, aliased, bottlenecked, or graph-like task geometry, while the DG case asks when a sparse, less noisy code makes the sequence generator more useful downstream.

## Lab Continuity

GeoMem extends several lab themes:

- intrinsic hippocampal sequences as computational motifs;
- replay and preplay as structured dynamics rather than passive readout;
- sequence reservoirs for navigation in unknown environments;
- sparse coding constraints on place representations;
- interactions between MEC input and hippocampal replay flexibility;
- environmental deformation of grid-like spatial codes.

## Lab-Aligned Pitch

The project should be introduced as:

> A theory of when hippocampal sequence generators are useful for navigation, grounded in sparse-input sequence models and extended by controlled manipulations of task geometry.

The broader claim about geometry-dependent memory architecture should follow after the lab-grounded argument is established.
