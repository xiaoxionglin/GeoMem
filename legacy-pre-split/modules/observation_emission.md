---
status: draft
priority: high
confidence: medium
depends_on:
  - ../00_core_thesis.md
  - formal_model.md
  - memory_architecture.md
evaluated: true
last_reviewed: 2026-05-22
---

# Observation Emission Module

## Purpose

This module separates two cases that should not be conflated in the story or the experiments:

1. the environment itself provides an intrinsically sparse, aliased, or missing sensory stream;
2. a dense or noisy sensory stream is transformed by a DG-like front end into a sparse, cleaner code before sequence memory or control.

The first case is an input problem. The second case is a preprocessing problem. They can produce similar downstream advantages for sequence memory, but they are not the same mechanism.

## Case A: Intrinsically Sparse Or Aliased Emission

Here the world observation channel itself is incomplete or ambiguous.

- the agent sees only a subset of the latent state;
- multiple latent states can map to the same observation;
- some route, cue, or context information is never directly present at decision time;
- sequence or graph memory is useful because the current observation is not sufficient.

This is the regime currently approximated by the trace diagnostics, aliased room tasks, cue-gap tasks, and sparse observation modes in the supervised benchmarks.

## Case B: DG-Sparsified Emission

Here the external sensory stream is not inherently sparse in the ecological sense. Instead, a dentate-gyrus-like stage transforms it into a sparse and less noisy code before it reaches a sequence generator or downstream controller.

- the raw sensory stream can be dense, correlated, or noisy;
- the sparse code is produced by a learned or fixed front end;
- the benefit comes from denoising, pattern separation, or decorrelation before sequence memory;
- the relevant comparison is between a front-end transform plus memory and a more direct sensory-to-action mapping.

This is the correct conceptual slot for the Lin/Yiu/Leibold line of work if the key mechanism is DG-style sparsification rather than an intrinsically impoverished sensory stream.

## Why The Split Matters

The two cases can both favor a sequence generator, but for different reasons:

- case A asks whether the agent can recover from missing or aliased evidence;
- case B asks whether a sparse, stable code makes a shift-register-like memory more useful than an unprocessed noisy stream.

If they are merged, the story can incorrectly imply that all sparse-input advantages come from the same source.

## Planned Use In GeoMem

Use this separation in three places:

- the formal trace and room-graph diagnostics should continue to represent intrinsic observation aliasing and history compression;
- the Lin-facing benchmark should describe DG-like sparsification explicitly, not as generic sparse sensing;
- the manuscript should treat the two cases as related but distinct mechanisms that may later be unified under a broader “observation-to-memory pipeline” story.

## Working Rule

Until the story is merged deliberately, use the following language:

- “intrinsically sparse or aliased observation” for case A;
- “DG-sparsified emission” or “sparsified preprocessed code” for case B.
