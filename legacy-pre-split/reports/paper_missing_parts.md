---
tags: [geomem, report, paper-audit]
date: 2026-05-21
status: current
related:
  - "[[current_results_synthesis]]"
  - "[[reviewer_opinion]]"
  - "[[../manuscript/paper_draft|paper_draft]]"
---

# Paper Missing Parts Audit

Date: 2026-05-21

## Resolved In This Cycle

| Missing part                                  | Status | Implementation                                                                                   |
| --------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------ |
| Explicit memory-architecture theory           | Done   | `modules/memory_architecture.md`                                                                 |
| Formal theory package and full proof appendix | Done   | `modules/formal_theory.md`, `manuscript/theory_appendix.md`                                      |
| Biological prediction table                   | Done   | `modules/biological_predictions.md`                                                              |
| Cost-aware architecture selection result      | Done   | `experiments/architecture_selection.py`, `figures/architecture_selection.*`                      |
| Multi-goal robustness                         | Done   | `experiments/lightweight_empirical_benchmark.py`                                                 |
| Paper-structured draft                        | Done   | `manuscript/paper_draft.md`                                                                      |
| Reviewer-facing critique file                 | Done   | `reports/reviewer_opinion.md`                                                                    |
| Supervised behavioral Lin bridge              | Done   | delayed-route tasks and `R/L/ell` sequence readouts in `experiments/rnn_navigation_benchmark.py` |
| Systematic paper-first sweeps                 | Done   | `systematic_formal_sweep.py`, `systematic_navigation_sweep.py`, `delay_span_sweep.py`            |

The three paper-facing sweep figures are:

![](../figures/systematic_formal_sweep.svg)

![](../figures/systematic_navigation_sweep.svg)

![](../figures/delay_span_sweep.svg)

## Still Missing For A Stronger Future Paper

These are not required for the current theory/diagnostic version, but they are required for a stronger computational-neuroscience model paper:

1. A trained hippocampus-inspired sequence-generator actor-critic agent.
2. Continuous egocentric visual input rather than lightweight egocentric-landmark proxies.
3. Perturbation experiments comparing sequence-memory ablation and path-integration/vector-memory ablation.
4. Direct RL comparison to the Lin/Yiu/Leibold 2026 architecture.

## Recommended Framing

Submit or develop the current version as a **theory and diagnostic paper**:

> Geometry, observations, and goals determine which history variables survive task-relevant compression. Local metric summaries handle compressible state, while residual route/context memory is needed when policy-relevant distinctions remain unresolved.

Do not yet frame it as:

> A complete biological model showing hippocampal sequence generators outperform path integration.

For the evidence stack behind this audit, see [[current_results_synthesis]].
