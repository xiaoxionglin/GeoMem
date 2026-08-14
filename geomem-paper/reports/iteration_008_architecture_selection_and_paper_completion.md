---
tags: [geomem, report, architecture-selection]
date: 2026-05-21
status: evaluated
related:
  - "[[iteration_007_lightweight_empirical_benchmark]]"
  - "[[iteration_009_rnn_reservoir_navigation]]"
---

# Iteration 008: Architecture Selection And Paper Completion

Date: 2026-05-21

## Missing Parts Identified

The paper was missing four reviewer-critical pieces:

1. **Explicit memory-architecture theory.**  
   The project showed when count observations fail, but it did not yet state an architecture-selection rule.

2. **Cost-aware architecture result.**  
   The benchmarks compared representations, but did not yet ask which architecture is the cheapest sufficient one.

3. **Biological prediction table.**  
   The hippocampal-entorhinal interpretation was mostly verbal.

4. **Paper-structured manuscript.**  
   The main draft still read like a research log rather than a Methods/Results paper.

## Implementation

Added:

- `../../geomem-theory/modules/memory_architecture.md`;
- `../../legacy-pre-split/modules/biological_predictions.md`;
- `../../geomem-experiments/experiments/architecture_selection.py`;
- `../../geomem-experiments/figures/architecture_selection.csv`;
- `../../geomem-experiments/figures/architecture_selection_summary.md`;
- `../../geomem-experiments/figures/architecture_selection.svg`;
- `../manuscript/paper_draft.md`.

The architecture-selection analysis uses the multi-goal lookup benchmark and chooses the lowest-cost representation whose mean optimal-action accuracy exceeds a threshold. Cost is defined as feature memory width:

- memoryless counts: cost 3;
- count history length `h`: cost `3h`;
- hybrid counts plus depth-last history length `h`: cost `3 + 2h`.

## Key Result

At the 0.95 accuracy threshold:

```text
environment         conservative_counts   cost   hybrid_allowed              cost
trace_commute_000   history_counts_h2      6      hybrid_counts_depth_last_h1 5
trace_commute_033   history_counts_h4     12      hybrid_counts_depth_last_h2 7
trace_commute_067   history_counts_h2      6      hybrid_counts_depth_last_h1 5
trace_commute_100   memoryless_counts      3      memoryless_counts           3
```

Interpretation: the cheapest sufficient architecture depends on the action algebra. When the task is fully commutative, vector/count memory is sufficient and cheapest. When action order matters, finite history or hybrid memory becomes necessary. In partial-commutation regimes, hybrid representations can reduce memory cost relative to storing count history alone.

![](../../geomem-experiments/figures/architecture_selection.svg)

## Reviewer Impact

This closes the main conceptual gap between "history helps" and "the best memory architecture depends on task geometry." The paper now has:

- a formal variable: action commutativity;
- an information diagnostic: observation and transition ambiguity;
- a control diagnostic: policy ambiguity;
- a lightweight model benchmark: lookup and learned classifier baselines;
- an architecture-selection result: cheapest sufficient memory class;
- biological predictions: hippocampal vs entorhinal perturbation regimes.

## Remaining Limits

The paper should still be framed as a theory/diagnostic paper. It should not claim that a hippocampus-inspired neural agent has already been shown to outperform path-integration agents. That larger training experiment remains future work.

The next recurrent-control step is tracked in [[iteration_009_rnn_reservoir_navigation]].
