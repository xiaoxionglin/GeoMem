---
tags: [geomem, report, paper-sweep]
date: 2026-05-22
status: current
related:
  - "[[current_results_synthesis]]"
  - "[[../manuscript/paper_draft|paper_draft]]"
---

# Iteration 013: Systematic Paper Sweeps

Date: 2026-05-22

## Question

Can the current formal, recurrent, and Lin-facing results be reorganized into factorial sweeps that support main paper figures rather than isolated task examples?

## Implemented Sweeps

1. `../../geomem-experiments/experiments/systematic_formal_sweep.py` varies trace alphabet size, trace depth, and repeated commutation relation graphs. It writes ambiguity, policy-sufficiency, and cheapest sufficient memory results into one formal table.
2. `../../geomem-experiments/experiments/systematic_navigation_sweep.py` builds a rooms-on-graph family with loop-rich, bottleneck, and tree-like global topology. It sweeps low/high aliasing and persistent/cue-gap observations against memoryless, explicit-history, hybrid, RNN, reservoir, and independent Lin-block readouts.
3. `../../geomem-experiments/experiments/delay_span_sweep.py` maps delayed sparse cue navigation over task delay `D` and memory span `H/L`, with a dense visible-state memoryless ceiling and recurrent/nilpotent controls.

## Main Outputs

- `../../geomem-experiments/figures/systematic_formal_sweep.*`
- `../../geomem-experiments/figures/systematic_navigation_sweep.*`
- `../../geomem-experiments/figures/delay_span_sweep.*`

![](../../geomem-experiments/figures/systematic_formal_sweep.svg)

![](../../geomem-experiments/figures/systematic_navigation_sweep.svg)

![](../../geomem-experiments/figures/delay_span_sweep.svg)

## Results

The formal sweep preserves the main trace result across relation replicates, alphabet sizes `K in {3,4}`, and depths `4` and `6`: counts-policy ambiguity and counts-transition ambiguity fall toward zero only at full commutation, and the cheapest sufficient lookup memory returns to memoryless counts only at full commutation.

The navigation sweep gives a controlled learned interaction. On high-aliasing cue-gap decisions, hybrid vector-plus-history reaches `0.593` on loop-rich room graphs, `0.709` on bottleneck graphs, and `0.544` on tree-like graphs, versus `0.474`, `0.616`, and `0.330` for the memoryless linear readout. The run records decision-level metrics rather than relying only on all-action accuracy.

The delay-span sweep turns the Lin bridge into a phase-boundary result. A dense visible-state memoryless readout is `1.000` at every tested delay. Under sparse cue gaps, `H/L=64` solves `D=64` for explicit history (`1.000`) and nearly solves it for the `R=3` independent Lin block (`0.993`), but both partial-span variants fall to about `0.65` to `0.66` at `D=80`; `H/L=80` restores ceiling performance.

## Interpretation

The older targeted RNN/reservoir and long-T-maze results remain useful controls, but they should now be subordinate to three paper figures: formal geometry sufficiency, factorized room-graph navigation, and delay-by-span sequence memory. This better matches the central claim that memory architecture should track geometry- and observation-defined history demand.

See [[current_results_synthesis]] for the claim stack and remaining gaps after this consolidation.
