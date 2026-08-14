---
tags: [geomem, report, reviewer]
date: 2026-05-20
status: living
related:
  - "[[current_results_synthesis]]"
  - "[[paper_missing_parts]]"
  - "[[../manuscript/paper_draft|paper_draft]]"
---

# Reviewer Opinions

Date: 2026-05-20

Role: computational neuroscience / machine learning reviewer

Purpose: this file is reserved for reviewer-style judgment of the project. It should not be used as an implementation log. New entries should evaluate whether the current theory, simulations, figures, manuscript structure, and biological interpretation would be convincing to an expert reviewer.

Current evidence hub: [[current_results_synthesis]]. Current paper-facing sweep report: [[iteration_013_systematic_paper_sweeps]].

## Standing Review Criteria

I will judge the work on six axes:

1. **Scientific contribution.** Does the paper make a non-obvious claim beyond "navigation uses both metric and sequence systems"?
2. **Formal clarity.** Are task geometry, commutativity, observation sufficiency, Markov sufficiency, and policy sufficiency defined precisely?
3. **Empirical support.** Do the simulations directly test the claims, with appropriate controls and negative cases?
4. **Biological relevance.** Is the bridge to hippocampal-entorhinal computation specific enough to generate predictions?
5. **Relation to prior work.** Does the paper clearly distinguish itself from SR, TEM, cognitive graph, path integration, and hippocampal sequence literature?
6. **Manuscript readiness.** Would the current draft be understandable and persuasive as a submitted paper?

## Review 001: Current State

## Summary

This work proposes that the appropriate memory architecture depends on task geometry and input statistics. The central claim is that vector/count-like state is sufficient when action order is irrelevant, while sequence or history-dependent memory is required when action order remains behaviorally relevant. The current implementation supports this claim using trace-monoid environments, ambiguity diagnostics, and a supervised shortest-path policy benchmark.

The idea is promising and increasingly well scoped. The strongest contribution is not the broad statement that navigation uses both metric and sequence systems, which is already familiar, but the operational account of **when compressed coordinate-like observations cease to be Markov or policy-sufficient**.

## Overall Judgment

Recommendation at current stage: **Revise and extend before submission**.

As a theory-and-methods paper, the work has a coherent core and encouraging diagnostic results. As an empirical computational neuroscience paper, it is not yet complete because the model-comparison benchmark is still symbolic/supervised and not yet tied to hippocampus-inspired agent dynamics.

Estimated score if submitted now:

- NeurIPS/ICLR-style ML workshop: borderline positive after cleanup.
- Computational neuroscience conference/workshop: positive if framed as a theory note.
- Full journal/conference paper: major revision.

## Strengths

1. **Clearer novelty than the initial plan.**  
   The project moved beyond "space is sequence" and now makes a more precise claim: geometry and observation structure determine whether history can be compressed.

2. **Trace-monoid environment is a good formal backbone.**  
   It directly manipulates action commutativity rather than relying on spatial shortcuts that may or may not create path equivalence.

3. **Good diagnostic ladder.**  
   The progression from commutativity to observation ambiguity, transition ambiguity, policy ambiguity, and supervised policy prediction is logically strong.

4. **Useful negative result.**  
   The shortcut-tree result is valuable because it shows that graph shortcuts are not automatically equivalent to commutation relations.

5. **Lab fit is credible.**  
   The work naturally extends Lin, Yiu, and Leibold 2026 by asking when hippocampal sequence generators should help, rather than simply showing that they can help under sparse input.

## Main Concerns

1. **The manuscript still reads like a research log.**  
   The current draft contains useful material, but it is organized chronologically around implementation steps. A paper needs a clean Methods/Results structure with each result supporting one claim.

2. **The biological claim is still underdeveloped.**  
   The formal trace environment is useful, but the bridge to hippocampal-entorhinal function remains mostly verbal. The paper needs sharper biological predictions or a direct connection to the Lin/Yiu/Leibold architecture.

3. **The learned benchmark is not yet biologically or architecturally rich.**  
   The supervised benchmark is a good sanity check, but table lookups and decision trees are not memory architectures in the neural sense. The paper should either frame them as diagnostic baselines or add recurrent/sequence-generator models.

4. **The current "history" model is not the same as a learned recurrent state.**  
   Finite history keys demonstrate information sufficiency, but they do not show that an RNN or hippocampal sequence generator will learn the relevant state efficiently.

5. **The trace environment is abstract.**  
   This is acceptable for a theory paper, but the paper must explain why this abstraction captures navigation-relevant structure rather than being only a formal-language exercise.

6. **Related work is not integrated.**  
   The bibliography exists, but the draft does not yet seriously engage with SR, TEM, cognitive graphs, latent sequence theories, POMDP/state abstraction, or formal trace-monoid literature.

## Required Revisions

1. **Restructure the manuscript.**  
   Recommended sections:
   - Introduction
   - Theory: history compression and task geometry
   - Methods: trace environments, observations, policies, diagnostics
   - Results
   - Biological interpretation
   - Related work
   - Discussion and limitations

2. **Convert diagnostics into figures with claims.**  
   Each figure should answer one question:
   - Figure 1: geometry controls commutativity.
   - Figure 2: count observations become non-Markov when order matters.
   - Figure 3: policy ambiguity predicts memory demand.
   - Figure 4: memory/history baselines outperform count-only baselines when geometry is noncommutative.

3. **Add a biological prediction table.**  
   Include predictions for:
   - dense vs sparse sensory input;
   - local metric arenas vs bottlenecked graph environments;
   - route aliasing;
   - hippocampal sequence disruption;
   - MEC/path-integration disruption.

4. **Clarify claims by level.**  
   Separate:
   - representational sufficiency;
   - Markov sufficiency;
   - policy sufficiency;
   - learned performance;
   - biological implementation.

5. **Add at least one recurrent model or explicitly defer it.**  
   If no neural model is added, the paper should be framed as a formal diagnostic/theory paper, not a demonstration of hippocampal sequence-generator superiority.

## Optional But Valuable Additions

- Train a small RNN on observation sequences to predict shortest-path action.
- Add the Lin/Yiu/Leibold sequence-generator model as a later-stage benchmark.
- Add a task family where sensory sparsity is varied independently of commutativity.
- Add robustness analysis across goal states, not only `ABCABCAB`.
- Add examples or diagrams of trace states so readers understand the environment visually.

## Decision-Critical Issue

The paper needs to decide what it wants to be.

Option A: **Theory/diagnostic paper**  
Then the current trace-monoid framework, ambiguity metrics, and supervised benchmark are enough after restructuring and better related work.

Option B: **Computational neuroscience model paper**  
Then it needs a recurrent or hippocampus-inspired agent benchmark, ideally connected to Lin/Yiu/Leibold 2026.

Option C: **AI memory architecture paper**  
Then it needs broader agent-memory baselines and tasks beyond the trace environment.

My recommendation is Option A first, with a clearly stated path to Option B.

## Reviewer Verdict

The core idea is worth pursuing. The current formal results are stronger than the initial framing and already reveal a nontrivial point: **commutativity, not shortcut density, is the right control variable for history compression**. The work becomes compelling when it shows that count/vector observations are not merely lossy but non-Markov and policy-insufficient in noncommutative geometry.

However, the manuscript is not yet ready as a full paper. The next revision should focus less on adding more diagnostics and more on turning the existing results into a clean argument with figures, methods, related work, and clearly bounded claims.

## Actionable Reviewer Requests

Before the next review, the project should address these items:

- Rewrite the manuscript around claims and figures rather than chronological progress.
- Add a Methods section that makes the trace-monoid environment reproducible without reading the code.
- Add a Results section where each result has a claim, method, quantitative outcome, and interpretation.
- Add a biological predictions table connecting task geometry to hippocampal sequence disruption and entorhinal/path-integration disruption.
- Decide whether the current paper is a theory/diagnostic paper or a neural-model paper. The current evidence supports the former more strongly.
- Treat supervised lookup and decision-tree baselines as diagnostic baselines, not as final neural memory architectures.

## Review 002: After Architecture-Selection Revision

Date: 2026-05-21

Recommendation at current stage: **promising theory/diagnostic paper; still not a complete neural-model paper**.

The revision improves the project substantially. The key new ingredient is the cost-aware architecture-selection analysis. The manuscript can now make the architecture claim directly: the cheapest sufficient memory representation changes with task geometry. In the fully commutative trace task, memoryless counts are sufficient and cheapest. In noncommutative trace tasks, the selector shifts to finite history or hybrid memory.

### Improved Strengths

1. **The architecture claim is now operational.**  
   The paper no longer only says that count memory fails. It now asks which architecture reaches policy sufficiency at minimal memory cost.

2. **The manuscript has a clearer paper structure.**  
   `manuscript/paper_draft.md` is organized around theory, methods, results, biological interpretation, and limitations.

3. **The biological predictions are testable.**  
   The perturbation table now makes separate predictions for hippocampal sequence disruption and entorhinal/path-integration disruption.

4. **The claim is properly bounded.**  
   The current version frames itself as a theory and diagnostic paper rather than overclaiming neural-agent evidence.

### Remaining Concerns

1. **Trace environments are still abstract.**  
   The formal control is useful, but a reviewer may ask whether the result transfers to more navigation-like aliased mazes.

2. **The `depth_last` cue is unusually informative.**  
   The manuscript now flags this caveat, but a future benchmark should remove route-reversal cues.

3. **No recurrent or hippocampus-inspired model has been trained yet.**  
   This is acceptable for a theory paper, but not for a full computational-neuroscience model paper.

### Updated Verdict

The current project is now convincing enough as a **theory/diagnostic manuscript draft** if the paper is explicit about scope. The strongest claim is:

> Task geometry determines which history variables can be compressed, and therefore which memory architecture is cheapest while remaining policy-sufficient.

The current project is not yet convincing enough for the stronger claim:

> Hippocampal sequence generators outperform path-integration systems in learned biological agents.

That stronger claim requires the next-stage recurrent or hippocampus-inspired benchmark.

## Review 003: After Lightweight RNN/Reservoir Revision

Date: 2026-05-21

Recommendation at current stage: **stronger and more credible, but still supervised-control rather than RL**.

The new RNN/reservoir benchmark addresses an important earlier weakness. The paper no longer relies only on representational diagnostics, lookup policies, and decision trees. It now shows that a trainable recurrent model and several fixed recurrent reservoirs can improve supervised navigation policy learning when observations are sparse or aliased.

### Improved Strengths

1. **The memory-architecture claim now has a neural-style control.**  
   Under sparse/aliased observations, mean action accuracy improves from `0.641` for memoryless linear and `0.552` for memoryless MLP to `0.736` for the NumPy RNN.

2. **The task set is less abstract.**  
   The benchmark now includes `aliased_rooms`, `aliased_loop_rooms`, `bottleneck_rooms`, and a cue-corridor diagnostic in addition to trace geometries.

3. **The result is not a single cherry-picked reservoir.**  
   Diagonal reservoirs are strongest on average in the current sparse/aliased run, while orthogonal, nilpotent, and block-hybrid reservoirs are competitive in different conditions. This supports the broader claim that recurrent dynamics should be matched to task geometry rather than declaring one recurrent matrix universally optimal.

4. **The direct memory-advantage summary is useful.**  
   `figures/rnn_memory_advantage.svg` cleanly shows where recurrent memory helps over the best memoryless baseline.

5. **Temporal sparsity is now separated from unit sparsity.**  
   The new temporal-dropout and cue-then-blank conditions address an important conceptual gap. Under sparse/aliased observations, the RNN remains above memoryless linear in full input (`0.736` vs `0.641`), temporal dropout (`0.664` vs `0.599`), and cue-then-blank (`0.635` vs `0.531`).

### Remaining Concerns

1. **This is still imitation learning, not reward-driven RL.**  
   The current benchmark tests whether memory can express and learn shortest-path action labels. It does not test exploration, reward sparsity, actor-critic stability, or policy-gradient credit assignment.

2. **The memoryless MLP is not yet a strong baseline.**  
   Its underperformance suggests either optimization instability or too little tuning. The memoryless linear baseline is currently the more reliable comparator.

3. **Greedy rollout success is noisy.**  
   Action accuracy is the clean supervised metric. Rollout success can be included as a secondary diagnostic, but the paper should not lean on it heavily.

4. **The cue-corridor task is small.**  
   It should be described as a diagnostic, not as decisive ecological evidence. A richer family of repeated-corridor or repeated-room navigation tasks remains the next empirical gap.

5. **Temporal dropout is still artificial.**  
   It is a good controlled test, but it is not yet a naturalistic visual sparsity model. The paper should describe it as a mechanistic stress test, not as a realistic sensory stream.

### Updated Verdict

The manuscript can now make the bounded learned-agent claim:

> In lightweight supervised navigation tasks, recurrent state improves policy prediction over memoryless baselines when task geometry and observation sparsity make current observations insufficient.

It still should not claim:

> Hippocampal sequence agents have been shown to outperform path-integration agents under full reinforcement learning.

The next version should keep the supervised-control benchmark as supporting evidence and reserve full RL for future work.

## Review 004: After Supervised Lin Bridge

Date: 2026-05-21

Recommendation at current stage: **good supervised bridge to the lab anchor; RL gap remains explicit**.

The revision addresses two reviewer concerns at once. First, the supervised behavioral examples are no longer only repeated rooms and abstract trace graphs: delayed landmark choice, landmark-gap detour, shortcut-route memory, bottleneck routes, and lightweight egocentric-landmark views now make the history demand behaviorally legible. Second, the sequence comparison now reports Lin-style `R`, `L`, and `ell = R + L - 1` instead of asking a generic nilpotent reservoir to stand in for the Lin architecture.

### Improved Strengths

1. **A sequence-horizon result is now visible.**  
   On sparse cue-then-blank behavioral tasks, `L=8` Lin-style sequence readouts reach `0.763` to `0.784` mean action accuracy, while the shorter `L=2` variants stay near `0.522` to `0.544`.

2. **The behavioral examples contain memory-critical decisions.**  
   Stratified directed-choice splits prevent delayed-choice tests from being dominated by forward actions.

3. **Input injection fairness is addressed.**  
   The `R=3`, `L=4` reference compares front-band injection against all-unit injection, so the manuscript can state when a finite sequence buffer is being overwritten by dense recurrent input.

4. **There is a first representation diagnostic.**  
   The Lin sequence summaries expose behavioral context readout accuracy in addition to policy accuracy.

### Remaining Concerns

1. **The Lin sequence readout is still fixed and supervised.**  
   It is not the trained hippocampus-inspired actor-critic architecture.

2. **The egocentric-landmark observation is a proxy.**  
   It is more behavioral than a symbolic state id, but it is not continuous egocentric vision.

3. **Small delayed-choice datasets can saturate.**  
   Perfect sparse delayed-choice accuracy should be reported as a lightweight controlled result, not as a large-scale navigation performance claim.

### Updated Verdict

The manuscript can now say:

> In supervised behavioral controls, explicit sequence horizon modulates sparse delayed-route performance in the direction expected from the Lin/Yiu/Leibold sequence-generator result.

It still needs actor-critic visual navigation before claiming a direct reproduction or extension of Lin 2026.

## Review 005: After Long Independent Sequence Bridge

Date: 2026-05-21

Recommendation at current stage: **the supervised bridge now tests the right horizon effect**.

The long bridge fixes an important mismatch in Review 004. The short `L <= 8` grid was a useful toy control but could not represent the long sequence regime relevant to Lin/Yiu/Leibold. The new experiment uses independent block-nilpotent input chains, `L` up to `64`, and a balanced 72-step delayed-cue task.

The result is clearer than the short-task grid: under sparse cue-then-blank input, memoryless linear, the small NumPy RNN, and all `L=4/16` blocks sit at `0.506` action accuracy, while the `L=64` blocks reach `0.694` to `0.729`. Context decoding for `R=3` also rises to `0.702` at `L=64`.

The residual gap is now specific rather than conceptual: these are supervised block readouts, not learned actor-critic egocentric visual sequence generators.

## Review 006: After Long Architecture Comparison

Date: 2026-05-21

Recommendation at current stage: **the supervised long-horizon control is now defensible; learned matched-control gap remains**.

The long delayed benchmark now compares independent Lin blocks against the prior memory architectures on the same task rather than against only memoryless linear and the small trained RNN. This is a meaningful improvement. The comparison includes memoryless MLP, explicit observed-history windows, every 32-unit reservoir family, the prior shared Lin buffers, and the trained NumPy RNN.

### Improved Strengths

1. **The old architecture families are now present in the critical long condition.**  
   Under sparse cue-then-blank input, the memoryless controls and NumPy RNN sit at `0.506`, the best previous shared Lin buffer reaches `0.506`, and the best 32-unit fixed reservoir reaches `0.505`.

2. **Explicit history anchors the interpretation.**  
   `H4` and `H16` remain at `0.506`, `H64` rises to `0.690`, and `H80` reaches `1.000`. The independent `L=64` blocks lie on that same horizon curve at `0.694` to `0.729`.

3. **The result is no longer just "nilpotent beats a weak RNN."**  
   The long comparison separates the old short buffers and generic lightweight reservoirs from the input-specific long block state relevant to the Lin bridge.

### Remaining Concern

The strongest alternative explanation is now capacity and effective horizon, not a missing baseline. The independent block state has many feature-specific delayed coordinates while the old RNN and generic reservoirs use the lightweight 32-unit setting. That is acceptable for the current supervised bridge because explicit history makes the information demand visible, but a stronger model claim needs a trained recurrent control matched for delayed capacity or a direct Lin-style actor-critic comparison.

### Updated Verdict

The manuscript can now state:

> In a supervised long delayed navigation control, independent long sequence blocks recover task-relevant cue information in the same regime where the prior short recurrent controls fail and explicit finite history begins to recover it.

It still should not state:

> Long Lin blocks are universally superior to recurrent agents at matched capacity and training budget.

## Review 007: After Systematic Paper Sweeps

Date: 2026-05-22

Recommendation at current stage: **the empirical organization is materially improved**.

The prior learned-control evidence was becoming hard to judge because separate trace diagnostics, named behavioral tasks, reservoir matrices, and long Lin controls each carried part of the argument. The new revision introduces three paper-facing sweeps: a repeated formal trace sweep, a factorized rooms-on-graph navigation sweep, and a delay-by-span sweep for the Lin-facing horizon claim.

### Improved Strengths

1. **The formal result is no longer a small hand-picked trace series.**  
   The main relation now repeats over action alphabet size, trace depth, and intermediate commutation relation graphs.

2. **The navigation result is factorized.**  
   Local room geometry is held fixed while global topology, aliasing, cue visibility, and primary memory architecture vary.

3. **The Lin bridge now has a phase-boundary control.**  
   The delay-by-span figure directly asks when memory span reaches task delay, with a visible-state ceiling and finite-history comparison.

### Remaining Concerns

The new organization makes the residual gap clearer. The room-graph sweep is still symbolic supervised navigation, and the recurrent models are lightweight controls rather than a tuned or actor-critic model family. The current claims should therefore remain about geometry-dependent sufficiency and supervised architecture pressure, not complete biological-agent superiority.

### Updated Verdict

This is now a more persuasive theory-and-diagnostic manuscript structure. A reviewer can see the argument as:

1. geometry controls history compressibility;
2. navigation-like local-metric/global-graph tasks expose the same memory pressure;
3. sequence span matters when delayed policy evidence exceeds shorter memory horizons.
