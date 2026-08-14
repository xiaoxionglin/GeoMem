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

# Theoretical Effort Module

## Purpose

The theory should make the paper's main contribution precise:

> Navigation tasks select memory demands by deciding which action-observation histories can be compressed into local metric state and which residual route, branch, cue, or latent-context distinctions must remain available for control.

This frames memory architecture as **task-relevant history compression**. It should explain when metric/path-integration-like memory is enough, when sequence or relational memory is needed, and when a hybrid memory is the principled solution.

The theoretical layer should serve the computational-neuroscience narrative. It must connect the trace controls, room-graph navigation tasks, delay-span results, and hippocampal-entorhinal interpretation. It should not become a generic state-abstraction paper detached from navigation.

## Paper-Facing Formal Question

The theory should answer three questions in the order the paper needs them.

1. **Metric sufficiency:** when can navigation history collapse into a compact local state such as displacement, coordinate, heading, or another path-integration-like variable?
2. **Residual memory necessity:** when do geometry and observations preserve history distinctions that cannot be recovered from such a metric summary and remain relevant for future prediction or policy?
3. **Hybrid memory:** when does the task decompose into a locally metric component and a globally route-, cue-, or context-dependent component, so a hybrid memory is sufficient and preferable to storing full history?

The paper-facing rule is:

> Select the lowest-cost memory representation that preserves the task-relevant quotient of history under the geometry, observation regime, and goal family.

The current paper can operationalize this rule with exact diagnostics and lightweight costs. Stronger optimality claims require an explicit cost class and architecture family.

## Narrative Commitments

The theory should support this argument chain:

1. Navigation unfolds through histories, not isolated observations.
2. A task quotient determines which history distinctions remain relevant.
3. Geometry shapes that quotient through path equivalence, local integrability, loops, bottlenecks, and route structure.
4. Observations shape it through cue density, aliasing, delayed evidence, and whether the sensory stream is intrinsically sparse or sparsified by an upstream code-transforming stage.
5. Metric summaries are efficient when order and route context collapse into a local state.
6. Sequence, graph, clone-state, or belief-like memory is needed for residual distinctions that remain policy-relevant.
7. Mixed local-metric/global-route tasks predict hybrid memory and motivate a hippocampal-entorhinal division of labor.

This is the narrative contribution. Trace monoids, ambiguity diagnostics, recurrent controls, and cue-gap bounds should each clarify one step in that chain.

## Central Formal Model

### Task Family

Define a navigation task family with:

- latent state space `S`;
- action set `A`;
- observation channel `Z(o | s)`;
- transition dynamics `T(s' | s, a)`;
- reward, goal, or shortest-path decision rule;
- initial-state and trajectory distributions;
- goal family `G` when policy sufficiency is evaluated across goals.

Every theorem should state whether it assumes deterministic transitions, exact shortest-path control, observation noise, finite horizon, invalid actions, and boundary effects.

### Histories And Memory

Let the available information at time `t` be an action-observation history `h_t`.

Define:

- memory representation `m_t = f(h_t)`;
- update rule `m_{t+1} = U(m_t, a_t, o_{t+1})`;
- architecture constraints such as additive integration, finite history window, recurrent state, graph state, belief state, or hybrid state.

The formal claim is first about memory **content**. RNNs, reservoirs, and hippocampus-inspired sequence generators are mechanisms that may or may not learn and maintain that content efficiently.

### Task-Relevant History Quotient

Define a task-relevant equivalence relation before using the word "sufficient."

- **Predictive equivalence:** histories induce the same relevant future distributions under future controls.
- **Markov equivalence:** the compressed memory has a well-defined update transition for the modeled task.
- **Policy equivalence:** histories admit the same optimal action set for the tested task.
- **Goal-family policy equivalence:** histories remain policy-equivalent over `G`.
- **Latent-state equivalence:** histories reach the same latent state when the task uses state identity as the reference.

The paper should choose one primary equivalence for the main theorem statements and use the others as diagnostics or corollaries. These equivalences need not coincide: a memory may discard latent state while remaining policy-sufficient for a narrow goal family.

### Metric-Route Decomposition

The flagship navigation model should formalize the mixed regime directly.

Use a state decomposition of the form

```text
s = (x, q)
```

where:

- `x` is a locally metric state within a patch, room, corridor, or chart;
- `q` is global route, branch, portal, latent-context, or task-phase state;
- local actions update `x` through an integrable local composition rule;
- transitions through bottlenecks, portals, repeated rooms, or delayed cue gates update or reveal `q`;
- observations may expose `x` densely while aliasing or delaying evidence about `q`, or may first be sparsified by an upstream code-transforming stage before they reach memory.

This model should make the paper's hybrid claim explicit:

- metric/path-integration-like memory should target `x`;
- sequence, graph, clone-state, or belief-like memory should target residual uncertainty or dependence in `q`;
- hybrid memory is justified when task-relevant history equivalence factorizes across those components well enough for separate memories to preserve the quotient.

Trace tasks provide the clean algebraic limit. Room-graph tasks provide the navigation-facing mixed example.

### Local Metric Proxy

Counts are a controlled abelian proxy, not the biological definition of path integration.

Formalize the minimal property needed for this paper:

- a local memory variable updates compositionally from self-motion or local actions;
- order collapses whenever the local action composition law makes two paths equivalent for task state;
- failure occurs when a local summary aliases residual route or context distinctions that remain task-relevant.

This keeps the trace count theorem useful while leaving room for displacement, heading, uncertainty, landmark correction, and entorhinal-like metric codes in navigation tasks.

## Geometry-Observation Phase Diagram

The theoretical contribution should predict regimes, not only isolated impossibility cases.

| Geometry and observations | History quotient | Preferred memory content | Current paper handle |
|---|---|---|---|
| Locally integrable metric state, dense state evidence | route order collapses for control | metric/vector state | commuting trace limit, dense controls |
| Locally metric state with sparse cue gaps | local state compresses, delayed evidence persists | metric state plus finite-span cue memory | delay-span sweep |
| Dense noisy sensory stream followed by DG-like sparsification | raw stream is not the same as the code reaching memory | sparsified code plus sequence memory | new observation-emission module |
| Route-dependent or noncommuting latent paths, full state observation | geometry distinguishes histories but observation resolves them | current state may suffice | state-revealing observation lemma; empirical negative control still useful |
| Route-dependent or aliased paths, sparse observation | residual route/context distinctions survive | sequence, graph, clone-state, or belief-like memory | trace ambiguity and room-graph sweeps |
| Mixed local rooms plus global bottlenecks or repeated contexts | quotient splits into local metric and global residual components | hybrid metric plus route/context memory | room-graph sweep and hybrid readouts |
| Nested unresolved dependencies | history needs structured stack-like state | stack or structured memory | Dyck-like limiting case only |

This table should discipline both experiments and manuscript claims. Geometry and observation regime jointly select memory demand.

## Main Theory Targets

These are the theory targets that should be visible in the paper because they carry the narrative.

The first formal package is now implemented in `formal_theory.md`, with full proofs in `../../geomem-paper/manuscript/theory_appendix.md`. It chooses goal-conditioned policy sufficiency as the primary criterion, uses Markov sufficiency as a transition diagnostic, uses memory-state cardinality only for the raw-history compression corollary, and keeps the empirical feature-width selector as a bounded proxy rather than a universal optimality claim.

### T1. Order Collapse In Metric Or Commutative Limits

**Target statement:** when task state is generated by a fully commutative path quotient, a compact order-discarding state such as an action-count vector is a complete Markov representation for that quotient.

**Navigation meaning:** vector-like local summaries are justified when path order no longer changes task-relevant state.

**Need to formalize:**

- trace action alphabet and quotient relation;
- finite-depth boundaries and invalid actions;
- mapping from trace state to count state;
- relation between the abelian trace limit and locally integrable metric navigation.

### T2. Limitation Of Additive Metric Proxies

**Target statement:** any additive action memory of the form `m(w) = sum_t phi(a_t)` merges words with the same action multiset. If the task requires different predictions or policies for such words, no readout from that memory is sufficient.

**Navigation meaning:** a local metric proxy fails when residual route or context is behaviorally relevant after metric compression.

**Need to formalize:**

- admissible additive/vector memory class;
- constructive noncommutative or route-aliased counterexample;
- relation to count baselines without equating biological path integration with counting.

### T3. Mixed Metric-Route Hybrid Sufficiency

**Target statement:** if the task-relevant history quotient decomposes into a locally compressible metric factor and a residual route/context factor, a hybrid memory over those factors is sufficient. Under an explicit cost model, it can be cheaper than storing full history.

**Navigation meaning:** hybrid memory is not a compromise baseline; it is the predicted representation for tasks with local metric structure and global route/context dependence.

**Need to formalize:**

- factorization condition for `s = (x, q)` or its task quotient;
- when `q` can be finite history, graph state, clone state, or belief state;
- cost comparison against full-history and pure metric memory;
- factorized versus entangled partial-commutation examples.

### T4. Delayed Evidence Requires Span

**Target statement:** in a balanced delayed-choice task where the cue occurs before a cue-free gap and the decision depends on that cue, a finite observation-history window shorter than the cue-decision gap cannot exceed chance on the cue-dependent decision.

**Navigation meaning:** sequence span matters when sparse evidence must bridge a real behavioral gap.

**Need to formalize:**

- cue distribution and decision labels;
- cue-independent observations during the gap;
- action-history leakage controls;
- distinction between finite-window impossibility and unrestricted recurrent memory.

## Supporting Lemmas And Diagnostics

These are necessary, but they should support the main theory rather than become the headline.

### L1. Representation Aliasing

If `f(h) = f(h')` but the histories have different task-relevant future distributions, `f` is not predictive-sufficient. If the same compressed memory and next action admit incompatible next-memory transitions, the representation is not Markov-sufficient.

Use this to justify transition ambiguity.

### L2. Policy Aliasing

If `f(h) = f(h')` but the optimal action sets for a tested goal are incompatible, no policy over `f(h)` is optimal on both histories.

Use this to justify policy ambiguity and handle optimal-action ties explicitly.

### L3. Goal-Family Separation

If a goal family is rich enough to expose different shortest-path decisions from distinct reachable states, merging those states is not policy-sufficient across that goal family.

Use this to justify multi-goal evaluation without claiming that every latent distinction matters for every reward.

### Optional Approximate Theory

Add approximate theory only after the exact version is stable:

- epsilon-predictive sufficiency;
- value or regret tolerance for policy sufficiency;
- approximate quotient size;
- noisy or finite-precision memory cost.

This is the bridge from exact small-task diagnostics to learned agents.

## Theory-To-Experiment Map

| Theory target | Current empirical handle | Missing control or formal debt |
|---|---|---|
| T1 order collapse | fully commutative trace condition | formal boundary statement complete |
| T2 additive limitation | count/vector baselines and order-sensitive trace tasks | formal counterexample complete; additive-integrator baseline optional |
| T3 hybrid mixed regime | hybrid readouts and room-graph navigation sweep | factorized versus entangled mixed task remains useful |
| T4 delayed span | delay-by-span sweep | cue-gap non-leakage assumption stated |
| L1 Markov aliasing | transition ambiguity | Markov-conflict lemma complete |
| L2 policy aliasing | policy-set ambiguity | tie-aware policy conflict formalized |
| L3 goal-family separation | multi-goal trace benchmarks | separation lemma complete; sampled goals probe coverage |

## Biological Prediction Interface

The formal theory should hand the manuscript testable biological predictions.

| Formal regime | Memory demand | Predicted systems-level interpretation |
|---|---|---|
| local metric quotient, dense evidence | compact metric state | entorhinal/path-integration-like variables should be sufficient more often |
| sparse evidence with real cue gap | delayed evidence span | hippocampal sequence-like buffering should gain importance |
| repeated or aliased route contexts | latent route/context disambiguation | hippocampal relational, sequence, or clone-like memory should matter |
| mixed metric patches and global bottlenecks | factorized local plus residual state | hippocampal-entorhinal cooperation should outperform a single pure summary |
| landmark or observation condition resolves route ambiguity | residual memory shrinks | sequence-memory advantage should shrink |

The theory should not claim an anatomical identity theorem. It should state which computational distinctions a perturbation or recording experiment would test.

## Novelty Contrast

The formal narrative must distinguish GeoMem from nearby programs.

| Nearby program | What it already explains | GeoMem question |
|---|---|---|
| Metric/path-integration and grid-like navigation | efficient local metric and vector operations | when does that compression cease to be task-sufficient? |
| Predictive and relational map theories | predictive occupancy and transferable structure | which memory content is needed before a predictive or relational code can support control? |
| Sequence-first hippocampal theories | latent context and spatial structure from ordered experience | when can sequence history be compressed away, and when must it remain? |
| Lin/Leibold sequence generators | sparse input interacts with hippocampus-inspired sequence dynamics | which task geometries should amplify or shrink that sparse-input advantage? |

This contrast should keep the paper focused on a **selection rule across task regimes**, not on declaring one memory format universal.

## Claim Ladder

| Level | Claim | Required support |
|---|---|---|
| Formal | task-relevant history quotients define required memory content | definitions and theorem statements |
| Algebraic | commutative path quotients allow order collapse; additive summaries fail on required residual order distinctions | T1 and T2 |
| Navigation formal | mixed metric-route tasks predict hybrid memory | T3 plus navigation-facing examples |
| Behavioral formal | delayed sparse evidence imposes span demand | T4 |
| Diagnostic | ambiguity metrics witness Markov or policy loss for proposed representations | exact small-task computation |
| Learned | architectures learn sufficient content efficiently | supervised or RL benchmark |
| Biological | hippocampal-entorhinal systems implement complementary parts of the demand | targeted interpretation and perturbation evidence |

## Claims To Keep Empirical

Keep these outside the theorem layer until stronger evidence exists:

- RNN, reservoir, and Lin-block optimization or sample-efficiency differences;
- learned RL superiority of a hippocampus-inspired sequence generator;
- claims that biological path integration is exactly action counting;
- claims that sequence memory is uniquely optimal in every noncommutative task;
- claims that natural mammalian environments are globally hyperbolic in a precise mathematical sense;
- hippocampal-entorhinal anatomical division of labor beyond a computational interpretation.

## Open Mathematical Questions

1. What is the cleanest mixed metric-route task family that is both navigation-like and theorem-friendly enough to instantiate the hybrid factorization result directly?
2. When does partial commutation factor into vector plus residual memory, and when does it require dependency-graph or belief-state memory?
3. Which stronger cost should underwrite a future optimality claim beyond the current cardinality corollary and feature-width diagnostic: bits, update cost, memory span, or learning cost?
4. Which geometric variables beyond commutativity should become formal axes: bottlenecks, tree-likeness, graph hyperbolicity, metric noise, loop closure, or cue sparsity?
5. How should the additive local-metric proxy be related to biological path integration without weakening the algebraic result or overclaiming equivalence?

## Immediate Theory Checklist

- [x] Define the main task family and choose goal-conditioned policy sufficiency as the primary criterion.
- [x] Write the mixed metric-route model and its factorization assumptions.
- [x] State T1-T4 with assumptions, quantifiers, boundary handling, and failure cases.
- [x] Prove the commutative order-collapse and cue-gap finite-window results.
- [x] Add constructive counterexamples for additive metric proxies and policy aliasing.
- [x] Bound the paper's cost language to feature width empirically and cardinality only for raw-history compression.
- [ ] Add a factorized versus entangled mixed-geometry control if the hybrid theorem becomes central.
- [ ] Add or label empirical negative controls showing geometry and observation regime jointly determine memory demand; the state-revealing observation lemma now covers the theory side.
- [x] Rewrite manuscript theory and results wording so theorem, diagnostic, learned, and biological claims remain separated.
- [ ] Propagate theorem labels into final figure captions once the main figure layout stabilizes.
