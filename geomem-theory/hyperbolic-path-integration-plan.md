this is from sven.
# Path integration in H²: sequences vs. continuous attractors

Working document. Status: pre-Phase-0.

**One-line framing.** Use the hyperbolic plane as a controlled abstraction of hierarchical / tree-like state spaces, and ask whether curvature determines *which kind of representation* a trained network builds for path integration — a continuous attractor manifold (Euclidean regime) or a sequential / symbolic memory (hyperbolic regime).

---

## 1. Core hypotheses and pre-registered predictions

Register these before Phase 2 results are seen. Each has a stated null.

### H1 — Compression (analytic, no training required)

In ℝ², an *n*-step trajectory ranges over radius ~*n*, so the endpoint costs O(log n) bits while the action word costs Θ(n) — compression ratio ~1/n. In H², distance grows linearly in *n* and bits-to-localize grow linearly in distance, so both cost Θ(n) — compression ratio O(1).

- **Prediction H1.** Empirical mutual information I(endpoint ; action word) / H(action word) → 0 as n grows for K = 0, and stays bounded away from 0 for K < 0.
- **Null.** Ratio decays at the same rate in both.
- **Note.** This is provable, not just measurable. Do the proof *and* the simulation; the simulation is a sanity check on the estimator, not the result.

### H2 — Representational dimensionality

- **Prediction H2.** Participation ratio / intrinsic dimensionality of the RNN hidden state, measured over trajectories of length L, saturates at ≈2–3 for K = 0 and grows monotonically with L for K < 0.
- **Growth rate prediction.** Dimensionality grows ~ linearly in L/R for K < 0 (Lyapunov exponent 1/R), until capacity-limited.
- **Null.** Saturation in both, at the same value.
- **Confound to control.** Dimensionality can grow simply because the task is harder. Match final decoding accuracy across K, not just parameter count.

### H3 — Mechanism

- **Prediction H3a.** Zero-input fixed-point analysis reveals a continuum of slow points (approximate line/plane attractor) for K = 0 and few or no slow points for K < 0, with dynamics dominated by transients.
- **Prediction H3b (route dependence).** At *fixed endpoint*, decoding accuracy and hidden state depend on the route taken for K < 0; they do not for K = 0. (Geometric basis: generic step/turn parameters generate a free group, so distinct action words → distinct endpoints.)
- **Prediction H3c (discreteness).** Perturbing a single early action produces a graded shift in hidden state for K = 0 and a discrete, quantized jump for K < 0 on tessellated substrates.
- **Null.** No systematic difference; both look like bump dynamics with different gains.

### H4 — Crossover, not dichotomy

Urdapilleta, Si & Treves (2015) show that *local* hyperbolic grids (heptagonal, 7 neighbours) do self-organize when grid spacing is small relative to the curvature radius. So:

- **Prediction H4.** All effects are governed by the dimensionless ratio L/R (path length in curvature radii), with a crossover near L/R ≈ 1. Below it, hyperbolic and Euclidean networks are indistinguishable. Above it, H2/H3 separate.
- **Sharper version (optional, higher risk).** The transition is a knee, not a smooth ramp.

---

## 2. Dimensionless parameters — fix these once

| Symbol   | Meaning                               | Sweep                                                |
| -------- | ------------------------------------- | ---------------------------------------------------- |
| K        | Gaussian curvature                    | continuous through 0; report as R = \|K\|^(-1/2)     |
| L/R      | total path length in curvature radii  | **primary axis**: 0.3, 1, 3, 10                      |
| s/R      | single step length in curvature radii | hold small (≤0.1) so discretization isn't the effect |
| N_place  | readout basis size                    | two conditions: fixed, and scaling with area         |
| N_hidden | RNN width                             | matched across K                                     |

**Everything must be reported against L/R, not against absolute length.** Absolute comparisons across K are meaningless.

---

## 3. Phases

### Phase 0 — Geometry kernel (1–2 weeks)

**Goal.** One code path that handles K < 0, K = 0, K > 0 without branching on the sign.

- [ ] Hyperboloid model in ℝ^(2,1); state (x, u) with ⟨x,x⟩ = −1, ⟨u,u⟩ = 1, ⟨x,u⟩ = 0
- [ ] `step(x, u, θ, s, K)` — turn then geodesic flow
- [ ] Renormalization every step; log precision failure point (expect d/R ≈ 30 in float64)
- [ ] Conversions to Poincaré disk / half-plane **for visualization only** — never for computation

**Validation suite (all four must pass):**
1. Hyperbolic law of cosines: cosh c = cosh a cosh b − sinh a sinh b cos γ
2. Geodesic triangle area = π − (α + β + γ)
3. Holonomy around a closed loop = enclosed area (Gauss–Bonnet)
4. Geodesic deviation grows as sinh(s/R), and as s in the K → 0 limit

**Gate 0.** All four identities reproduce to machine precision at d/R ≤ 10. If not, stop and fix — every later result depends on this.

**Background concepts for this phase:** models of H² (hyperboloid, Poincaré disk, half-plane) and the isometries between them; geodesics and the exponential map; Gauss–Bonnet; the Jacobi equation.

---

### Phase 1 — Trajectory generation and target code (1 week)

This phase is short but is where the project can quietly fail.

- [ ] Intrinsic random walk: sample turn angle and step length in the tangent space, **not** a Euclidean walk pushed through a chart
- [ ] Verify ballistic drift for K < 0 (E[d(x₀, X_n)] ~ cn) and diffusive √n for K = 0. This is a correctness check on the walk.
- [ ] Place-field basis: centres sampled uniformly w.r.t. **hyperbolic area**; fields defined by geodesic distance, exp(−d(x,c)²/2σ²)
- [ ] Two readout conditions: (a) N_place fixed — capacity-limited; (b) N_place ∝ area — controlled coverage

**Traps, stated explicitly:**
- Sampling field centres uniformly in the Poincaré disk imposes a Euclidean prior. Don't.
- There is no "arena." The walk escapes to infinity; the environment is defined by the time horizon T, not a boundary.
- Coverage at fixed σ needs ~e^(R/…) cells. Condition (a) is where H2/H3 should show up; condition (b) is the control.

**Gate 1.** Walk statistics match theory (ballistic vs. diffusive) and the place code decodes position accurately from ground-truth positions alone (no RNN). If the readout can't represent position, nothing downstream is interpretable.

**Background concepts:** volume growth and area in H²; random walks on manifolds — transience, positive speed, the boundary at infinity; sampling from non-uniform measures.

---

### Phase 2 — Load-bearing experiment: RNN path integration, K swept (3–5 weeks)

**Goal.** The one experiment where "sequences beat attractors" is an empirical claim rather than a property of the substrate.

- [ ] Vanilla RNN, action input, place-code readout — deliberately match the Cueva & Wei / Sorscher setup so the K = 0 case reproduces known results
- [ ] Sweep K continuously through 0 at matched L/R, matched N_hidden, matched final accuracy
- [ ] Two action conditions, run separately:
  - **egocentric turn** (θ, s) — expect accumulating heading error, amplified by sinh
  - **ideal-point bearing** (α, s) — no accumulating heading error; expect *more* attractor-like solutions
- [ ] Sanity check: at K = 0, hexagonal grid units should emerge. If they don't, the pipeline is wrong, not the hypothesis.

**Primary analyses (chosen to discriminate H2/H3, not to describe):**

| Analysis | Attractor signature | Sequence signature |
|---|---|---|
| Participation ratio vs. L | saturates | grows |
| Zero-input fixed points | continuum of slow points | few; transient dynamics |
| Decode accuracy vs. route, fixed endpoint | route-independent | route-dependent |
| Single early-action perturbation | graded shift | discrete jump |

**Gate 2 — the decision point.**
- Smooth separation with K → scaling result, proceed to Phase 3 as supporting evidence.
- Knee near L/R ≈ 1 → phase transition; this is the headline, restructure the paper around it.
- **Null result** → do not push harder on hyperbolic environments. Reframe toward the Sharpee direction: hyperbolic *representations* of Euclidean tasks. Record this now so the pivot isn't a defeat.

**Background concepts:** continuous attractor networks and ring attractors; fixed-point / slow-manifold analysis of RNNs; participation ratio and intrinsic dimensionality estimation; the conformal isometry hypothesis for grid cells.

---

### Phase 3 — Discrete substrates (3 weeks, can run in parallel with Phase 2)

**Goal.** Not to confirm the sequential result (which the substrate partly gives for free), but to measure *how hard it is* for a capacity-limited RNN to solve, and to look for a specific mechanism.

- [ ] Three substrates: {5,4} tiling dual graph; degree-3 regular tree (δ → 0 limit); square lattice (control)
- [ ] Actions = edge crossings; position readout = tile identity
- [ ] Measure: minimum N_hidden for accurate integration as a function of path length
  - **Prediction:** linear in the tree, linear with smaller slope in the tiling, logarithmic in the lattice
- [ ] **Dehn signature.** On surface groups the word problem is solved by Dehn's algorithm — scan, find a subword more than half a relator, replace with the shorter complement. Linear time, stack-flavoured. Test: does the hidden state encode a *reduced* word? Does it compress cancelling subpaths (go-and-return) to the same state as never having moved?

**Gate 3.** Capacity scaling separates the three substrates. If tree and lattice scale identically, the RNN is not exploiting structure and the measurement is uninformative.

**Background concepts:** regular tessellations {p,q} of H² and their dual graphs; Cayley graphs; Gromov δ-hyperbolicity and the tree ↔ H² ↔ ℝ² interpolation; the word problem and Dehn's algorithm; free groups and ping-pong.

---

### Phase 4 — Landmarks and distal cues (later)

**Goal.** Test whether exteroceptive input rescues the capacity limit, and exploit a free geometric prediction.

- [ ] Add landmarks with noisy range and/or bearing observations
- [ ] **Prediction.** In geodesic polar coordinates about a landmark, transverse localization error from bearing is σ_φ·sinh(r) while range error stays σ_r. So a capacity-limited agent in H² should learn to weight **range heavily and bearing barely**, with the crossover at r ≈ R. In ℝ² both remain useful.
- [ ] Test by ablation: remove bearing input, remove range input, measure degradation as a function of landmark distance
- [ ] Secondary: does the network discover an ideal-point (Busemann) reference rather than a landmark-centred one?

**Background concepts:** Busemann functions and horocycles; the boundary at infinity; Fisher information on manifolds; landmark-based vs. path-integration-based localization.

---

## 4. Standing risks

1. **Chart leakage.** Any Euclidean prior smuggled in through a coordinate choice (field centres, initialization, loss weighting) will produce artefacts that look like results. Audit every sampling step for chart-independence.
2. **Numerical range.** Coordinates grow like e^(d/R). Cap L/R at ~10 in float64 or move to log-space / relative representations.
3. **Difficulty confound.** Hyperbolic path integration is simply harder. Any representational difference must be shown at matched task performance, or it's a difficulty effect.
4. **Substrate tautology (Phase 3).** Tree and tiling give the sequential answer partly by construction. Phase 2 carries the claim; Phase 3 supports it.
5. **Trainability ceiling.** Reachable states grow like e^(L/R). Expect training to become infeasible before the geometry becomes uninteresting.

---

## 5. Open decisions

- [ ] Discrete-time steps vs. continuous curvature control κ(t) for Phase 2 — currently leaning discrete for comparability with existing grid-cell work
- [ ] Whether to include K > 0 (sphere) as a third arm — cheap given the unified kernel, and makes "curvature matters" a two-sided claim rather than a hyperbolic special case
- [ ] Whether the compression result (H1) is a standalone short paper
- [ ] Target venue, which determines whether the framing is NeuroAI or geometric ML

---

## 6. Literature anchors

**Empirical hook.** Zhang, Rich, Lee & Sharpee, *Hippocampal spatial representations exhibit a hyperbolic geometry that expands with experience*, Nat Neurosci 26:131–139 (2023).

**The counterexample to engage.** Urdapilleta, Si & Treves, *Can rodents conceive hyperbolic spaces?*, J R Soc Interface 12:20141214 (2015). Heptagonal grids self-organize on hyperbolic surfaces at appropriate spacing-to-curvature ratios. Constrains H2/H4.

**Methodology to port.** Cueva & Wei, ICLR (2018); Banino et al., Nature (2018); Sorscher et al., *A unified theory for the computational and mechanistic origins of grid cells*, Neuron (2023).

**Debate to enter.** Peer, Brunec, Newcombe & Epstein, *Structuring knowledge with cognitive maps and cognitive graphs*, TiCS 24:37–54 (2020).

**Related machinery.** Ghadimi Atigh, Keller-Ressel & Mettes, *Hyperbolic Busemann learning with ideal prototypes*, NeurIPS (2021) — an ideal-point code, already built.

**Behavioural precedent.** Hyperbolic VR navigation studies (CHI PLAY 2019 and follow-ups): humans navigate hyperbolic environments without major disorientation, possibly handling branching structure more intuitively than in Euclidean space.

**Gap.** No published work trains an RNN to path-integrate in H² with curvature as a swept parameter, or asks what representation emerges.
