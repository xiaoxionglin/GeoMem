---
tags: [geomem, theory, lin-block, reservoirs, state-space-models, sequence-memory]
status: draft
updated: 2026-05-26
related:
  - "oscillatory_contraction_vs_nilpotent_forgetting.md"
  - "../../geomem-experiments/experiments/rnn_navigation_benchmark.py"
  - "../../geomem-paper/reports/iteration_011_long_lin_blocks.md"
  - "../../geomem-paper/reports/iteration_012_long_architecture_comparison.md"
  - "../../geomem-paper/reports/iteration_013_systematic_paper_sweeps.md"
---

# Lin-Block Advantage Over Generic Reservoirs And SSMs

## Summary

The current `lin_block` architecture should be interpreted as a deliberately structured state-space memory, not as a generic reservoir. Its advantage is not "nilpotent dynamics are always better." The advantage is more specific:

> If the task-relevant residual variable is a sparse feature that must be carried across a known finite delay, then a per-feature nilpotent chain gives the linear readout a well-conditioned delay-coordinate basis. Generic reservoirs may preserve the same information, but they spend capacity on mixed, fading, oscillatory, or nonlinear temporal bases that are not necessarily aligned with exact cue transport.

This note connects the current benchmark to reservoir computing and modern state-space-model literature. It also states where the current argument is still weak.

## Literature Anchors

Reservoir computing treats a fixed recurrent system as a dynamic feature map and trains mainly the readout. Echo State Networks and Liquid State Machines established this readout-training paradigm: the reservoir maps input history into a high-dimensional transient state, and a linear readout extracts the task variable. The key property is not recurrence alone, but a useful fading-memory representation of recent input history.

Short-term memory capacity work gives the first constraint. For a linear echo-state network with scalar i.i.d. input, total linear memory capacity is bounded by the number of reservoir units `N`, and orthogonal or well-conditioned linear reservoirs can approach that bound. This means a reservoir cannot store arbitrary lagged input history for free; architecture decides how finite capacity is distributed over lags, features, and nonlinear combinations.

The broader information-processing-capacity framework generalizes this idea: a dynamical system has finite capacity for reconstructing functions of past inputs, and capacity allocated to one family of functions cannot simultaneously be allocated elsewhere without tradeoff. In GeoMem terms, the relevant question is not "which reservoir has memory?" but "which temporal basis allocates capacity to the task-relevant quotient?"

Modern state-space models make this basis question explicit. LMU and HiPPO derive recurrent memories from online projection of input history onto structured polynomial bases. S4 uses structured state-space parameterizations to model long sequences efficiently. Mamba adds input-dependent selectivity, addressing a weakness of time-invariant SSMs on content-dependent discrete sequences. These models clarify the design axis that the current benchmark only touches: memory systems differ by the temporal basis, compression objective, and input-dependent gating they implement.

## Formal Lens

Ignore the pointwise nonlinearity for a moment. A fixed recurrent memory is a state-space map:

```text
h_t = A h_{t-1} + B u_t
y_t = C h_t
```

Unrolling gives:

```text
h_t = B u_t + A B u_{t-1} + A^2 B u_{t-2} + ... + A^k B u_{t-k} + ...
```

The columns of `A^k B` define the temporal-feature basis available to the readout. The difference between `lin_block`, random reservoirs, symmetric reservoirs, oscillatory reservoirs, and SSMs is therefore a difference in basis design.

For a task requiring the cue `u_{t-D,i}` at a later decision, the readout needs a direction in state space that isolates feature `i` at lag `D`. The quality of that direction depends on:

- controllability: whether input feature `i` reaches enough state dimensions across lags;
- observability/readout conditioning: whether the lagged feature can be linearly decoded from `h_t`;
- interference: whether other features and lags occupy overlapping state directions;
- horizon: whether the relevant lag has decayed, rotated, saturated, or been overwritten;
- resource cost: how many state dimensions are used to preserve the needed variable.

The `lin_block` architecture wins only when its delay-line basis is a good match to this decoding problem.

## What `lin_block` Implements

In the current code, `lin_block` creates one finite chain per input feature:

```text
feature i at time t -> chain_i[0:R]
chain_i[j] -> chain_i[j+1]
```

with span:

```text
ell = R + L - 1
hidden_dim = input_dim * ell
```

This is a block-diagonal nilpotent state-space model. Each feature gets its own finite impulse-response memory. With `R = 1` and a linear activation, this is close to an explicit delay line. With `R > 1`, the input is written into several early chain positions, producing a small distributed temporal smear that can improve robustness but reduces exact lag purity.

The important property is separability:

```text
state ~= [feature_1 lags, feature_2 lags, ..., feature_d lags]
```

The readout can then learn simple rules such as:

```text
if cue_left was active at lag D, choose left
if cue_right was active at lag D, choose right
```

without first undoing random feature mixing.

## Comparison To Reservoir Families

### Modal View: Symmetric, Orthogonal, And Diagonal Reservoirs

From a dynamical-systems point of view, symmetric, orthogonal, and diagonal reservoirs are closely related in one important sense: they are mostly modal memories. A normal linear system can be decomposed into independent modes:

```text
A = Q Lambda Q*
h_t = sum_k A^k B u_{t-k}
```

The reservoir differs mainly by the eigenvalue spectrum `Lambda` and by how the input projection `B` excites those modes.

- A diagonal reservoir is already in its modal basis.
- A symmetric reservoir has an orthogonal eigenbasis with real eigenvalues, so it is a rotated diagonal reservoir.
- An orthogonal reservoir has eigenvalues on the unit circle, so it is a rotated mixture of sign flips and complex rotations.

So yes: at this abstraction level, these reservoirs are variants of "store history in independent modes." They differ less by qualitative architecture than by spectrum:

```text
symmetric/diagonal: real decays or sign-changing real modes
orthogonal: norm-preserving rotations/reflections
oscillatory: explicitly rotation-dominated complex modes
```

This is useful for the paper because it prevents overinterpreting small performance differences among modal reservoirs. The deeper contrast is not symmetric versus diagonal; it is modal compression versus addressable delay-line memory.

### Random Reservoir

A random reservoir uses random `A` and random dense `B`. It can create rich nonlinear mixed features and can work well when the readout only needs some generic projection of recent history. But exact delayed-feature recall is not guaranteed. The cue, distractors, and irrelevant sensory dimensions are all mixed into the same recurrent basis.

The random reservoir's advantage is breadth: it may support many unknown functions of recent history. Its weakness is allocation: unless `N` is large enough and the random basis is well conditioned, the task-relevant lag-feature direction may be weak or entangled.

For GeoMem, this is a general-purpose memory prior. `lin_block` is a more specialized prior.

### Symmetric Reservoir

A symmetric recurrent matrix has real eigenmodes. These modes behave more like stable or unstable directions than like ordered delay coordinates. With spectral radius below one, the state is dominated by decaying projections onto eigenvectors.

This can be useful for denoising, slow context persistence, and attractor-like state estimation. It is less naturally suited to "what happened exactly `D` steps ago?" because real modes do not provide an ordered transport mechanism. A symmetric reservoir may remember that a cue occurred, but the lag identity is encoded through a mixture of decay amplitudes, which can be ill-conditioned for precise temporal localization.

### Oscillatory Reservoir

An oscillatory or skew-symmetric reservoir gives complex modes and rotating state trajectories. This resembles a Fourier-like temporal basis. It can be useful when the task has phase, rhythm, periodic structure, or cyclic temporal context.

For sparse cue-gap tasks, oscillatory memory is a mismatch unless the readout can combine enough frequencies to reconstruct a localized pulse. That requires many modes. With limited capacity, oscillatory bases tend to spread a cue across phase dimensions rather than preserving a clean delay slot.

Oscillatory and nilpotent reservoirs are similar in one important respect: both can carry information over long horizons without ordinary exponential decay. The difference is how the information is indexed.

- Oscillatory memory preserves information as phase distributed across modes.
- Nilpotent delay memory preserves information as position along a finite chain.

For a control task that asks "which cue occurred `D` steps ago?", position-coded memory is often easier for a linear readout than phase-coded memory.

### Orthogonal Reservoir

An orthogonal reservoir can preserve norm and support long memory better than a strongly contractive random reservoir. This connects to memory-capacity results showing that orthogonal or well-conditioned linear dynamics can approach the linear capacity bound.

However, preservation is not the same as accessibility. If inputs are randomly mixed into orthogonal rotations, lagged features remain in the state but are represented in rotated superpositions. A linear readout can recover them only if the basis is sufficiently large, well conditioned, and not damaged by nonlinear saturation. Orthogonal reservoirs are strong candidates for high-dimensional long memory, but they are not automatically better for sparse finite cue transport.

### Diagonal Reservoir

A diagonal reservoir implements independent leaky traces with different decay rates. This is a Laplace-like or exponential basis over history. It is efficient for smooth histories, slow context, and multi-timescale integration.

Its weakness is temporal precision. Reconstructing a sharply localized lag from exponentials is possible in principle but can be numerically ill-conditioned with few units. For delayed T-maze-style tasks, a diagonal basis may retain "cue evidence existed recently" while failing to isolate the needed cue at the relevant delay if distractors or repeated cues are present.

### Generic Nilpotent Reservoir

The generic nilpotent reservoir is closest to `lin_block` because `A` is a shift matrix. The difference is input organization. In the current benchmark, generic nilpotent uses one fixed-size shift chain and a random dense input projection:

```text
u_t -> random mixture -> one shift chain
```

This stores a history of random mixtures, not independent feature histories. With enough units and a suitable `B`, it can approximate delay memory. But with fixed `N = 32`, multiple input features and lags compete for the same chain. The independent Lin block removes that competition by allocating a separate chain to each feature:

```text
u_t[i] -> chain_i
```

So the current `lin_block` advantage over `reservoir_nilpotent` is not the shift itself. It is the combination of shift dynamics, feature-specific input routing, and larger span-dependent state size.

## Analytic Comparison: Oscillatory Versus Nilpotent Delay Memory

Consider a scalar input stream `u_t` and a linear reservoir:

```text
h_t = A h_{t-1} + B u_t
h_t = B u_t + A B u_{t-1} + A^2 B u_{t-2} + ...
```

To recover the input from exactly `D` steps ago, a linear readout `c` must satisfy:

```text
c* A^D B = 1
c* A^k B = 0  for irrelevant lags k
```

The vectors `phi_k = A^k B` are the lag codes. The task is easy when `phi_D` is linearly separable from the other lag codes.

### Nilpotent Delay Line

For an ideal delay line of length `L`, let:

```text
A e_j = e_{j+1},  j < L
A e_L = 0
B = e_1
```

Then:

```text
A^k B = e_{k+1}  for 0 <= k < L
A^k B = 0        for k >= L
```

The lag codes are orthogonal coordinate vectors. If `D < L`, choose:

```text
c = e_{D+1}
```

Then:

```text
c* A^D B = 1
c* A^k B = 0 for k != D, 0 <= k < L
```

So delayed recall is exact inside the horizon and impossible outside it. This is a hard finite window: no decay inside the window, no memory after the cutoff.

### Oscillatory Basis

For an oscillatory reservoir, use complex modes:

```text
A = diag(exp(i omega_1), ..., exp(i omega_m))
B = b
```

Then:

```text
phi_k = A^k B
phi_k[j] = b_j exp(i omega_j k)
```

The inner product between two lag codes is:

```text
<phi_k, phi_l> = sum_j |b_j|^2 exp(i omega_j (k-l))
```

Thus lag separability depends on the Fourier kernel induced by the selected frequencies. If the frequencies are uniformly spread and `|b_j|` is balanced, the normalized cross-lag similarity behaves like a Dirichlet kernel:

```text
similarity(delta) ~= sin(m Delta_omega delta / 2) / (m sin(Delta_omega delta / 2))
```

This means lag codes are not generally orthogonal. Nearby lags can have substantial overlap unless the reservoir has enough well-spaced modes. A readout can reconstruct a delta-like lag selector only by combining many oscillatory modes. With limited `m`, the selector has side lobes: events at neighboring lags leak into the estimate of the target lag.

### Consequence

Both systems can preserve information, but they preserve different coordinates:

```text
nilpotent delay line:  lag -> spatial slot
oscillatory reservoir: lag -> phase pattern
```

For exact delayed cue recall with distractors, nilpotent is better because it gives a local basis. Oscillatory memory is better when the task variable is phase-like, periodic, or smoothly distributed over time.

## Intuitive Scenario Where Nilpotent Beats Oscillatory

Use a delayed cue task with sparse, nonperiodic events:

```text
t = 0      cue_left or cue_right appears once
t = 1..D   blank corridor, possible irrelevant flickers
t = D+1    choose left or right
```

The relevant variable is a discrete event at a particular earlier time. A nilpotent delay line stores the cue in a known slot at decision time. A linear readout can inspect that slot directly.

An oscillatory reservoir also carries the cue, but as a phase pattern distributed across modes. If irrelevant flickers occur during the corridor, each flicker creates another phase pattern. The decision readout must synthesize a narrow temporal filter that selects the cue lag while rejecting neighboring lags. With few modes, this filter has broad side lobes, so distractors leak into the decision.

Therefore nilpotent should beat oscillatory when:

- cue identity is sparse and event-like;
- the delay is finite and known;
- the correct readout needs lag-specific recall;
- distractors or irrelevant observations occur at nearby lags;
- the readout is linear and data-limited;
- the reservoir capacity is not large enough to synthesize a sharp Fourier-like temporal selector.

Oscillatory should beat nilpotent when:

- the task is periodic or phase-based;
- temporal position is circular rather than finite;
- the relevant variable is rhythm, heading phase, theta phase, or cyclic context;
- long preservation matters more than exact lag addressability;
- the readout can use smooth phase structure rather than a localized event slot.

### Block-Hybrid Reservoir

The block-hybrid reservoir combines several dynamics, such as symmetric, oscillatory, and nilpotent subblocks. This is closer in spirit to the GeoMem hybrid thesis. It can hedge across task families: stable context, cyclic dynamics, and finite sequence traces all receive some capacity.

The tradeoff is dilution. If the task is a pure sparse cue-gap task, only the sequence subblock may be useful. If the hidden size is fixed, hybridization can reduce the dimensions available for the exact delayed cue.

## Comparison To Modern State-Space Models

### LMU And HiPPO

LMU and HiPPO are important because they make the memory objective explicit. Instead of using arbitrary random recurrence, they derive dynamics that maintain coefficients of an online projection of the input history. LMU uses a Legendre basis over a sliding window; HiPPO generalizes this to online polynomial projections under different measures.

Relative to these models, `lin_block` is a crude basis: it uses delay coordinates rather than an orthogonal polynomial compression. That is inefficient for smooth signals, because a polynomial basis can represent a smooth history with far fewer dimensions than storing every feature-lag pair.

But delay coordinates can be better for sparse symbolic events. If the task depends on whether a particular cue appeared at a particular lag, an explicit delay-line basis is direct and linearly decodable. Polynomial bases compress the history, but exact sparse-event recovery may require enough basis terms and can introduce ringing or distributed decoding.

This gives a sharper hypothesis:

> `lin_block` should outperform polynomial SSM memories on sparse event transport at matched decoding simplicity, but LMU/HiPPO-style memories should outperform `lin_block` on smooth trajectory reconstruction or low-dimensional continuous histories at matched state size.

### S4

S4 uses structured state-space parameterizations to model long-range dependencies efficiently. It is best viewed as a learned or designed long-convolution kernel family with efficient recurrent and convolutional forms. Compared with `lin_block`, S4 is far more expressive and resource-efficient for long continuous sequences.

The relevant distinction is control of the kernel. `lin_block` hard-codes a finite rectangular delay basis. S4 learns structured kernels that can span long horizons and mix timescales. Therefore S4-like models are a better baseline if the claim is about scalable sequence modeling. `lin_block` remains useful as an interpretable mechanistic probe: it tells us whether the task can be solved by carrying sparse evidence across a finite span.

### Mamba And Selective SSMs

Mamba identifies a limitation of time-invariant SSMs: they do not easily perform content-based selective memory. Mamba makes SSM parameters input-dependent, allowing the model to decide what to propagate or forget based on the current token.

This is directly relevant to GeoMem. Real navigation is not just "store the last `D` inputs." The agent should store cues, goals, branches, and context switches selectively while ignoring irrelevant sensory variation. A fixed `lin_block` cannot do that. It stores everything in its span. That is useful for controlled sparse-cue experiments but inefficient in rich visual streams.

For the biological story, the stronger architecture would be:

```text
DG-like sparsifying front end -> selective sequence/state-space memory -> task readout or policy
```

The current `lin_block` approximates only the middle transport mechanism after sparsification. It does not implement content-dependent gating.

## Deeper Theoretical Claims

### Claim 1: The Advantage Is Basis Alignment

The best recurrent memory is the one whose temporal basis matches the task-relevant quotient. `lin_block` aligns with finite delayed sparse-event recall. Diagonal reservoirs align with exponential integration. Oscillatory reservoirs align with phase. Symmetric reservoirs align with stable context. Random reservoirs hedge across possible functions. SSMs such as HiPPO/S4 align with compressed continuous histories.

This is the architecture-level version of the GeoMem thesis.

### Claim 2: Delay-Line Memory Trades Compression For Decodability

`lin_block` spends `O(input_dim * delay)` dimensions to make delayed features easy to decode. This is expensive but robust. Polynomial, diagonal, orthogonal, and random reservoirs may compress more aggressively, but decoding the exact cue-lag variable can become harder.

This is not a free win. It is a cost-decoding tradeoff:

```text
lin_block: high dimension, simple decoding
compressed SSM/reservoir: lower dimension, harder or task-dependent decoding
```

### Claim 3: Nilpotence Alone Is Not The Mechanism

Nilpotence gives finite memory. It does not by itself give feature-specific memory. The generic nilpotent reservoir and independent Lin block differ in the controllability matrix:

```text
generic nilpotent: [B, AB, A^2B, ...] with random dense B into one chain
lin_block: block-structured [B, AB, A^2B, ...] preserving feature identity
```

So any paper claim should avoid saying "nilpotent reservoirs solve the task." The accurate claim is "feature-routed nilpotent chains provide an interpretable finite-span delayed-feature basis."

### Claim 4: The Current Benchmark Is A Span Test, Not A Universal Architecture Test

The long delayed T-maze is close to a pure span test. It asks whether the architecture carries a sparse cue across a blank interval. That is exactly the condition under which `lin_block` should look strong.

This benchmark does not test:

- rich visual denoising;
- continuous path integration;
- content-dependent forgetting;
- nonlinear relational inference;
- long-horizon planning;
- learning the memory basis from reward.

Therefore the theory should use the benchmark to support a narrow mechanistic claim, then propose broader tests.

## Predictions For New Experiments

### Capacity-Matched Tests

If `reservoir_nilpotent` is given the same hidden size as `lin_block` but retains random dense input projection, it should improve but may still underperform on feature-specific sparse-cue tasks because feature identity remains randomly mixed.

If `reservoir_nilpotent` is given both the same hidden size and feature-structured input projection, it should approach `lin_block`. That would confirm that the advantage comes from basis structure rather than from the name "Lin."

### Basis-Matched Tests

Diagonal reservoirs should improve when the target is a slow latent context variable rather than a precise lagged cue.

Oscillatory reservoirs should improve when the task requires phase or periodic route context.

Orthogonal reservoirs should improve when the task requires long preservation of dense high-dimensional signals, provided the readout has enough data and regularization.

Symmetric reservoirs should improve in denoising or attractor-like context tasks where stable state estimation matters more than ordered recall.

LMU/HiPPO-style memories should outperform `lin_block` on smooth continuous trajectory reconstruction under tight state budgets.

Mamba-style selective SSMs should outperform fixed `lin_block` in rich observation streams where only some events should be stored.

### Geometry-Observation Interaction

`lin_block` should matter most when:

- the current observation is sparse or blank;
- the relevant cue occurred within a finite known span;
- the cue is already sparsified or denoised;
- the policy depends on cue identity more than metric integration;
- distractor history is limited enough that storing everything is not too costly.

It should matter less when:

- dense observations expose the current latent state;
- path integration supplies the needed metric variable;
- the relevant memory is smooth and low-dimensional;
- the task requires selective update rather than passive storage;
- the needed dependency exceeds the fixed span.

## Paper-Facing Wording

Use this:

> Independent Lin blocks implement a feature-routed finite impulse-response memory. Their advantage over generic reservoirs in sparse delayed-cue tasks is expected from basis alignment: the task asks for a sparse event to be transported across a finite blank interval, and the Lin block exposes feature-specific delayed coordinates to a linear readout. This does not imply that Lin blocks are universally superior; random, orthogonal, diagonal, oscillatory, polynomial, and selective SSM memories allocate capacity to different temporal bases and should win in different task geometries.

Avoid this:

> Lin blocks are the best reservoir architecture.

Also avoid this:

> Nilpotent reservoirs explain the result.

The better statement is:

> Feature-routed nilpotent chains explain the result under sparse finite-delay cue transport.

## Missing Work Before Strong Claims

The current theory still needs the following controls before making broad claims:

- capacity-matched generic reservoirs;
- feature-routed nilpotent controls without Lin-specific terminology;
- LMU/HiPPO/S4-style baselines for continuous-history tasks;
- selective SSM or gated sequence baselines for rich sensory streams;
- explicit memory-capacity curves for each architecture on the benchmark inputs;
- readout-conditioning analysis of the controllability matrix `[B, AB, A^2B, ...]`;
- cost model separating hidden units, parameters, energy, and biological plausibility.

Until then, the claim should remain:

> `lin_block` is a strong mechanistic probe for finite-span sparse-event memory, and its advantage is predicted by temporal-basis alignment.

## References To Read

- [Jaeger, H. (2001/2002). The echo-state approach and short-term memory capacity](https://www.scholarpedia.org/article/Echo_state_network). The key result for this project is the memory-capacity framing: finite recurrent state allocates finite capacity across delayed inputs.
- [Maass, W., Natschlager, T., & Markram, H. (2002). Real-time computing without stable states](https://pubmed.ncbi.nlm.nih.gov/12433288/). This is the Liquid State Machine foundation: transient dynamics can support computation without discrete stable states.
- [Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training](https://www.sciencedirect.com/science/article/pii/S1574013709000173). Useful as the general reservoir-computing review.
- [White, O. L., Lee, D. D., & Sompolinsky, H. (2004). Short-term memory in orthogonal neural networks](https://pubmed.ncbi.nlm.nih.gov/15089576/). Important for the claim that orthogonal/well-conditioned linear dynamics can approach memory-capacity bounds.
- [Ganguli, S., Huh, D., & Sompolinsky, H. (2008). Memory traces in dynamical systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC2596211/). Relevant for understanding how network dynamics distribute memory over time and modes.
- [Dambre, J., Verstraeten, D., Schrauwen, B., & Massar, S. (2012). Information processing capacity of dynamical systems](https://www.nature.com/articles/srep00514). Provides the broader capacity-allocation view beyond simple linear delay recall.
- [Rodan, A., & Tino, P. (2011). Minimum complexity echo state network](https://pubmed.ncbi.nlm.nih.gov/21075721/). Relevant because simple deterministic reservoirs can compete with random ESNs, supporting the claim that structure can matter more than randomness.
- [Voelker, A. R., Kajić, I., & Eliasmith, C. (2019). Legendre Memory Units](https://papers.nips.cc/paper/9689-legendre-memory-unit). A principled continuous-time sliding-window memory using an orthogonal polynomial basis.
- [Gu, A., Dao, T., Ermon, S., Rudra, A., & Re, C. (2020). HiPPO](https://arxiv.org/abs/2008.07669). General framework for online projection of history onto polynomial bases.
- [Gu, A., Goel, K., & Re, C. (2022). S4](https://openreview.net/pdf?id=uYLFoz1vlAC). Structured state-space sequence model for long-range dependencies.
- [Gu, A., & Dao, T. (2023/2024). Mamba](https://arxiv.org/abs/2312.00752). Selective state-space model; relevant because fixed SSMs lack content-dependent memory selection.
