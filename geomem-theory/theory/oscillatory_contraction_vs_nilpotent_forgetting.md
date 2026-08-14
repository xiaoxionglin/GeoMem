---
tags: [geomem, theory, reservoirs, oscillatory, nilpotent, forgetting]
status: draft
updated: 2026-05-26
related:
  - "lin_block_advantage.md"
  - "../../geomem-experiments/experiments/rnn_navigation_benchmark.py"
---

# Oscillatory Contraction Versus Nilpotent Forgetting

## Core Point

The intuition is correct for standard echo-state or fading-memory reservoirs:

> A reusable oscillatory reservoir needs some contraction, damping, leak, saturation, noise, or active reset to forget initial conditions and old inputs. In the linear time-invariant case, this turns memory amplitude into an exponential decay. A nilpotent delay line instead has a finite hard horizon: information is preserved in addressable slots until it leaves the chain, then it is exactly gone.

So the important distinction is not "long memory versus short memory." Both oscillatory and nilpotent reservoirs can preserve information over long horizons. The distinction is the **forgetting curve**:

```text
contractive oscillatory reservoir: gradual exponential forgetting
nilpotent delay line: flat memory inside the window, sharp cutoff at the horizon
```

This matters for tasks where a cue must be remembered across a fixed blank interval while nearby distractors should not contaminate the decision.

## Linear Setup

Consider a linear reservoir driven by input `u_t`:

```text
h_t = A h_{t-1} + B u_t
```

Unrolling from the past:

```text
h_t = sum_{k >= 0} A^k B u_{t-k}
```

The lag-`k` input is represented by:

```text
phi_k = A^k B
```

A linear readout `c` reconstructs a weighted history:

```text
y_t = c* h_t = sum_{k >= 0} c* A^k B u_{t-k}
```

The scalar impulse response is:

```text
g_k = c* A^k B
```

The reservoir is useful for delayed recall at delay `D` when `g_D` is large and `g_k` is small for irrelevant lags.

## Contractive Oscillatory Reservoir

A pure oscillatory mode has eigenvalues on the unit circle:

```text
lambda_j = exp(i omega_j)
```

This preserves amplitude indefinitely. That sounds attractive, but it causes a problem for a driven reservoir: old inputs and initial conditions do not vanish. Without some reset, gating, nonlinearity, or damping, the reservoir is not a fading-memory system. In echo-state language, the current state may keep depending on arbitrarily old state/input components.

The standard stable oscillatory form adds contraction:

```text
lambda_j = r_j exp(i omega_j),  0 <= r_j < 1
```

Then:

```text
A^k mode_j = r_j^k exp(i omega_j k)
```

For a readout:

```text
g_k = sum_j alpha_j r_j^k exp(i omega_j k)
```

If all modes share `r`, the envelope is bounded by:

```text
|g_k| <= C r^k
```

Thus forgetting is exponential:

```text
memory amplitude at lag k ~= r^k = exp(k log r)
```

The memory time constant is:

```text
tau = -1 / log r
```

For `r` close to one:

```text
tau ~= 1 / (1-r)
```

This gives long memory, but not sharp memory. A cue at lag `D` and a distractor at lag `D+1` or `D-1` have similar amplitudes:

```text
r^(D+1) / r^D = r
r^(D-1) / r^D = 1/r
```

When `r` is close to one, adjacent lags are barely separated by amplitude. The readout must use phase interference across multiple frequencies to localize time. With limited modes, this produces side lobes and leakage.

## Nilpotent Delay Line

An ideal nilpotent delay line of length `L` has:

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

For readout `c = e_{D+1}`:

```text
g_k = c* A^k B
```

so:

```text
g_k = 1  if k = D
g_k = 0  if k != D, 0 <= k < L
g_k = 0  if k >= L
```

This gives exact lag addressing inside the window. The forgetting curve for the raw stored input is:

```text
stored amplitude = 1  for k < L
stored amplitude = 0  for k >= L
```

That is a sharp finite cutoff, not exponential decay.

## Consequence: Precision Versus Persistence

Oscillatory contraction is good for persistent smooth memory:

```text
old evidence gradually fades
phase carries temporal structure
```

Nilpotent delay memory is good for finite event indexing:

```text
event remains addressable at a specific slot
then disappears exactly at the horizon
```

The cost is different:

- Oscillatory memory can preserve information compactly over long times, especially for periodic or smooth signals.
- Nilpotent memory spends one slot per lag per feature, which is expensive but easy to decode.

## Does High Thresholding Make Oscillation Equivalent?

Not generally. A high threshold can make an oscillatory reservoir look more event-like at the readout, but it does not by itself turn phase memory into a nilpotent delay line.

There are three cases.

### 1. Undamped Oscillation Plus Threshold

For an undamped oscillator:

```text
x_k = A cos(omega k + phi)
```

a thresholded unit fires when:

```text
x_k > theta
```

This creates phase windows:

```text
cos(omega k + phi) > theta / A
```

The response can be sharp in phase, but it is periodic. The same threshold crossing returns every cycle. Therefore it does not forget:

```text
k = k_0, k_0 + T, k_0 + 2T, ...
```

This can be useful for cyclic variables, but it is bad for one-shot delayed cue memory because the cue can reappear as a phase alias.

### 2. Damped Oscillation Plus Threshold

For a damped oscillator:

```text
x_k = A r^k cos(omega k + phi),  0 < r < 1
```

thresholding gives:

```text
A r^k cos(omega k + phi) > theta
```

The threshold creates an apparent finite horizon because eventually:

```text
A r^k < theta
```

Solving for the last possible threshold crossing:

```text
k_max ~= log(theta / A) / log(r)
```

This looks like finite forgetting, but it is not the nilpotent kind. The cutoff depends on initial amplitude `A`, threshold `theta`, contraction `r`, phase `phi`, and noise. It is also not a clean lag coordinate: before the threshold is crossed, nearby lags are still represented by overlapping phases and amplitudes.

So thresholding can convert exponential decay into a nonlinear detection boundary, but the underlying memory is still exponentially damped.

### 3. Bank Of Oscillators Plus Thresholds

A bank of oscillators with different phases or frequencies can tile time. With thresholded readouts, it can approximate localized temporal windows:

```text
lag D -> subset of oscillators above threshold
```

This starts to resemble a temporal basis or radial-basis code. But it differs from a nilpotent delay line in two ways:

- phase codes are periodic unless there is damping, reset, or gating;
- finite frequency banks have finite temporal resolution and side-lobe leakage.

To make this behave like a true delay line, the system needs extra machinery: reset after trial start, monotonic elapsed-time code, gating, or enough nonperiodic modes to avoid phase aliasing. At that point, the useful mechanism is no longer "oscillation alone"; it is oscillation plus event gating or reset.

## Bottom Line On Thresholding

High thresholding can sharpen a decision boundary, but it does not automatically create sharp memory forgetting.

| Mechanism | Effect of threshold | Failure mode |
| --- | --- | --- |
| undamped oscillation | sharp phase windows | periodic aliasing, no forgetting |
| damped oscillation | apparent finite detection horizon | horizon depends on amplitude/noise; underlying decay is exponential |
| oscillator bank | approximate temporal windows | side lobes and phase aliasing unless many modes or resets |
| nilpotent delay line | true finite lag slots | hard cutoff; inefficient for smooth or periodic signals |

Thus a high-threshold oscillatory reservoir can approximate finite-window behavior in a narrow regime, but it does not function the same way as a nilpotent delay line unless it is augmented with reset/gating or enough basis functions to build a delay-coordinate code.

## Why Nilpotent Can Beat Oscillatory In A Cue-Gap Task

Consider:

```text
t = 0       cue_left or cue_right appears
t = 1..D    blank interval with possible distractors
t = D+1     choose left or right
```

The desired temporal filter is close to a delta function:

```text
g_k = 1 if k = D
g_k = 0 otherwise
```

Nilpotent delay line:

- implements this directly with one slot;
- no decay before the horizon;
- no contamination from adjacent lag slots if the readout is linear;
- sharp forgetting after the window.

Contractive oscillatory reservoir:

- stores the cue with amplitude `r^D`;
- stores distractors at nearby lags with similar amplitudes;
- requires a sum of oscillatory modes to synthesize a delta-like temporal filter;
- with few modes, the filter is broad and has side lobes;
- with strong contraction, the target cue decays too much;
- with weak contraction, distractors and old inputs persist too much.

This creates a tradeoff:

```text
large r: long memory, poor forgetting of distractors
small r: good forgetting, weak delayed cue
```

The nilpotent line avoids this particular tradeoff when the task delay is within its fixed span.

## A Simple Error Bound

Suppose the target cue is at lag `D`, and a distractor of comparable magnitude occurs at lag `D-delta`. In a single-radius oscillatory reservoir, their amplitude ratio before phase decoding is:

```text
target amplitude     = r^D
distractor amplitude = r^(D-delta)
ratio distractor/target = r^(-delta)
```

For small `delta` and `r` near one:

```text
r^(-delta) ~= exp(delta (1-r))
```

So the distractor is not strongly suppressed by contraction. Suppressing nearby distractors requires phase cancellation, not just damping.

With `m` oscillatory modes, the best time-localized filter behaves like a finite Fourier approximation to a delta function. Its temporal width is on the order of:

```text
width ~= O(1 / bandwidth)
```

and side-lobe leakage falls only as modes and frequency placement improve. This is the familiar Fourier localization tradeoff: compact frequency support limits temporal sharpness.

For a nilpotent line, adjacent lags are orthogonal:

```text
<e_{D+1}, e_{D-delta+1}> = 0
```

so distractor leakage is zero in the ideal linear case.

## Where Oscillatory Should Be Better

Oscillatory reservoirs should beat nilpotent delay lines when the task itself is phase-like:

- remembering heading phase or cyclic route phase;
- tracking rhythmic cue timing;
- computing periodic context;
- representing circular variables;
- maintaining information over indefinite horizons where a hard cutoff is harmful;
- using smooth temporal correlations rather than isolated sparse events.

In those cases, nilpotent's sharp cutoff is a weakness: it destroys information at `L`, and it represents periodic structure inefficiently.

## Implication For GeoMem

This gives a cleaner architecture prediction:

| Task temporal geometry | Better basis |
| --- | --- |
| finite sparse cue gap | nilpotent delay / Lin block |
| exact lag-specific recall | nilpotent delay / explicit history |
| periodic or phase context | oscillatory reservoir |
| slow context with gradual forgetting | diagonal or symmetric leaky modes |
| mixed stable, cyclic, and finite-span demands | block-hybrid reservoir |

For the current delayed-cue benchmark, nilpotent Lin blocks are expected to beat oscillatory reservoirs because the task demands a sharp finite temporal selector, not a phase code.

## Implemented Control Benchmark

`../../geomem-experiments/experiments/temporal_basis_benchmark.py` now tests this distinction directly.

The results are calibrated rather than one-sided:

- Nilpotent delay lines show the predicted sharp span boundary. `L32` and `L64` are at ceiling inside their spans and near chance immediately outside.
- A large damped oscillatory bank can still solve delayed-cue recall. `osc_r099_m48` reaches `1.000` at delay `64` and `0.918` at delay `96`, showing that enough Fourier-like modes can approximate a lag selector.
- Thresholding is not equivalent to delay memory. In the current run, the thresholded oscillatory bank loses useful long-delay information and performs near chance at long delays.
- A matched two-dimensional phase oscillator is the most efficient phase-memory basis, reaching cosine alignment `0.9999` on the modulo-phase task.

So the refined claim is:

> Nilpotent is better when the task demands direct finite lag addressing under limited basis size. Oscillation is better when the task variable is phase-like. Large enough bases can approximate each other, but at increased cost.

## References

- [Jaeger, H. (2001/2002). Echo-state networks and short-term memory capacity](https://www.scholarpedia.org/article/Echo_state_network). ESNs motivate the echo-state/fading-memory constraint and memory-capacity framing.
- [Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training](https://www.sciencedirect.com/science/article/pii/S1574013709000173). General reservoir-computing review, including spectral-radius and reservoir-design issues.
- [White, O. L., Lee, D. D., & Sompolinsky, H. (2004). Short-term memory in orthogonal neural networks](https://pubmed.ncbi.nlm.nih.gov/15089576/). Relevant for long memory in norm-preserving recurrent dynamics.
- [Ganguli, S., Huh, D., & Sompolinsky, H. (2008). Memory traces in dynamical systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC2596211/). Useful for thinking about how dynamical systems distribute memory over modes and time.
- [Dambre, J., Verstraeten, D., Schrauwen, B., & Massar, S. (2012). Information processing capacity of dynamical systems](https://www.nature.com/articles/srep00514). Frames reservoir memory as finite capacity allocation across functions of past inputs.
