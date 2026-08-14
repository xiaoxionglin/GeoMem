# geomem-env API Reference

This document is for another Codex thread that needs to use `geomem-env` safely without rediscovering the package shape.

## Import Setup

From inside `geomem-env`:

```bash
PYTHONPATH=src python3 your_script.py
```

From the sibling `geomem-experiments` repo:

```bash
PYTHONPATH=../geomem-env/src:experiments python3 experiments/your_script.py
```

Use public imports from `geomem_env`, not from `geomem_env.environment`:

```python
from geomem_env import default_environments, effective_commutativity
```

Experiment scripts that still import `commutativity_probe` are supported by a compatibility wrapper in `../geomem-experiments/experiments/commutativity_probe.py`, but new code should import from `geomem_env`.

## Core Types

```python
State = tuple[int, ...]
Action = str
Transition = dict[tuple[State, Action], State]
Observation = Hashable
ObservationFn = Callable[[State], Observation]
Policy = frozenset[str]
```

`Environment` is the central deterministic, finite-state transition system:

```python
env.name              # str
env.states            # tuple[State, ...]
env.actions           # tuple[Action, ...]
env.transition        # dict[(state, action), next_state]
env.valid             # frozenset[(state, action)]
env.max_depth         # int | None
env.metadata          # EnvironmentMetadata
env.start_states      # tuple[State, ...]
env.goal_states       # tuple[State, ...]
```

Methods:

```python
env.is_valid(state, action)       # bool
env.valid_actions(state)          # tuple[Action, ...]
env.step(state, action)           # State; invalid/missing transitions self-loop
env.rollout(start, actions)       # (final_state, all_actions_valid)
env.reset(seed=None, options=None) # (state, info), stateless Gymnasium-shaped helper
env.step_result(state, action)    # StepResult(state, reward, terminated, truncated, valid, info)
```

Important: `Environment` is stateless. `step_result()` does not mutate internal state. If an RL library needs a mutable environment, wrap this API.

## Constructors

Locally metric and graph families:

```python
torus_lattice(width=5, height=5)
clamped_lattice(width=5, height=5)
lattice(width=5, height=5)          # alias for clamped_lattice
tree(branching=3, depth=4)
shortcut_tree(branching=3, depth=4, shortcut_density=1.0)
bottleneck_rooms(size=3)
room_graph_environment("loop_rich" | "bottleneck" | "tree_like", size=3)
room_graph_family(size=3)
```

Behavioral and aliased navigation families:

```python
aliased_rooms(size=3)
aliased_loop_rooms(size=2, rooms=3)
aliased_t_maze(corridor_length=8)
landmark_gap_detour()
shortcut_route_choice()
behavioral_navigation_family()
```

Trace-monoid/formal families:

```python
trace_monoid(commuting_pairs, max_depth=8)
trace_monoid_family(max_depth=8)
trace_monoid_general(action_count, commuting_pairs, max_depth, relation_id=0)
trace_env(commuting_pairs, name_suffix, max_depth=6)  # compatibility naming helper
```

Example:

```python
from geomem_env import trace_monoid

env = trace_monoid(frozenset({("A", "B")}), max_depth=6)
print(env.name)  # trace_commute_033
```

## Registry and Metadata

Use the registry for standard taxonomy coverage:

```python
from geomem_env import default_environments, registered_environment_specs, environment_summary

envs = default_environments()
specs = registered_environment_specs()
summary = environment_summary(envs[0])
```

`EnvironmentMetadata` fields:

```python
env.metadata.family
env.metadata.geometry
env.metadata.observation
env.metadata.memory_pressure
dict(env.metadata.axes)
env.metadata.tags
```

Families currently include:

- `lattice`
- `tree`
- `shortcut_tree`
- `rooms`
- `room_graph`
- `aliased_navigation`
- `cue_gap`
- `route_choice`
- `trace_monoid`

## Validation and Graph Helpers

```python
validate_environment(env)            # list[str], empty means valid
graph_edges(env, valid_only=True)    # list[(state, action, next_state)]
reversible_navigation_graph(env)     # dict[state, list[(action_or_undo, neighbor)]]
default_goal_state(env)              # deterministic default goal
distances_to_goal(env, goal)         # shortest-path distances on reversible graph
optimal_policy_sets(env, goal=None)  # state -> optimal action set
```

Always run `validate_environment(env)` after adding a new constructor.

## Observation Helpers

Trace states:

```python
observe_trace_state(state, "full", n_actions=len(env.actions))
observe_trace_state(state, "counts", n_actions=len(env.actions))
observe_trace_state(state, "depth_last", n_actions=len(env.actions))
observe_trace_state(state, "depth", n_actions=len(env.actions))
```

Room states shaped as `(room, x, y)`:

```python
observe_room_state(state, "full")
observe_room_state(state, "room")
observe_room_state(state, "local_position")
observe_room_state(state, "room_local_position")
observe_room_state(state, "boundary_signature")
```

Behavioral/cue-gap states:

```python
observe_behavior_state(state, env.name, "full")
observe_behavior_state(state, env.name, "cue_only")
observe_behavior_state(state, env.name, "stage_only")
observe_behavior_state(state, env.name, "local_symbol")
observe_behavior_state(state, env.name, "cue_then_blank")
```

Generic summaries:

```python
action_counts(state, n_actions=None)
sparse_observation_symbol(state, env.name)
numeric_state_observation(state, width=6)
local_view_observation(env, state)
standard_observation_specs(env)
```

`standard_observation_specs(env)` returns `ObservationSpec` objects with `.name`, `.observe`, and `.description`.

## Sensory Observation Channels

Use channels when observations are sampled sensory cues rather than exact state summaries. Channels are seeded and stateless, so repeated calls with the same state and seed are reproducible.

```python
from geomem_env import (
    egocentric_noisy_landmark_channel,
    gaussian_local_state_channel,
    gaussian_sensory_channel,
    sparse_visual_feature_channel,
)

env = room_graph_environment("bottleneck")
gaussian = gaussian_sensory_channel(env, sigma=0.1, dim=8, seed=19)
local = gaussian_local_state_channel(env, sigma=0.1, width=6)
sparse = sparse_visual_feature_channel(env, n_features=6, alias_rooms=True, dropout_p=0.1)
ego = egocentric_noisy_landmark_channel(env, sigma=0.05, n_landmarks=6)

obs = gaussian.observe(env.states[0], seed=17)
```

Channel types:

- `gaussian_sensory_channel(env, sigma=0.1, dim=8, seed=0)`: default realistic channel; aliased current-state sensory prototype plus Gaussian noise, with no action bits or history.
- `gaussian_local_state_channel(env, sigma, width=6)`: normalized coordinate-like vector plus independent Gaussian noise.
- `sparse_visual_feature_channel(env, n_features=6, alias_rooms=True, dropout_p=0.0)`: sparse visual feature vector that can intentionally alias rooms/states.
- `egocentric_noisy_landmark_channel(env, sigma, n_landmarks=6)`: valid-action affordance bits plus noisy landmark features; use as a control, not the default realistic observation.

The environment does not emit learned action embeddings or action histories. For model rollouts, pass the previous action as a separate model-side token:

```text
model input at t: [sensory observation o_t, previous action token a_{t-1}]
model output at t: action a_t
environment input: a_t
next model input: [sensory observation o_{t+1}, previous action token a_t]
```

`ObservationChannel` fields:

```python
channel.name
channel.kind
channel.dim
channel.observe(state, seed=None)
dict(channel.metadata)
```

## Diagnostics

Action geometry:

```python
effective_commutativity(env, length=4, n_sequences=200, seed=1, valid_only=False)
pairwise_commutator_matrix(env, valid_only=False)
```

Observation and control sufficiency:

```python
observation_ambiguity(env, observe)
transition_ambiguity(env, observe)
policy_ambiguity(env, observe, goal=None)
residual_history_diagnostics(env, observe, goal=None)
```

Sampled diagnostics for stochastic channels:

```python
sampled_observation_ambiguity(env, channel, n_samples=3, seed=1, bin_width=0.25)
sampled_transition_ambiguity(env, channel, n_samples=3, seed=1, bin_width=0.25)
sampled_policy_ambiguity(env, channel, n_samples=3, seed=1, bin_width=0.25)
sampled_residual_history_diagnostics(env, channel, n_samples=3, seed=1, bin_width=0.25)
```

Gaussian observations are continuous, so sampled diagnostics quantize observations with `quantized_observation(..., bin_width=...)` before estimating ambiguity.

Standard tables:

```python
standard_sensory_diagnostic_table(envs=None, sigma=0.1, dim=8, n_samples=3)
standard_diagnostic_table(envs=None)
trace_diagnostic_table(envs=None, modes=("full", "counts", "depth_last", "depth"))
architecture_expectations(residual_history_score)
```

`standard_sensory_diagnostic_table()` is the Gaussian-first default. `standard_diagnostic_table()` and `trace_diagnostic_table()` are exact formal/control diagnostics over symbolic observation summaries.

Typical trace diagnostic:

```python
from geomem_env import trace_monoid_family, observe_trace_state, transition_ambiguity

env = trace_monoid_family(max_depth=8)[0]
observe = lambda state: observe_trace_state(state, "counts", n_actions=len(env.actions))
print(transition_ambiguity(env, observe)["weighted_ambiguous_fraction"])
```

Typical taxonomy diagnostic:

```python
from geomem_env import default_environments, standard_diagnostic_table

rows = standard_diagnostic_table(default_environments())
print(rows[0].keys())
```

## Diagnostic Return Keys

`observation_ambiguity()`:

- `n_states`
- `n_observations`
- `mean_candidates`
- `max_candidates`
- `normalized_ambiguity`
- `retained_entropy_fraction`

`transition_ambiguity()`:

- `mean_next_candidates`
- `max_next_candidates`
- `ambiguous_fraction`
- `weighted_ambiguous_fraction`

`policy_ambiguity()`:

- `goal_depth`
- `n_policy_states`
- `policy_ambiguous_observation_fraction`
- `weighted_policy_ambiguity`
- `mean_policy_set_size`

`residual_history_diagnostics()`:

- `transition_residual`
- `policy_residual`
- `residual_history_score`

`architecture_expectations()`:

- `vector_count`
- `finite_history`
- `sequence`
- `reservoir`
- `hybrid`

## Common Pitfalls

- Do not import from sibling experiment modules for environment definitions. Import from `geomem_env`.
- `State` is always a tuple of integers. Do not use lists as states; they are not hashable.
- Invalid actions usually self-loop through `env.step()`. Use `env.is_valid()` or `env.rollout(...)[1]` when validity matters.
- `valid_only=True` in commutativity diagnostics filters invalid paths; raw commutativity includes boundary and protocol effects.
- Do not use exact `observation_ambiguity()` for continuous noisy channels. Use sampled diagnostics with an explicit `bin_width`.
- `observe_room_state()` expects 3-tuples. Use `observe_behavior_state()` for T-maze, detour, and route-choice tasks.
- `policy_ambiguity()` uses a deterministic default goal if none is supplied. For paper-specific experiments, pass an explicit goal.
- `standard_diagnostic_table()` can be moderately expensive because it evaluates every default environment under multiple observation modes.
- The package is finite-state and dependency-light. It does not own model training, generated figures, or mutable RL environments.

## Minimal Sanity Check

Run this after changing public APIs:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m py_compile src/geomem_env/*.py tests/*.py
```

For the sibling experiment migration:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../geomem-env/src:experiments python3 -m py_compile experiments/*.py
```
