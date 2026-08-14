# geomem-env

Reusable environment and task-geometry code for GeoMem.

## Owns

- `Environment`, `State`, `Action`, and transition helpers.
- Lattice, tree, bottleneck-room, shortcut-tree, and trace-monoid constructors.
- Room-graph constructors for loop-rich, bottleneck, and tree-like navigation topologies.
- Behavioral navigation environments used by supervised policy and RNN benchmarks.
- Observation channels and controls: symbolic state modes, local room views, sparse visual features, egocentric landmarks, and Gaussian sensory prototypes.
- Metadata, registry helpers, and environment summaries for comparing environment families.
- Geometry diagnostics used by experiments: effective commutativity, observation ambiguity, transition ambiguity, policy ambiguity, sampled sensory ambiguity, and residual-history pressure.
- Standard diagnostic tables and environment-gallery support for paper-facing figures.

## Dependency Direction

This package is the lowest-level Python dependency. It must not import models or experiments.

## Relationship To Experiments

Experiment scripts may orchestrate sweeps, write CSV/SVG outputs, and choose benchmark settings. Reusable task definitions, observation logic, and diagnostics should live here so `geomem-experiments/` can stay thin and reproducible.

## Smoke Test

```bash
PYTHONPATH=src python3 - <<'PY'
from geomem_env import (
    environment_summary,
    gaussian_sensory_channel,
    room_graph_environment,
    sampled_residual_history_diagnostics,
)

env = room_graph_environment("bottleneck")
channel = gaussian_sensory_channel(env, sigma=0.1, dim=8, seed=19)

print(environment_summary(env)["geometry"])
print(channel.kind, channel.dim)
print(tuple(round(value, 3) for value in channel.observe(env.states[0], seed=17)))
print(round(sampled_residual_history_diagnostics(env, channel, n_samples=2)["residual_history_score"], 3))
PY
```

## Current API Shape

- See `ENVIRONMENTS.md` for the main environment presentation with figure captions.
- See `docs/api_reference.md` for the practical API guide intended for other Codex threads.
- `Environment` is a deterministic, stateless transition system with Gymnasium-shaped helpers: `reset(seed=..., options=...)` returns `(state, info)` and `step_result(state, action)` returns a `StepResult`.
- `EnvironmentMetadata` records family, geometry, observation assumptions, memory pressure, comparison axes, and tags.
- `registered_environment_specs()` lists standard diagnostic families; `default_environments()` instantiates the current default suite.
- `validate_environment()`, `graph_edges()`, and `environment_summary()` provide package-level checks and summaries.
- `observation_ambiguity()`, `transition_ambiguity()`, and `policy_ambiguity()` accept arbitrary observation functions so simplified and richer environments can be compared on shared axes.
- `gaussian_sensory_channel()` is the default realistic observation path: aliased current-state sensory prototypes plus Gaussian noise, with no action bits or action history.
- `observe_trace_state()` and `observe_room_state()` provide formal/control observation summaries for diagnostic comparisons.
- `standard_sensory_diagnostic_table()` evaluates sampled Gaussian sensory observations; `standard_diagnostic_table()` remains available for exact symbolic controls.
- `docs/diagnostic_transfer_report.md` records which diagnostics transfer from simplified to richer environments and where interpretation still needs care.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```
