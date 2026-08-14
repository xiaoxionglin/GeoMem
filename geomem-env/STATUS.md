# geomem-env Status

Current status: the structured plan in `Goal.md` is implemented for the current finite-state GeoMem environment layer.

## Phase Completion

### Phase 1: Stabilize the Core Interface

Complete.

- `Environment` provides states, actions, transitions, validity, rollout, start/goal metadata, and stateless Gymnasium-shaped helpers: `reset(seed=..., options=...)` and `step_result(state, action)`.
- Shared helpers now cover validation, valid edges, graph extraction, reversible navigation graphs, environment summaries, and default goal selection.
- `../geomem-experiments/experiments/commutativity_probe.py` is now a compatibility wrapper over `geomem_env`, so legacy imports still work while definitions live in the package.

### Phase 2: Build the Baseline Taxonomy

Complete.

- The package includes locally metric lattices, trees, shortcut trees, bottleneck rooms, room graphs, trace monoids, aliased repeated-room tasks, cue-gap tasks, route-choice tasks, and looped aliased rooms.
- `EnvironmentMetadata` records family, geometry, observation assumptions, memory pressure, tags, and comparison axes.
- `registered_environment_specs()` and `default_environments()` define the standard diagnostic suite.

### Phase 3: Unify Diagnostics

Complete.

- Reusable diagnostics include effective commutativity, pairwise action reorderability, observation ambiguity, transition ambiguity, goal-conditioned policy ambiguity, and residual history demand.
- The diagnostics accept arbitrary observation functions, so the same measures apply to trace counts, local room position, sparse symbols, cue-gap summaries, and local affordance vectors.

### Phase 4: Bridge Simplified and Realistic Environments

Complete for the current controlled finite-state layer.

- Behavioral navigation families and richer observation summaries are now package-owned.
- `numeric_state_observation()` and `local_view_observation()` provide coordinate-like and higher-dimensional local-feature views.
- `docs/diagnostic_transfer_report.md` records which diagnostics transfer cleanly from simplified to richer environments and where interpretation still needs care.

### Phase 5: Migrate Experiments onto `geomem-env`

Complete for reusable environment constructors and diagnostics.

- Duplicated environment constructors were removed from `../geomem-experiments/experiments/rnn_navigation_benchmark.py` and `../geomem-experiments/experiments/systematic_navigation_sweep.py`; those scripts now consume package-backed constructors through the compatibility wrapper.
- `../geomem-experiments/experiments/systematic_formal_sweep.py` now constructs formal trace environments through `trace_monoid_general()`.
- Existing experiment scripts continue to import `commutativity_probe` where useful, but that module now delegates to `geomem_env` rather than owning environment definitions.

### Phase 6: Validate Against the GeoMem Claim

Complete for environment-side validation.

- `standard_diagnostic_table()` produces a standard diagnostic table across all default environment families and observation summaries.
- `architecture_expectations()` maps residual history demand to expected sufficiency or failure modes for vector/count, finite-history, sequence, reservoir, and hybrid representations.
- Model training and paper figure generation remain in `geomem-experiments`, consistent with this package's non-goals.

## Verification

These checks pass:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m py_compile src/geomem_env/*.py tests/*.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../geomem-env/src:experiments python3 -m py_compile experiments/*.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../geomem-env/src:experiments python3 - <<'PY'
import commutativity_probe as cp
import rnn_navigation_benchmark as rnn
import systematic_navigation_sweep as sns
import systematic_formal_sweep as sfs
print(cp.default_environments()[10].name)
print(rnn.task_specs()[3].env.name)
print(sns.task_specs()[0].env.metadata.family)
print(sfs.trace_environment(sfs.relation_specs(4, 6)[0]).metadata.family)
PY
```

## Future Extensions

These are not blockers for the current `Goal.md` phases, but they are the next research extensions:

- continuous rendered visual observations rather than finite symbolic/local-feature observations;
- full mutable Gymnasium environment classes if external RL agents require them;
- experiment-side model comparisons and figure generation using the new package-owned environment definitions.
