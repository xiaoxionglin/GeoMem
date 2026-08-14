# GeoMem Split Migration

Status: first-pass internal split implemented. Each `geomem-*` folder has been initialized as a separate git repository on branch `main`; no initial commits have been made. The pre-split top-level project has been archived under `legacy-pre-split/`.

## New Subprojects

- `geomem-env`: reusable environment and geometry package.
- `geomem-models`: reusable model, reservoir, RNN, Lin-block, and temporal-basis package.
- `geomem-experiments`: runnable experiment scripts and generated artifacts.
- `geomem-theory`: theory notes, formal modules, and literature-facing conceptual work.
- `geomem-paper`: manuscript, reports, and Obsidian-facing paper context.

## Dependency Direction

```text
geomem-env <- geomem-models <- geomem-experiments
geomem-env <------------------ geomem-experiments

geomem-theory -> geomem-paper
geomem-experiments/figures -> geomem-paper figure embeds
```

`geomem-env` is the lowest-level code dependency. `geomem-experiments` may depend on everything needed to run benchmarks. `geomem-paper` should not own generated artifacts.

## Migration Policy

This is an incremental split. The original top-level folders remain intact until the new subprojects pass smoke tests and the workflow stabilizes.

Use clean snapshot commits for each `geomem-*` folder. Do not try to preserve the current top-level history in this first pass.

The root `README.md` now acts as the paper-workspace entrypoint. `legacy-pre-split/` is archival recovery material only.

## First-Pass Verification

Run:

```bash
PYTHONPATH=geomem-env/src python3 -c "from geomem_env import trace_monoid_family, effective_commutativity; env=trace_monoid_family()[0]; print(env.name, round(effective_commutativity(env, length=3, n_sequences=20, seed=1, valid_only=True), 3))"
PYTHONPATH=geomem-models/src python3 -c "from geomem_models import BasisSpec, lag_codes; print(lag_codes(BasisSpec('nil', 'nilpotent', 4, span=4), 4).shape)"
cd geomem-experiments && python3 -m py_compile experiments/*.py
```
