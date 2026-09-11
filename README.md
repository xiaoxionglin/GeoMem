# GeoMem Workspace

This root folder is now a paper-oriented workspace for the split GeoMem project.

## Start Here

- `geomem-paper/manuscript/paper_draft.md`: current paper draft.
- `geomem-paper/reports/current_results_synthesis.md`: current evidence and claim calibration.
- `literature/reading-list.md`: canonical annotated reading list.
- `split_migration.md`: migration map and verification commands.

## Environment Index

For new sessions focused on environments, start in `geomem-env/`.

- `geomem-env/ENVIRONMENTS.md`: main environment presentation with captions, figures, action rules, observation formulas, taxonomy, and complexity regimes.
- `geomem-env/docs/api_reference.md`: API reference for constructors, metadata, graph helpers, observation channels, and diagnostics.
- `geomem-env/README.md`: package quick start and smoke test for the default Gaussian sensory observation interface.
- `geomem-env/Goal.md`: environment-layer responsibilities, non-goals, and implementation plan.
- `geomem-env/STATUS.md`: current completion status and verification commands.

Current environment convention:

- Default realistic observations use `gaussian_sensory_channel()`: an aliased current-state sensory prototype plus Gaussian noise.
- The environment does not expose action history, valid-action masks, or ground-truth action embeddings in the default sensory interface.
- Model-side experiments carry the previous action token into the next model input; that lives in `geomem-experiments/experiments/rnn_navigation_benchmark.py`.
- Exact symbolic observations and egocentric valid-action observations remain available as controls, not as the preferred realistic setting.

## Active Subprojects

- `geomem-paper/`: manuscript, reports, evaluations, and Obsidian-facing writing context.
- `geomem-experiments/`: runnable experiment scripts and generated figures/CSVs.
- `geomem-theory/`: theory notes and formal modules.
- `geomem-env/`: reusable environment and task-geometry Python package.
- `geomem-models/`: reusable model, reservoir, RNN, Lin-block, and temporal-basis Python package.

Workspace-wide literature lives in `literature/` rather than under any one subproject.

## Archived Pre-Split Project

The original top-level folders were moved to `legacy-pre-split/`. Treat that directory as read-only recovery material; new work should happen in the active `geomem-*` subprojects.
