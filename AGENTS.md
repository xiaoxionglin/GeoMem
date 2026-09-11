# Repository Guidelines

## Project Structure & Module Organization

This root folder is a split GeoMem workspace. Keep executable and subproject-specific work in the active subprojects:

- `geomem-env/`: reusable environments, observations, and diagnostics. Source is in `src/geomem_env/`, tests in `tests/`, notes in `docs/`. It owns graph/task constructors, sensory channels, and commutativity, ambiguity, policy, and residual-history diagnostics.
- `geomem-models/`: reusable memory models. Source is in `src/geomem_models/`. It owns reservoirs, lightweight RNNs, Lin/block sequence states, temporal bases, readouts, and matrix diagnostics.
- `geomem-experiments/`: benchmark, sweep, plotting, and backfill scripts under `experiments/`; CSV/SVG outputs under `figures/`. Keep orchestration here and migrate reusable logic into sibling packages.
- `geomem-paper/`: manuscripts, reports, evaluations, and paper-facing context.
- `geomem-theory/`: formal modules and conceptual notes.
- `literature/`: canonical workspace-wide reading list, literature comparison matrix, and focused literature notes. Papers relevant to multiple subprojects belong here and must not be mirrored elsewhere.
- `legacy-pre-split/`: archived recovery material. Treat as read-only unless explicitly migrating content.

## Build, Test, and Development Commands

Run commands from the relevant subproject unless shown otherwise.

```bash
cd geomem-env && PYTHONPATH=src python3 -m unittest discover -s tests
cd geomem-env && PYTHONPATH=src python3 -m py_compile src/geomem_env/*.py
cd geomem-models && PYTHONPATH=src python3 -m py_compile src/geomem_models/*.py
cd geomem-experiments && python3 -m py_compile experiments/*.py
cd geomem-experiments && python3 experiments/commutativity_probe.py
```

Use the experiment smoke tests in `geomem-experiments/README.md`; avoid expensive RNN sweeps for routine checks.

## Coding Style & Naming Conventions

Python targets 3.10+. Use 4-space indentation, type hints for public helpers, `from __future__ import annotations` where useful, and dataclasses for immutable records. Prefer deterministic seeds. Use snake_case for functions, modules, and variables; use PascalCase for classes such as `Environment` and `StepResult`.

Keep dependency direction clean: `geomem-env` must not import models or experiments; `geomem-models` should avoid experiment imports; experiments may depend on both sibling packages.

## Testing Guidelines

The active test suite is `geomem-env/tests/` and uses `unittest`. Name test files `test_*.py` and test methods `test_*`. Add focused tests when changing constructors, diagnostics, observation channels, or public package APIs. For experiment scripts, at minimum run `py_compile` plus one small smoke script that exercises the changed path.

## Commit & Pull Request Guidelines

The subproject repositories have no commits yet, so no local convention is established. Use concise, imperative messages such as `Add sensory alias diagnostics` or `Document experiment smoke tests`. Pull requests should name the changed subproject, summarize verification commands, note generated artifacts, and link any affected paper/report claim.

## Agent-Specific Instructions

Do not rewrite archived `legacy-pre-split/` material during normal edits. Keep generated CSV/SVG outputs in `geomem-experiments/figures/` and avoid committing caches such as `__pycache__/`.


## Analysis guideline

after making new analyses, update it to the report. Do not simply append the results chronologically. Whenever possible, integrate the new result into the existing figures and report sections, e.g. the new result that extend the parameter range, or another variant of architecture tested for the same metrics. That also means, when adding a new variant, test it on all the existing metrics, do not omit anything unless there's a strong reason.
The report should be tailored such that it takes minimal effort for humans to understand.
