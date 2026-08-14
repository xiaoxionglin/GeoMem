# geomem-models

Reusable model and temporal-memory code for GeoMem.

## Owns

- Fixed reservoir matrix families and matrix diagnostics.
- Lin-style sequence states and independent block-nilpotent chains.
- Lightweight NumPy RNN.
- Linear/ridge readout helpers.
- Temporal-basis utilities for nilpotent, damped oscillatory, thresholded oscillatory, and phase-oscillator analyses.
- Representation builders used by supervised navigation, delayed-cue, phase-reconstruction, and architecture-selection benchmarks.
- Shared model utilities for comparing memoryless controls, explicit finite histories, reservoirs, trained RNNs, Lin buffers, and hybrid representations.

## Dependency Direction

This package depends on NumPy and should avoid importing experiment scripts. It may use types from `geomem-env` only when needed.

## Relationship To Experiments

Experiment scripts should own datasets, sweeps, plots, and result files. Reusable model families, readouts, recurrent matrix analyses, temporal bases, and representation encoders should live here so benchmarks can share them without duplicating code.

## Smoke Test

```bash
PYTHONPATH=src python3 - <<'PY'
import numpy as np
from geomem_models import lin_block_sequence_states, lag_codes, BasisSpec
xs = [np.eye(3, dtype=np.float32)]
states, _ = lin_block_sequence_states(xs, sequence_r=1, sequence_l=4)
print(states.shape)
print(lag_codes(BasisSpec("nil", "nilpotent", dim=4, span=4), 4).shape)
PY
```
