from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BasisSpec:
    name: str
    family: str
    dim: int
    span: int | None = None
    radius: float | None = None
    modes: int | None = None
    threshold: float | None = None


def lag_codes(spec: BasisSpec, max_lag: int) -> np.ndarray:
    if spec.family == "nilpotent":
        if spec.span is None:
            raise ValueError("nilpotent basis requires span")
        codes = np.zeros((max_lag + 1, spec.span), dtype=np.float64)
        for lag in range(min(max_lag + 1, spec.span)):
            codes[lag, lag] = 1.0
        return codes
    if spec.family == "matched_phase_oscillatory":
        lags = np.arange(max_lag + 1, dtype=np.float64)
        omega = 2 * math.pi / 16
        return np.stack([np.cos(omega * lags), np.sin(omega * lags)], axis=1)
    if spec.family in {"oscillatory", "thresholded_oscillatory"}:
        if spec.radius is None or spec.modes is None:
            raise ValueError("oscillatory basis requires radius and modes")
        freqs = np.linspace(math.pi / (2 * spec.modes), math.pi - math.pi / (2 * spec.modes), spec.modes)
        lags = np.arange(max_lag + 1, dtype=np.float64)[:, None]
        envelope = spec.radius ** lags
        codes = np.concatenate([envelope * np.cos(lags * freqs[None, :]), envelope * np.sin(lags * freqs[None, :])], axis=1)
        codes = codes / math.sqrt(spec.modes)
        if spec.family == "thresholded_oscillatory":
            if spec.threshold is None:
                raise ValueError("thresholded oscillatory basis requires threshold")
            codes = np.where(np.abs(codes) >= spec.threshold, codes, 0.0)
        return codes
    raise ValueError(f"unknown basis family: {spec.family}")


def delayed_cue_dataset(codes: np.ndarray, delay: int, n: int, rng: np.random.Generator, distractor_p: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((n, codes.shape[1]), dtype=np.float64)
    y = rng.choice(np.array([-1.0, 1.0]), size=n)
    x += y[:, None] * codes[delay][None, :]
    if delay > 0:
        mask = rng.random((n, delay)) < distractor_p
        values = rng.choice(np.array([-1.0, 1.0]), size=(n, delay))
        for lag in range(delay):
            x += (mask[:, lag] * values[:, lag])[:, None] * codes[lag][None, :]
    return x, y


def phase_dataset(codes: np.ndarray, max_delay: int, period: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    delays = rng.integers(0, max_delay + 1, size=n)
    x = codes[delays]
    y = np.zeros((n, 2), dtype=np.float64)
    phase = 2 * math.pi * (delays % period) / period
    y[:, 0] = np.cos(phase)
    y[:, 1] = np.sin(phase)
    return x, y
