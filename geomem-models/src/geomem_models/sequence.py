from __future__ import annotations

import math

import numpy as np


def lin_sequence_matrix(sequence_r: int, sequence_l: int, hidden_dim: int) -> tuple[np.ndarray, int, int]:
    sequence_ell = sequence_r + sequence_l - 1
    band_width = max(1, hidden_dim // sequence_ell)
    active_dim = min(hidden_dim, band_width * sequence_ell)
    matrix = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
    for band in range(sequence_ell - 1):
        source = slice(band * band_width, (band + 1) * band_width)
        target = slice((band + 1) * band_width, (band + 2) * band_width)
        matrix[target, source] = np.eye(band_width, dtype=np.float32)
    return matrix, band_width, active_dim


def lin_sequence_states(
    xs: list[np.ndarray],
    sequence_r: int,
    sequence_l: int,
    input_dim: int,
    seed: int,
    hidden_dim: int = 32,
    input_injection: str = "front_bands",
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w_rec, band_width, active_dim = lin_sequence_matrix(sequence_r, sequence_l, hidden_dim)
    w_in = np.zeros((hidden_dim, input_dim), dtype=np.float32)
    if input_injection == "front_bands":
        rows = min(active_dim, sequence_r * band_width)
        w_in[:rows] = rng.normal(0.0, 0.7 / math.sqrt(input_dim), size=(rows, input_dim))
    elif input_injection == "all_units":
        w_in[:active_dim] = rng.normal(0.0, 0.7 / math.sqrt(input_dim), size=(active_dim, input_dim))
    else:
        raise ValueError(f"unknown Lin input injection: {input_injection}")
    states = []
    for x in xs:
        h = np.zeros(hidden_dim, dtype=np.float32)
        for frame in x:
            h = np.tanh(w_in @ frame + w_rec @ h)
        states.append(h)
    return np.asarray(states, dtype=np.float32), w_rec


def lin_block_sequence_matrix(input_dim: int, sequence_r: int, sequence_l: int) -> tuple[np.ndarray, np.ndarray]:
    sequence_ell = sequence_r + sequence_l - 1
    hidden_dim = input_dim * sequence_ell
    w_rec = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
    w_in = np.zeros((hidden_dim, input_dim), dtype=np.float32)
    scale = 1.0 / math.sqrt(max(1, sequence_r))
    for feature_idx in range(input_dim):
        base = feature_idx * sequence_ell
        for step in range(sequence_ell - 1):
            w_rec[base + step + 1, base + step] = 1.0
        for step in range(sequence_r):
            w_in[base + step, feature_idx] = scale
    return w_rec, w_in


def lin_block_sequence_states(xs: list[np.ndarray], sequence_r: int, sequence_l: int) -> tuple[np.ndarray, np.ndarray]:
    w_rec, w_in = lin_block_sequence_matrix(xs[0].shape[1], sequence_r, sequence_l)
    states = []
    for x in xs:
        h = np.zeros(w_rec.shape[0], dtype=np.float32)
        for frame in x:
            h = np.tanh(w_in @ frame + w_rec @ h)
        states.append(h)
    return np.asarray(states, dtype=np.float32), w_rec
