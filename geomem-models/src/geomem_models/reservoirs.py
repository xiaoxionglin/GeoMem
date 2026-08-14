from __future__ import annotations

import math

import numpy as np


def normalize_spectral_radius(matrix: np.ndarray, radius: float = 0.9) -> np.ndarray:
    eigvals = np.linalg.eigvals(matrix)
    current = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    if current < 1e-8:
        return matrix.astype(np.float32)
    return (matrix * (radius / current)).astype(np.float32)


def make_reservoir_matrix(kind: str, hidden_dim: int, seed: int = 0, radius: float = 0.9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "random":
        matrix = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), size=(hidden_dim, hidden_dim))
    elif kind == "symmetric":
        base = rng.normal(size=(hidden_dim, hidden_dim))
        matrix = (base + base.T) / 2.0
    elif kind == "oscillatory":
        base = rng.normal(size=(hidden_dim, hidden_dim))
        matrix = base - base.T
    elif kind == "orthogonal":
        matrix, _ = np.linalg.qr(rng.normal(size=(hidden_dim, hidden_dim)))
    elif kind == "nilpotent":
        matrix = np.zeros((hidden_dim, hidden_dim))
        for idx in range(hidden_dim - 1):
            matrix[idx + 1, idx] = 1.0
    elif kind == "diagonal":
        return np.diag(np.linspace(0.15, radius, hidden_dim)).astype(np.float32)
    elif kind == "low_rank":
        u = rng.normal(size=(hidden_dim, 4))
        v = rng.normal(size=(4, hidden_dim))
        matrix = u @ v / math.sqrt(hidden_dim)
    elif kind == "sparse":
        mask = rng.random((hidden_dim, hidden_dim)) < 0.12
        matrix = rng.normal(0.0, 1.0, size=(hidden_dim, hidden_dim)) * mask
    elif kind == "block_hybrid":
        matrix = np.zeros((hidden_dim, hidden_dim))
        third = hidden_dim // 3
        matrix[:third, :third] = make_reservoir_matrix("symmetric", third, seed + 1, radius=0.7)
        matrix[third:2 * third, third:2 * third] = make_reservoir_matrix("oscillatory", third, seed + 2, radius=0.9)
        matrix[2 * third:, 2 * third:] = make_reservoir_matrix("nilpotent", hidden_dim - 2 * third, seed + 3, radius=0.9)
    else:
        raise ValueError(f"unknown reservoir kind: {kind}")
    return normalize_spectral_radius(np.asarray(matrix, dtype=np.float32), radius=radius)


def reservoir_states(xs: list[np.ndarray], kind: str, hidden_dim: int, input_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w_rec = make_reservoir_matrix(kind, hidden_dim, seed=seed)
    w_in = rng.normal(0.0, 0.7 / math.sqrt(input_dim), size=(hidden_dim, input_dim)).astype(np.float32)
    states = []
    for x in xs:
        h = np.zeros(hidden_dim, dtype=np.float32)
        for frame in x:
            h = np.tanh(w_in @ frame + w_rec @ h)
        states.append(h)
    return np.asarray(states, dtype=np.float32), w_rec


def matrix_metrics(matrix: np.ndarray) -> dict[str, float]:
    eigvals = np.linalg.eigvals(matrix)
    singular = np.linalg.svd(matrix, compute_uv=False)
    singular_sum = float(np.sum(singular))
    probs = singular / singular_sum if singular_sum > 1e-8 else np.ones_like(singular) / len(singular)
    norm = float(np.linalg.norm(matrix)) + 1e-8
    current = np.eye(matrix.shape[0], dtype=np.float32)
    powers = []
    for _ in range(1, 16):
        current = current @ matrix
        powers.append(float(np.linalg.norm(current, ord=2)))
    return {
        "spectral_radius": float(np.max(np.abs(eigvals))) if eigvals.size else 0.0,
        "symmetry_index": float(np.linalg.norm(matrix - matrix.T) / norm),
        "normality_index": float(np.linalg.norm(matrix @ matrix.T - matrix.T @ matrix) / (norm * norm)),
        "effective_rank": float(np.exp(-np.sum(probs * np.log(probs + 1e-12)))),
        "transient_amplification": max(powers) if powers else 0.0,
        "power_decay_10": powers[9] if len(powers) > 9 else float("nan"),
        "oscillatory_score": float(np.mean(np.abs(np.imag(eigvals)) > 0.05)) if eigvals.size else 0.0,
        "nilpotent_score": float(1.0 / (1.0 + (powers[9] if len(powers) > 9 else 0.0))),
    }
