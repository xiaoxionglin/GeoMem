from __future__ import annotations

import numpy as np


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    eye = np.eye(x_aug.shape[1], dtype=x.dtype)
    eye[-1, -1] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + ridge * eye, x_aug.T @ y)


def ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ w


def fit_linear_predictor(x_train: np.ndarray, y_train: list[int], ridge: float = 1e-2):
    labels = sorted(set(y_train))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y = np.zeros((len(y_train), len(labels)), dtype=np.float64)
    for row, label in enumerate(y_train):
        y[row, label_to_idx[label]] = 1.0
    w = ridge_fit(np.asarray(x_train, dtype=np.float64), y, ridge=ridge)

    def predict(x: np.ndarray) -> np.ndarray:
        scores = ridge_predict(np.asarray(x, dtype=np.float64), w)
        return np.asarray([labels[int(idx)] for idx in np.argmax(scores, axis=1)])

    return predict
