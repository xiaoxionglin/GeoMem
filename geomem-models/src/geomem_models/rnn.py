from __future__ import annotations

import math

import numpy as np


class NumpyRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        self.w_in = rng.normal(0.0, 0.4 / math.sqrt(input_dim), size=(hidden_dim, input_dim)).astype(np.float32)
        self.w_rec = rng.normal(0.0, 0.4 / math.sqrt(hidden_dim), size=(hidden_dim, hidden_dim)).astype(np.float32)
        self.b_h = np.zeros(hidden_dim, dtype=np.float32)
        self.w_out = rng.normal(0.0, 0.4 / math.sqrt(hidden_dim), size=(output_dim, hidden_dim)).astype(np.float32)
        self.b_out = np.zeros(output_dim, dtype=np.float32)

    def final_states(self, xs: list[np.ndarray]) -> np.ndarray:
        states = []
        for x in xs:
            h = np.zeros(self.w_rec.shape[0], dtype=np.float32)
            for frame in x:
                h = np.tanh(self.w_in @ frame + self.w_rec @ h + self.b_h)
            states.append(h)
        return np.asarray(states, dtype=np.float32)

    def logits(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        h = np.zeros(self.w_rec.shape[0], dtype=np.float32)
        states = [h]
        for frame in x:
            h = np.tanh(self.w_in @ frame + self.w_rec @ h + self.b_h)
            states.append(h)
        return self.w_out @ h + self.b_out, states

    def predict(self, xs: list[np.ndarray]) -> np.ndarray:
        return np.asarray([int(np.argmax(self.logits(x)[0])) for x in xs])

    def fit(self, xs: list[np.ndarray], ys: list[int], epochs: int = 8, lr: float = 0.015, seed: int = 0) -> list[float]:
        rng = np.random.default_rng(seed)
        losses: list[float] = []
        for _ in range(epochs):
            order = rng.permutation(len(xs))
            total_loss = 0.0
            for idx in order:
                x = xs[int(idx)]
                y = int(ys[int(idx)])
                logits, states = self.logits(x)
                logits = logits - np.max(logits)
                probs = np.exp(logits)
                probs = probs / np.sum(probs)
                total_loss += -math.log(float(probs[y]) + 1e-12)
                grad_logits = probs
                grad_logits[y] -= 1.0
                grad_w_out = np.outer(grad_logits, states[-1])
                grad_b_out = grad_logits
                dh = self.w_out.T @ grad_logits
                grad_w_in = np.zeros_like(self.w_in)
                grad_w_rec = np.zeros_like(self.w_rec)
                grad_b_h = np.zeros_like(self.b_h)
                for t in range(len(x), 0, -1):
                    h = states[t]
                    h_prev = states[t - 1]
                    dz = dh * (1.0 - h * h)
                    grad_w_in += np.outer(dz, x[t - 1])
                    grad_w_rec += np.outer(dz, h_prev)
                    grad_b_h += dz
                    dh = self.w_rec.T @ dz
                for grad in (grad_w_in, grad_w_rec, grad_b_h, grad_w_out, grad_b_out):
                    np.clip(grad, -1.0, 1.0, out=grad)
                self.w_in -= lr * grad_w_in
                self.w_rec -= lr * grad_w_rec
                self.b_h -= lr * grad_b_h
                self.w_out -= lr * grad_w_out
                self.b_out -= lr * grad_b_out
            losses.append(total_loss / max(1, len(xs)))
        return losses
