"""Small, dependency-light scaled unscented Kalman filter implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


VectorFunction = Callable[[np.ndarray], np.ndarray]
ScalarFunction = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class ScalarUpdate:
    accepted: bool
    innovation: float
    innovation_variance: float
    nis: float


class UnscentedKalmanFilter:
    """Scaled UKF with sequential scalar measurements and NIS rejection.

    The implementation follows the standard unscented-transform equations.
    Recomputing sigma points before each scalar update permits asynchronous
    sensors and independent rejection of a bad pressure or acceleration sample.
    """

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        *,
        alpha: float = 0.3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.x = np.asarray(mean, dtype=float).copy()
        self.P = np.asarray(covariance, dtype=float).copy()
        self.n = self.x.size
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.lam = alpha**2 * (self.n + kappa) - self.n
        self.scale = self.n + self.lam
        if self.scale <= 0:
            raise ValueError("alpha and kappa produce a non-positive sigma scale")

        self.wm = np.full(2 * self.n + 1, 0.5 / self.scale)
        self.wc = self.wm.copy()
        self.wm[0] = self.lam / self.scale
        self.wc[0] = self.wm[0] + (1.0 - alpha**2 + beta)
        self._stabilize_covariance()

    def sigma_points(self) -> np.ndarray:
        self._stabilize_covariance()
        jitter = 1e-10
        for _ in range(8):
            try:
                root = np.linalg.cholesky(self.scale * (self.P + np.eye(self.n) * jitter))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            values, vectors = np.linalg.eigh(self.P)
            values = np.maximum(values, 1e-9)
            root = vectors @ np.diag(np.sqrt(self.scale * values))

        points = np.empty((2 * self.n + 1, self.n), dtype=float)
        points[0] = self.x
        for i in range(self.n):
            points[i + 1] = self.x + root[:, i]
            points[self.n + i + 1] = self.x - root[:, i]
        return points

    def predict(self, transition: VectorFunction, process_covariance: np.ndarray) -> None:
        propagated = np.asarray([transition(point) for point in self.sigma_points()])
        mean = np.sum(self.wm[:, None] * propagated, axis=0)
        delta = propagated - mean
        covariance = np.einsum("i,ij,ik->jk", self.wc, delta, delta)
        self.x = mean
        self.P = covariance + np.asarray(process_covariance, dtype=float)
        self._stabilize_covariance()

    def update_scalar(
        self,
        measurement: float,
        observation: ScalarFunction,
        variance: float,
        *,
        gate_nis: float | None = None,
    ) -> ScalarUpdate:
        points = self.sigma_points()
        transformed = np.asarray([observation(point) for point in points], dtype=float)
        predicted = float(np.dot(self.wm, transformed))
        residuals = transformed - predicted
        innovation_variance = float(np.dot(self.wc, residuals**2) + variance)
        innovation = float(measurement - predicted)
        nis = innovation**2 / max(innovation_variance, 1e-12)

        if gate_nis is not None and nis > gate_nis:
            return ScalarUpdate(False, innovation, innovation_variance, nis)

        state_delta = points - self.x
        cross_covariance = np.sum(
            self.wc[:, None] * state_delta * residuals[:, None], axis=0
        )
        gain = cross_covariance / innovation_variance
        self.x = self.x + gain * innovation
        self.P = self.P - np.outer(gain, gain) * innovation_variance
        self._stabilize_covariance()
        return ScalarUpdate(True, innovation, innovation_variance, nis)

    def _stabilize_covariance(self) -> None:
        self.P = 0.5 * (self.P + self.P.T)
        values, vectors = np.linalg.eigh(self.P)
        values = np.maximum(values, 1e-10)
        self.P = vectors @ np.diag(values) @ vectors.T

