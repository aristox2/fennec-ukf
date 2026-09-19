from __future__ import annotations

import unittest

import numpy as np

from fennec_ukf.filter import UnscentedKalmanFilter


class UnscentedKalmanFilterTests(unittest.TestCase):
    def test_sigma_points_reproduce_mean_and_covariance(self) -> None:
        mean = np.array([2.0, -1.0, 0.5])
        covariance = np.array([[4.0, 0.3, 0.0], [0.3, 2.0, 0.2], [0.0, 0.2, 1.0]])
        ukf = UnscentedKalmanFilter(mean, covariance)
        points = ukf.sigma_points()
        recovered_mean = np.sum(ukf.wm[:, None] * points, axis=0)
        delta = points - recovered_mean
        recovered_covariance = np.einsum("i,ij,ik->jk", ukf.wc, delta, delta)
        np.testing.assert_allclose(recovered_mean, mean, atol=1e-10)
        np.testing.assert_allclose(recovered_covariance, covariance, atol=1e-8)

    def test_scalar_gate_rejects_extreme_outlier(self) -> None:
        ukf = UnscentedKalmanFilter(np.zeros(3), np.eye(3))
        before_mean = ukf.x.copy()
        before_covariance = ukf.P.copy()
        result = ukf.update_scalar(1000.0, lambda x: x[0], 1.0, gate_nis=10.83)
        self.assertFalse(result.accepted)
        np.testing.assert_allclose(ukf.x, before_mean)
        np.testing.assert_allclose(ukf.P, before_covariance)

    def test_nonlinear_pressure_update_moves_height_upward(self) -> None:
        ukf = UnscentedKalmanFilter(np.zeros(3), np.diag([100.0, 10.0, 10.0]))
        pressure = lambda x: 88000.0 * np.exp(-x[0] / 8000.0)
        result = ukf.update_scalar(82000.0, pressure, 100.0**2)
        self.assertTrue(result.accepted)
        self.assertGreater(ukf.x[0], 0.0)


if __name__ == "__main__":
    unittest.main()

