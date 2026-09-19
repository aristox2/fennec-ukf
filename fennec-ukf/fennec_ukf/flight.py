"""Fennec-specific asynchronous sensor fusion around the generic UKF."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data import PressureModel
from .filter import UnscentedKalmanFilter


@dataclass(frozen=True)
class FilterConfiguration:
    alpha: float = 0.3
    beta: float = 2.0
    kappa: float = 0.0
    pressure_sigma_pa: float = 250.0
    transonic_pressure_sigma_pa: float = 1_200.0
    deployment_pressure_sigma_pa: float = 800.0
    acceleration_sigma_mps2: float = 14.0
    gate_nis: float = 10.83
    output_step_s: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _jerk_covariance(dt: float, spectral_density: float) -> np.ndarray:
    dt2, dt3 = dt * dt, dt * dt * dt
    dt4, dt5 = dt3 * dt, dt3 * dt * dt
    return spectral_density * np.array(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ]
    )


def _jerk_density(time_s: float) -> float:
    if time_s < 15.0:
        return 500.0
    if time_s < 45.0:
        return 35.0
    if time_s < 58.0:
        return 90.0
    if time_s < 330.0:
        return 18.0
    return 55.0


class FennecUKF:
    """Estimate [height, vertical velocity, vertical acceleration]."""

    def __init__(
        self,
        pressure_models: dict[str, PressureModel],
        configuration: FilterConfiguration | None = None,
    ) -> None:
        self.models = pressure_models
        self.config = configuration or FilterConfiguration()

    def run(self, recorders: Iterable[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        events = pd.concat(recorders, ignore_index=True).sort_values(
            ["time_s", "sensor"], kind="stable"
        )
        events = events[np.isfinite(events.time_s) & np.isfinite(events.pressure_pa)]
        start = float(events.time_s.min())
        ukf = UnscentedKalmanFilter(
            np.array([0.0, 0.0, 0.0]),
            np.diag([80.0**2, 100.0**2, 80.0**2]),
            alpha=self.config.alpha,
            beta=self.config.beta,
            kappa=self.config.kappa,
        )

        rows: list[dict] = []
        innovations: list[dict] = []
        current_time = start
        counters = {
            "pressure_accepted": {"telemetrum": 0, "easymini": 0},
            "pressure_rejected": {"telemetrum": 0, "easymini": 0},
            "acceleration_accepted": 0,
            "acceleration_rejected": 0,
        }

        for event in events.itertuples(index=False):
            time_s = float(event.time_s)
            dt = max(0.0, time_s - current_time)
            if dt > 0:
                def transition(state: np.ndarray, interval: float = dt) -> np.ndarray:
                    height, velocity, acceleration = state
                    if time_s >= 48.5:
                        # Once recovery begins, the inertial vertical acceleration
                        # should return toward zero instead of carrying ballistic
                        # coast acceleration through the parachute descent.
                        next_acceleration = acceleration * np.exp(-interval / 1.5)
                    else:
                        next_acceleration = acceleration
                    return np.array(
                        [
                            height + velocity * interval + 0.25 * (acceleration + next_acceleration) * interval**2,
                            velocity + 0.5 * (acceleration + next_acceleration) * interval,
                            next_acceleration,
                        ]
                    )

                ukf.predict(transition, _jerk_covariance(dt, _jerk_density(time_s)))
                current_time = time_s

            sensor = str(event.sensor)
            pressure_sigma = self.config.pressure_sigma_pa
            if 5.0 <= time_s <= 13.0:
                pressure_sigma = self.config.transonic_pressure_sigma_pa
            elif 45.0 <= time_s <= 58.0:
                pressure_sigma = self.config.deployment_pressure_sigma_pa

            pressure_result = ukf.update_scalar(
                float(event.pressure_pa),
                lambda state, model=self.models[sensor]: model.pressure(state[0]),
                pressure_sigma**2,
                gate_nis=self.config.gate_nis,
            )
            pressure_key = "pressure_accepted" if pressure_result.accepted else "pressure_rejected"
            counters[pressure_key][sensor] += 1
            innovations.append(
                {
                    "time_s": time_s,
                    "sensor": sensor,
                    "measurement": "pressure",
                    "innovation": pressure_result.innovation,
                    "innovation_sigma": np.sqrt(pressure_result.innovation_variance),
                    "nis": pressure_result.nis,
                    "accepted": pressure_result.accepted,
                }
            )

            # Only TeleMetrum's ascent acceleration is fused. The two recorder
            # acceleration columns do not share consistent behavior under canopy.
            acceleration = float(event.acceleration_mps2)
            if sensor == "telemetrum" and time_s <= 48.0 and np.isfinite(acceleration):
                acceleration_result = ukf.update_scalar(
                    acceleration,
                    lambda state: float(state[2]),
                    self.config.acceleration_sigma_mps2**2,
                    gate_nis=self.config.gate_nis,
                )
                key = "acceleration_accepted" if acceleration_result.accepted else "acceleration_rejected"
                counters[key] += 1
                innovations.append(
                    {
                        "time_s": time_s,
                        "sensor": sensor,
                        "measurement": "acceleration",
                        "innovation": acceleration_result.innovation,
                        "innovation_sigma": np.sqrt(acceleration_result.innovation_variance),
                        "nis": acceleration_result.nis,
                        "accepted": acceleration_result.accepted,
                    }
                )

            rows.append(
                {
                    "time_s": time_s,
                    "height_ukf_m": ukf.x[0],
                    "velocity_ukf_mps": ukf.x[1],
                    "acceleration_ukf_mps2": ukf.x[2],
                    "height_variance_m2": ukf.P[0, 0],
                    "velocity_variance_m2ps2": ukf.P[1, 1],
                    "acceleration_variance_m2ps4": ukf.P[2, 2],
                }
            )

        posterior = pd.DataFrame(rows).groupby("time_s", as_index=False).last()
        innovation_frame = pd.DataFrame(innovations)
        return posterior, innovation_frame, counters


def resample_posterior(
    posterior: pd.DataFrame,
    primary: pd.DataFrame,
    backup: pd.DataFrame,
    step_s: float,
) -> pd.DataFrame:
    start = max(0.0, float(posterior.time_s.min()))
    end = min(float(primary.time_s.max()), float(backup.time_s.max()))
    timeline = np.arange(start, end + step_s * 0.5, step_s)
    result = pd.DataFrame({"time_s": timeline})

    for column in posterior.columns.drop("time_s"):
        result[column] = np.interp(timeline, posterior.time_s, posterior[column])
    result["height_sigma_m"] = np.sqrt(np.maximum(result.pop("height_variance_m2"), 0.0))
    result["velocity_sigma_mps"] = np.sqrt(
        np.maximum(result.pop("velocity_variance_m2ps2"), 0.0)
    )
    result["acceleration_sigma_mps2"] = np.sqrt(
        np.maximum(result.pop("acceleration_variance_m2ps4"), 0.0)
    )

    for prefix, frame in (("tm", primary), ("em", backup)):
        for source, target in (
            ("height_m", f"{prefix}_height_m"),
            ("speed_mps", f"{prefix}_speed_mps"),
            ("pressure_pa", f"{prefix}_pressure_pa"),
        ):
            result[target] = np.interp(timeline, frame.time_s, frame[source])
    return result
