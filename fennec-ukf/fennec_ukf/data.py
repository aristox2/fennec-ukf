"""Recorder loading, privacy-safe export, time alignment, and pressure models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PUBLIC_COLUMNS = {
    "time": "time_s",
    "state_name": "state",
    "acceleration": "acceleration_mps2",
    "pressure": "pressure_pa",
    "height": "height_m",
    "speed": "speed_mps",
    "battery_voltage": "battery_v",
}


@dataclass(frozen=True)
class PressureModel:
    """Empirical nonlinear conversion between relative height and pressure."""

    coefficients: np.ndarray
    valid_height_min: float = -200.0
    valid_height_max: float = 12_500.0

    def pressure(self, height_m: float) -> float:
        height = float(np.clip(height_m, self.valid_height_min, self.valid_height_max))
        c0, c1, c2 = self.coefficients
        return float(np.exp(c0 + c1 * height + c2 * height * height))


def load_recorder(path: str | Path, name: str) -> pd.DataFrame:
    source = pd.read_csv(path, skipinitialspace=True)
    original_missing = [column for column in PUBLIC_COLUMNS if column not in source.columns]
    public_names = list(PUBLIC_COLUMNS.values())
    public_missing = [column for column in public_names if column not in source.columns]
    if not original_missing:
        selected = source[list(PUBLIC_COLUMNS)].rename(columns=PUBLIC_COLUMNS).copy()
    elif not public_missing:
        selected = source[public_names].copy()
    else:
        raise ValueError(
            f"{name} is missing required columns. Expected the original Altus Metrum "
            f"names or the privacy-safe names: {', '.join(public_names)}"
        )
    selected["state"] = selected["state"].astype(str).str.strip()
    for column in selected.columns.drop("state"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna(subset=["time_s", "pressure_pa", "height_m"])

    numeric = [column for column in selected.columns if column != "state"]
    grouped_numeric = selected.groupby("time_s", as_index=False)[numeric[1:]].median()
    grouped_state = selected.groupby("time_s", as_index=False)["state"].last()
    result = grouped_numeric.merge(grouped_state, on="time_s", how="left")
    result["sensor"] = name
    return result.sort_values("time_s", kind="stable").reset_index(drop=True)


def apogee_time(frame: pd.DataFrame) -> float:
    window = frame[(frame.time_s >= 30.0) & (frame.time_s <= 70.0)]
    if window.empty:
        raise ValueError("recorder has no samples in the expected apogee window")
    return float(window.loc[window.pressure_pa.idxmin(), "time_s"])


def align_recorders(
    primary: pd.DataFrame, backup: pd.DataFrame
) -> tuple[pd.DataFrame, float, float]:
    # Jointly estimate relative time and constant height-datum offsets in a
    # clean late-coast window. A deterministic grid search avoids pretending the
    # two clocks have a verified synchronization event.
    comparison_time = primary.loc[primary.time_s.between(25.0, 45.0), "time_s"].to_numpy()
    primary_height = np.interp(comparison_time, primary.time_s, primary.height_m)
    candidates = np.linspace(-1.5, 1.5, 601)
    best: tuple[float, float, float] | None = None
    for candidate in candidates:
        backup_height = np.interp(
            comparison_time, backup.time_s.to_numpy() + candidate, backup.height_m
        )
        difference = backup_height - primary_height
        bias = float(np.median(difference))
        score = float(np.median(np.abs(difference - bias)))
        if best is None or score < best[0]:
            best = (score, float(candidate), bias)
    assert best is not None
    _, offset, height_bias = best
    aligned = backup.copy()
    aligned["time_s"] = aligned["time_s"] + offset
    aligned["height_m"] = aligned["height_m"] - height_bias
    return aligned, float(offset), height_bias


def fit_pressure_model(frame: pd.DataFrame) -> PressureModel:
    finite = frame[np.isfinite(frame.pressure_pa) & np.isfinite(frame.height_m)].copy()
    finite = finite[(finite.pressure_pa > 0) & finite.height_m.between(-100.0, 12_100.0)]
    x = finite.height_m.to_numpy(dtype=float)
    y = np.log(finite.pressure_pa.to_numpy(dtype=float))
    design = np.column_stack([np.ones_like(x), x, x * x])
    mask = np.ones(x.size, dtype=bool)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(5):
        residual = y - design @ coefficients
        center = np.median(residual[mask])
        mad = np.median(np.abs(residual[mask] - center))
        scale = max(1.4826 * mad, 1e-7)
        mask = np.abs(residual - center) < 4.0 * scale
        coefficients = np.linalg.lstsq(design[mask], y[mask], rcond=None)[0]
    return PressureModel(coefficients=coefficients)


def write_public_copy(frame: pd.DataFrame, path: str | Path) -> None:
    columns = [
        "time_s",
        "state",
        "acceleration_mps2",
        "pressure_pa",
        "height_m",
        "speed_mps",
        "battery_v",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame[columns].to_csv(path, index=False, float_format="%.5f")
