#!/usr/bin/env python3
"""Run the Fennec UKF and create reproducible CSV, JSON, and PNG outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fennec_ukf.data import (
    align_recorders,
    fit_pressure_model,
    load_recorder,
    write_public_copy,
)
from fennec_ukf.flight import FilterConfiguration, FennecUKF, resample_posterior


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetrum", default="data/telemetrum.csv")
    parser.add_argument("--easymini", default="data/easymini.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--export-public-data", action="store_true")
    return parser.parse_args()


def create_plot(estimate, innovations, output: Path, gate: float) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    colors = {"tm": "#2d6cdf", "em": "#25a18e", "ukf": "#ef8354"}

    axes[0].plot(estimate.time_s, estimate.tm_height_m, color=colors["tm"], lw=1, alpha=0.65, label="TeleMetrum")
    axes[0].plot(estimate.time_s, estimate.em_height_m, color=colors["em"], lw=1, alpha=0.65, label="EasyMini")
    axes[0].plot(estimate.time_s, estimate.height_ukf_m, color=colors["ukf"], lw=2, label="UKF posterior")
    axes[0].fill_between(
        estimate.time_s,
        estimate.height_ukf_m - 2 * estimate.height_sigma_m,
        estimate.height_ukf_m + 2 * estimate.height_sigma_m,
        color=colors["ukf"], alpha=0.18, label="±2 model σ",
    )
    axes[0].set_ylabel("Height AGL (m)")
    axes[0].set_title("Fennec UKF: dual-pressure flight-state estimate")
    axes[0].legend(ncol=4, loc="upper right")

    axes[1].plot(estimate.time_s, estimate.tm_speed_mps, color=colors["tm"], lw=1, alpha=0.55, label="TeleMetrum reported")
    axes[1].plot(estimate.time_s, estimate.em_speed_mps, color=colors["em"], lw=1, alpha=0.55, label="EasyMini reported")
    axes[1].plot(estimate.time_s, estimate.velocity_ukf_mps, color=colors["ukf"], lw=2, label="UKF posterior")
    axes[1].fill_between(
        estimate.time_s,
        estimate.velocity_ukf_mps - 2 * estimate.velocity_sigma_mps,
        estimate.velocity_ukf_mps + 2 * estimate.velocity_sigma_mps,
        color=colors["ukf"], alpha=0.18,
    )
    axes[1].axvspan(5, 13, color="#f6bd60", alpha=0.16, label="adaptive transonic pressure noise")
    axes[1].set_ylabel("Vertical velocity (m/s)")
    axes[1].legend(ncol=4, loc="upper right")

    pressure = innovations[innovations.measurement == "pressure"]
    for sensor, color in (("telemetrum", colors["tm"]), ("easymini", colors["em"])):
        subset = pressure[pressure.sensor == sensor]
        axes[2].scatter(subset.time_s, np.minimum(subset.nis, 1e4), s=4, alpha=0.35, color=color, label=sensor)
    axes[2].axhline(gate, color="#ef8354", lw=1.5, label=f"NIS gate {gate:.2f}")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-3, 1e4)
    axes[2].set_ylabel("Pressure NIS")
    axes[2].set_xlabel("Recorder-aligned time (s)")
    axes[2].legend(ncol=3, loc="upper right")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = arguments()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    telemetrum = load_recorder(args.telemetrum, "telemetrum")
    easymini_raw = load_recorder(args.easymini, "easymini")
    easymini, alignment_offset, height_bias = align_recorders(telemetrum, easymini_raw)

    if args.export_public_data:
        write_public_copy(telemetrum, output / "telemetrum_public.csv")
        write_public_copy(easymini_raw, output / "easymini_public.csv")

    models = {
        "telemetrum": fit_pressure_model(telemetrum),
        "easymini": fit_pressure_model(easymini),
    }
    configuration = FilterConfiguration()
    posterior, innovations, counters = FennecUKF(models, configuration).run(
        [telemetrum, easymini]
    )
    estimate = resample_posterior(
        posterior, telemetrum, easymini, configuration.output_step_s
    )

    apogee_index = int(estimate.height_ukf_m.idxmax())
    peak_index = int(estimate.velocity_ukf_mps.idxmax())
    summary = {
        "filter": "scaled unscented Kalman filter",
        "state": ["height_m", "vertical_velocity_mps", "vertical_acceleration_mps2"],
        "measurement_channels": [
            "TeleMetrum raw pressure",
            "EasyMini raw pressure",
            "TeleMetrum ascent acceleration",
        ],
        "easymini_time_offset_s": alignment_offset,
        "easymini_height_bias_removed_m": height_bias,
        "alignment_method": "robust joint time/height offset fit over T+25–45 s, expressed on the TeleMetrum clock and height datum",
        "posterior_apogee_m": float(estimate.loc[apogee_index, "height_ukf_m"]),
        "posterior_apogee_time_s": float(estimate.loc[apogee_index, "time_s"]),
        "posterior_peak_velocity_mps": float(estimate.loc[peak_index, "velocity_ukf_mps"]),
        "posterior_peak_velocity_time_s": float(estimate.loc[peak_index, "time_s"]),
        "configuration": configuration.as_dict(),
        "update_counts": counters,
        "limitations": [
            "The two recorder clocks do not contain a verified shared timecode.",
            "The separate custom-payload IMU stream is not present in the supplied flight files.",
            "Reported posterior covariance depends on configurable process and measurement noise; it is not an independently calibrated confidence interval.",
            "No independent tracking truth was available to calculate trajectory error.",
        ],
    }

    estimate.to_csv(output / "fennec_ukf_estimate.csv", index=False, float_format="%.6f")
    innovations.to_csv(output / "fennec_ukf_innovations.csv", index=False, float_format="%.6f")
    (output / "fennec_ukf_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    create_plot(estimate, innovations, output / "fennec_ukf_overview.png", configuration.gate_nis)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
