# Fennec UKF

## what the filter is looking at 

- TeleMetrum raw static pressure
- EasyMini raw static pressure
- TeleMetrum vertical acceleration through T+48 s
- Each recorder's pressure-derived height and reported speed for calibration, comparison, and plotting only

The supplied files do **not** contain the separate custom-payload IMU stream. BUUUUT i will make sure i make sure it's more accesible.

## why a UKF?

the state is `x = [height, vertical velocity, vertical acceleration]`. a constant-acceleration process model propagates the sigma points. Each altimeter uses a nonlinear pressure observation:

```text
log pressure = c0 + c1·height + c2·height²
```

the coefficients are robustly fitted to each recorder's pressure-height relationship. because ressure updates are asynchronous, a normalized-innovation-squared gate rejects samples that are inconsistent with the predicted state and covariance. pressure noise is raised through the transonic and deployment windows rather than treating every barometric sample as equally reliable.

## run it!

```powershell
cd path\to\fennec-ukf
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py run_fennec_ukf.py
```

outputs appear in `outputs/`:

- `fennec_ukf_estimate.csv` — posterior state and model covariance at 20 Hz, with both recorder traces
- `fennec_ukf_innovations.csv` — innovation, NIS, and accept/reject decision for each update
- `fennec_ukf_summary.json` — configuration, update counts, derived results, and limitations
- `fennec_ukf_overview.png` — height, velocity, uncertainty, and pressure-gating plot

Run tests:

```powershell
py -m unittest discover -s tests -v
```

## replace the data

The loader accepts the original Altus Metrum CSV schema. Required columns are `time`, `state_name`, `acceleration`, `pressure`, `height`, `speed`, and `battery_voltage`. Extra columns are ignored. The included copies remove GPS coordinates, callsigns, serial numbers, and calendar timestamps.

```powershell
py run_fennec_ukf.py --telemetrum path\to\telemetrum.csv --easymini path\to\easymini.csv
```

## assumptions that stay attached to results (for now)

1. the recorder clocks do not contain a verified shared timecode. A robust joint time/height offset is fitted over the clean T+25–45 s coast window and used to express EasyMini on the TeleMetrum clock and height datum. Both corrections are reported in the summary JSON.
2. the covariance is conditional on the parameters in `FilterConfiguration`; it is not independently calibrated against tracking truth.
3. no independent radar, optical trajectory, or payload-IMU truth is available in the supplied files. Absolute trajectory error cannot be verified.
4. both barometers can share aerodynamic pressure errors. Two agreeing pressure sensors are not automatically independent truth.
5. the filter is an offline analysis tool, not flight-critical software.

## design map <3

- `fennec_ukf/filter.py` — generic scaled UKF and scalar NIS gating
- `fennec_ukf/data.py` — recorder parser, privacy-safe export, alignment, pressure calibration
- `fennec_ukf/flight.py` — Fennec process model, adaptive noise, and asynchronous fusion
- `run_fennec_ukf.py` — command-line entry point, outputs, plots, and summary
- `tests/test_filter.py` — sigma-point, outlier-gate, and nonlinear-update tests

