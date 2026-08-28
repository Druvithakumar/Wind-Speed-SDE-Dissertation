# Wind-Speed-SDE-Dissertation

MSc Dissertation — Data-Driven Modelling of Wind Speed Uncertainty (Reading strand)

This repository contains the complete Python pipeline used to produce every SDE-modelling
result for the University of Reading dataset in this dissertation: parameter estimation,
the three SDE models (I, II, III), numerical method validation against Higham (2001), the
eight-task model validation suite, and the hybrid SDE–LSTM extension.

**Note:** the Met Office storm datasets (Storm Bram, Storm Goretti, Kew Gardens) used
elsewhere in the dissertation are not included here, as that data was supplied privately
for this project and is not redistributed.

## What's included

| Section | Content |
|---|---|
| 1–2 | Hourly data aggregation |
| 3 | Weibull MLE fit, ACF/α fit, SDE Models I/II/III (1000-simulation Monte Carlo) |
| 4 | Stationarity testing (ADF) |
| 5 | Seasonal Weibull analysis |
| 6 | Higham (2001) numerical method validation (Brownian motion, stochastic integrals, Euler–Maruyama, strong/weak convergence, Milstein) |
| 7 | Robustness analysis (±10% parameter perturbation) |
| 8–9 | ARIMA / GARCH / LSTM benchmarks and comparison table |
| Tasks 1–8 | Full model validation suite: goodness-of-fit, ACF coverage, Monte Carlo validation, extreme events, prediction accuracy, information criteria, computational efficiency, skill score |
| Chapter 7 extras | Hybrid SDE+LSTM model, Diebold–Mariano significance test, Model III rejection sampling investigation, exceedance probabilities (optional — see below) |

## Setup

```bash
pip install numpy scipy matplotlib pandas statsmodels arch scikit-learn tensorflow --break-system-packages
```

Place the raw University of Reading wind speed CSV at:
```
Dataset/Reading_wind.csv
```
(raw format: 2 header rows, columns `Date, Time, U10, U10max`, 5-minute resolution)

## Running

```bash
python3 reading_pipeline_full.py
```

By default this runs the core pipeline plus all eight Chapter 6 validation tasks
(`run_validation=True`). To also run the Chapter 7 extras (hybrid model, rejection
sampling investigation, exceedance probabilities), edit the last line of the script:

```python
main(run_validation=True, run_chapter7=True)
```

**Expected runtime:** the 1000-simulation Monte Carlo step (Section 3) is the slowest
part of the core pipeline, typically 20–40 minutes depending on hardware. Adding
`run_chapter7=True` trains several LSTM models on top of this and can push total
runtime well over an hour.

A separate zero-shot benchmark against Amazon's Chronos foundation model is available
via `chronos_benchmark()`, but requires an additional install
(`pip install chronos-forecasting torch`) and downloads a pretrained model on first run,
so it is not included in the default run.

## Output

- `plots/` — all figures (`.png`)
- `Dataset/` — all result tables (`.csv`) and cached intermediate outputs (`.npz`)

## Reproducibility note

Deterministic components (data loading, Weibull/ACF fitting, the three SDE simulations,
Higham validation, ARIMA) reproduce exactly given the same input data. GARCH's simulation
step and LSTM training involve randomness that is not perfectly seed-locked across runs,
so those specific results will be close to, but not bit-identical to, the values reported
in the dissertation.
