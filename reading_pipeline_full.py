"""
================================================================================
MASTER PIPELINE (Reading dataset only) — Project 53:
Data-Driven Modelling of Wind Speed Uncertainty
================================================================================
This script reproduces every Reading-strand result in the dissertation:
hourly aggregation, Weibull MLE fit, ACF/alpha fit, SDE Models I/II/III
(1000-simulation Monte Carlo), stationarity test, seasonal Weibull analysis,
the six Higham (2001) numerical-method validation scripts, a robustness
check, and the ARIMA/GARCH/LSTM benchmark comparison.

NOTE: This repository intentionally contains ONLY the University of Reading
2023 dataset pipeline. The Met Office storm datasets (Storm Bram, Storm
Goretti, Kew Gardens) used elsewhere in the dissertation were supplied
privately for this project and are NOT redistributed here.

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------
1. Install dependencies:
       pip install -r requirements.txt

2. Place the raw Reading CSV at:
       Dataset/Reading_wind.csv
   (University of Reading Department of Meteorology, 2023, openly available
   via the MODE3 interface: https://www.met.reading.ac.uk/~brugge/mode3.html)
   The raw file has 2 header rows and 4 columns: Date, Time, U10, U10max.

3. Run:
       python reading_pipeline.py

   Outputs are written to ./Dataset/ (CSV/NPZ) and ./plots/ (figures).

--------------------------------------------------------------------------
EXPECTED RUNTIME
--------------------------------------------------------------------------
Section 3 (SDE simulation, 1000 runs x 3 models) is the slow part —
Model II alone evaluates the incomplete Gamma function at every one of
87,600 fine time-steps per simulation. Expect roughly 20-40 minutes for
the full pipeline on a standard laptop CPU; the Higham scripts (Section 6)
and everything else combined take well under a minute.
================================================================================
"""

import os
import time
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import weibull_min, gaussian_kde
from scipy.stats import norm as sp_norm
from scipy.special import gamma as Gamma, gammainc
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

try:
    import torch  # only required if chronos_benchmark() is called
except ImportError:
    torch = None

warnings.filterwarnings("ignore")

# ==============================================================================
# PATHS — all relative to this script's location, so the repo runs anywhere
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

DATA_PATH_RAW = os.path.join(DATA_DIR, "Reading_wind.csv")
DATA_PATH_HOURLY = os.path.join(DATA_DIR, "Reading_wind_hourly.csv")


def plot_path(filename):
    return os.path.join(PLOT_DIR, filename)


def data_path(filename):
    return os.path.join(DATA_DIR, filename)


# ==============================================================================
# SECTION 1 — Hourly aggregation: initial exploration / sanity check
# ==============================================================================
def load_raw_reading_data():
    """Shared loader used by every section that needs the raw 5-minute data."""
    df = pd.read_csv(
        DATA_PATH_RAW,
        skiprows=2,
        names=["Date", "Time", "U10", "U10max"],
        na_values=["", "m/s", "hhmm", "UTC"],
    )
    df = df[df["Date"] != "UTC"]
    df = df[pd.to_numeric(df["U10"], errors="coerce").notna()]
    df["U10"] = pd.to_numeric(df["U10"], errors="coerce")
    df["U10max"] = pd.to_numeric(df["U10max"], errors="coerce")
    df = df.dropna(subset=["U10"])
    df["Date"] = df["Date"].astype(str)
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").astype(int)
    df["Datetime"] = pd.to_datetime(df["Date"], format="%Y%m%d") + pd.to_timedelta(
        df["Time"], unit="m"
    )
    df = df.set_index("Datetime")
    return df


def section1_hourly_aggregate():
    print("\n" + "=" * 70)
    print("SECTION 1: Hourly Aggregation — Initial Exploration")
    print("=" * 70)
    df = load_raw_reading_data()
    df_hourly = df[["U10", "U10max"]].resample("1h").mean().dropna()
    print(df_hourly.head(10))
    print("Shape:", df_hourly.shape)
    print("Min U10:", df_hourly["U10"].min())
    print("Max U10:", df_hourly["U10"].max())
    print("Mean U10:", df_hourly["U10"].mean())


# ==============================================================================
# SECTION 2 — Hourly time series plot + CSV export (REQUIRED by Sections
# 4, 5, 8)
# ==============================================================================
def section2_hourly_plot():
    print("\n" + "=" * 70)
    print("SECTION 2: Hourly Time Series Plot + CSV Export")
    print("=" * 70)
    df = load_raw_reading_data()
    df_hourly = df[["U10", "U10max"]].resample("1h").mean().dropna()
    df_hourly.to_csv(DATA_PATH_HOURLY)
    print("Hourly CSV saved:", DATA_PATH_HOURLY)

    plt.figure(figsize=(14, 4))
    plt.plot(df_hourly.index, df_hourly["U10"], color="steelblue", linewidth=0.5)
    plt.xlabel("Date")
    plt.ylabel("Wind Speed (m/s)")
    plt.title("10m Mean Wind Speed - University of Reading 2023 (Hourly)")
    plt.tight_layout()
    plt.savefig(plot_path("wind_timeseries_hourly.png"), dpi=150)
    plt.close()
    print("Plot saved.")


# ==============================================================================
# SECTION 3 — Weibull fit + ACF/alpha fit + SDE Models I, II, III
# Returns fitted parameters needed by later sections.
# ==============================================================================
def section3_weibull_acf_sde_models():
    print("\n" + "=" * 70)
    print("SECTION 3: Weibull Fit + ACF/Alpha Fit + SDE Models I, II, III")
    print("=" * 70)

    df = load_raw_reading_data()
    df_hourly = df[["U10"]].resample("1h").mean().dropna()
    u = df_hourly["U10"].dropna()
    u_pos = u.values[u.values > 0]
    print(f"Hourly observations: {len(u)}")

    # --- Figure 1: Weibull MLE fit ---
    k, loc, lam = weibull_min.fit(u_pos, floc=0)
    print(f"Weibull MLE: k={k:.4f}, lambda={lam:.4f} m/s")
    x_pdf = np.linspace(0, u_pos.max() + 1, 300)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(u_pos, bins=60, density=True, color="steelblue", alpha=0.5, label="Data (hourly)")
    ax.plot(x_pdf, gaussian_kde(u_pos)(x_pdf), "b-", linewidth=1.5, label="KDE")
    ax.plot(
        x_pdf,
        weibull_min.pdf(x_pdf, k, scale=lam),
        "k-",
        linewidth=2,
        label=f"Weibull MLE (k={k:.4f}, \u03bb={lam:.4f})",
    )
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Probability density")
    ax.set_title("Probability density of hourly mean wind speed - Reading 2023")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_path("fig1_weibull_mle.png"), dpi=150)
    plt.close()
    print("Fig 1 saved")

    # --- Figure 2: ACF + exponential fit -> alpha ---
    max_lag = 120
    acf_vals = acf(u.values, nlags=max_lag, fft=True)
    lags = np.arange(0, max_lag + 1)

    def exp_decay(tau, alpha):
        return np.exp(-alpha * tau)

    popt, _ = curve_fit(exp_decay, lags, acf_vals, p0=[0.05], bounds=(0, np.inf))
    alpha_fit = popt[0]
    print(f"Alpha from ACF fit: alpha={alpha_fit:.6f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags, acf_vals, color="grey", linewidth=1.2, label="Data")
    ax.plot(
        lags,
        exp_decay(lags, alpha_fit),
        "k-",
        linewidth=2,
        label=f"Exponential fit (\u03b1={alpha_fit:.4f})",
    )
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time lag [h]")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of hourly mean wind speed - Reading 2023")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_path("fig2_acf_exponential.png"), dpi=150)
    plt.close()
    print("Fig 2 saved")

    # --- Weibull moments ---
    mu_W = lam * Gamma(1 + 1 / k)
    var_W = lam ** 2 * Gamma(1 + 2 / k) - mu_W ** 2
    sigma_W = np.sqrt(var_W)
    print(f"mu_W={mu_W:.4f}, sigma_W={sigma_W:.4f}")

    # --- SDE parameters ---
    alpha = alpha_fit
    sigma_ou = np.sqrt(2 * alpha)
    b3 = np.sqrt(2 * alpha) * sigma_W
    u0 = lam * (np.log(2)) ** (1 / k)

    dt_fine = 0.1
    N_fine = 8760 * 10
    step_per_hour = 10
    N_SIM = 1000

    print(f"\nRunning {N_SIM} simulations with dt={dt_fine} for each model...")
    print("(This is the slow step — see runtime note at the top of this file.)")

    U1_all = np.zeros((N_SIM, 8760))
    U2_all = np.zeros((N_SIM, 8760))
    U3_all = np.zeros((N_SIM, 8760))

    def b2_model2(uv):
        eps = 1e-9
        uv = np.maximum(uv, eps)
        pw = np.maximum(weibull_min.pdf(uv, k, scale=lam), eps)
        a_val = 1 + 1 / k
        z_val = (uv / lam) ** k
        inc_gam = Gamma(a_val) * (1 - gammainc(a_val, z_val))
        term = lam * inc_gam - mu_W * np.exp(-z_val)
        return np.maximum((2 * alpha / pw) * term, 0.0)

    for sim in range(N_SIM):
        if sim % 100 == 0:
            print(f"  Simulation {sim}/{N_SIM}...")

        np.random.seed(sim)
        dW = np.random.normal(0, np.sqrt(dt_fine), N_fine)

        # Model I: memoryless Weibull transformation of an OU process
        X = np.zeros(N_fine)
        for i in range(1, N_fine):
            X[i] = X[i - 1] - alpha * X[i - 1] * dt_fine + sigma_ou * dW[i]
        U1_fine = weibull_min.ppf(
            np.clip(sp_norm.cdf(X), 1e-9, 1 - 1e-9), k, scale=lam
        )
        U1_all[sim] = U1_fine[::step_per_hour]

        # Model II: Fokker-Planck drift-first
        U2 = np.zeros(N_fine)
        U2[0] = u0
        for i in range(1, N_fine):
            uv = max(U2[i - 1], 0.001)
            a_ = -alpha * (uv - mu_W)
            b_ = np.sqrt(b2_model2(uv))
            U2[i] = max(uv + a_ * dt_fine + b_ * dW[i], 0.001)
        U2_all[sim] = U2[::step_per_hour]

        # Model III: Fokker-Planck diffusion-first (constant diffusion),
        # reflecting boundary at zero
        U3 = np.zeros(N_fine)
        U3[0] = u0
        for i in range(1, N_fine):
            uv = max(U3[i - 1], 0.001)
            a_ = alpha * sigma_W ** 2 * (k / uv) * ((k - 1) / k - (uv / lam) ** k)
            step = uv + a_ * dt_fine + b3 * dW[i]
            U3[i] = abs(step) if step < 0.01 else step
        U3_all[sim] = U3[::step_per_hour]

    print("All simulations done.")
    print(f"Model I   mean={U1_all.mean():.4f}  std={U1_all.std():.4f}")
    print(f"Model II  mean={U2_all.mean():.4f}  std={U2_all.std():.4f}")
    print(f"Model III mean={U3_all.mean():.4f}  std={U3_all.std():.4f}")

    # --- Figure 3: single trajectory, shared Wiener increments ---
    np.random.seed(7)
    dW_shared = np.random.normal(0, np.sqrt(dt_fine), N_fine)

    X = np.zeros(N_fine)
    for i in range(1, N_fine):
        X[i] = X[i - 1] - alpha * X[i - 1] * dt_fine + sigma_ou * dW_shared[i]
    U1_single = weibull_min.ppf(
        np.clip(sp_norm.cdf(X), 1e-9, 1 - 1e-9), k, scale=lam
    )[::step_per_hour]

    U2_single = np.zeros(N_fine)
    U2_single[0] = u0
    for i in range(1, N_fine):
        uv = max(U2_single[i - 1], 0.001)
        a_ = -alpha * (uv - mu_W)
        b_ = np.sqrt(b2_model2(uv))
        U2_single[i] = max(uv + a_ * dt_fine + b_ * dW_shared[i], 0.001)
    U2_single = U2_single[::step_per_hour]

    U3_single = np.zeros(N_fine)
    U3_single[0] = u0
    for i in range(1, N_fine):
        uv = max(U3_single[i - 1], 0.001)
        a_ = alpha * sigma_W ** 2 * (k / uv) * ((k - 1) / k - (uv / lam) ** k)
        step = uv + a_ * dt_fine + b3 * dW_shared[i]
        U3_single[i] = abs(step) if step < 0.01 else step
    U3_single = U3_single[::step_per_hour]

    t = np.arange(8760)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t[:120], U1_single[:120], label="Model I", linewidth=1.5, color="steelblue")
    ax.plot(t[:120], U2_single[:120], label="Model II", linewidth=1.5, color="darkorange")
    ax.plot(t[:120], U3_single[:120], label="Model III", linewidth=1.5, color="green")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Wind speed [m/s]")
    ax.set_title("Wind speed trajectories \u2013 Models I to III")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path("fig3_sde_trajectories.png"), dpi=150)
    plt.close()
    print("Fig 3 saved")

    # --- Figure 4: PDF from 1000 simulations pooled ---
    x_pdf = np.linspace(0, 10, 300)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_pdf, weibull_min.pdf(x_pdf, k, scale=lam), "k-", linewidth=2, label="Weibull PDF")
    for sims, col, lab in zip(
        [U1_all.flatten(), U2_all.flatten(), U3_all.flatten()],
        ["steelblue", "darkorange", "green"],
        ["Model I", "Model II", "Model III"],
    ):
        kde = gaussian_kde(sims)
        ax.plot(x_pdf, kde(x_pdf), color=col, linewidth=1.8, linestyle="--", label=lab)
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Probability density")
    ax.set_title("Probability density of wind speed \u2013 Models I to III (1000 simulations)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_path("fig4_sde_pdf_comparison.png"), dpi=150)
    plt.close()
    print("Fig 4 saved")

    # --- Figure 5: ACF comparison ---
    max_lag = 120
    lags = np.arange(0, max_lag + 1)
    acf1 = acf(U1_all[0], nlags=max_lag, fft=True)
    acf2 = acf(U2_all[0], nlags=max_lag, fft=True)
    acf3 = acf(U3_all[0], nlags=max_lag, fft=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags, np.exp(-alpha * lags), "k-", linewidth=2, label="Exponential function")
    ax.plot(lags, acf1, color="steelblue", linewidth=1.5, linestyle="--", label="Model I")
    ax.plot(lags, acf2, color="darkorange", linewidth=1.5, linestyle="--", label="Model II")
    ax.plot(lags, acf3, color="green", linewidth=1.5, linestyle="--", label="Model III")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time lag [h]")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of wind speed generated by Models I to III")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_path("fig5_acf_comparison.png"), dpi=150)
    plt.close()
    print("Fig 5 saved")

    # --- Save arrays for reuse / further analysis ---
    np.savez(
        data_path("section3_outputs.npz"),
        k=k,
        lam=lam,
        alpha=alpha,
        mu_W=mu_W,
        sigma_W=sigma_W,
        U1_all=U1_all,
        U2_all=U2_all,
        U3_all=U3_all,
        u_observed=u.values,
    )
    print(f"Saved section3 outputs to {data_path('section3_outputs.npz')}")

    return {
        "k": k,
        "lam": lam,
        "alpha": alpha,
        "mu_W": mu_W,
        "sigma_W": sigma_W,
        "U1_all": U1_all,
        "U2_all": U2_all,
        "U3_all": U3_all,
    }


# ==============================================================================
# SECTION 4 — Stationarity test (ADF)
# ==============================================================================
def section4_stationarity_test():
    print("\n" + "=" * 70)
    print("SECTION 4: Stationarity Test (ADF)")
    print("=" * 70)
    df = pd.read_csv(DATA_PATH_HOURLY)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    u = df["U10"].dropna()

    result = adfuller(u)
    print("=== ADF Test - Full Year ===")
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"p-value       : {result[1]:.6f}")
    for key, val in result[4].items():
        print(f"   {key}: {val:.4f}")
    print("Result: STATIONARY" if result[1] < 0.05 else "Result: NON-STATIONARY")

    seasons = {
        "Winter (Q1)": u[u.index.month.isin([1, 2, 3])],
        "Spring (Q2)": u[u.index.month.isin([4, 5, 6])],
        "Summer (Q3)": u[u.index.month.isin([7, 8, 9])],
        "Autumn (Q4)": u[u.index.month.isin([10, 11, 12])],
    }
    print("\n=== ADF Test by Season ===")
    for season, data in seasons.items():
        res = adfuller(data.dropna())
        status = "Stationary" if res[1] < 0.05 else "Non-stationary"
        print(f"{season}: ADF={res[0]:.4f}, p={res[1]:.6f} --> {status}")

    rolling_mean = u.rolling(window=168).mean()
    rolling_std = u.rolling(window=168).std()

    plt.figure(figsize=(14, 5))
    plt.plot(u.index, u.values, color="steelblue", linewidth=0.4, alpha=0.5, label="Hourly wind speed")
    plt.plot(rolling_mean.index, rolling_mean.values, color="red", linewidth=1.5, label="7-day rolling mean")
    plt.plot(rolling_std.index, rolling_std.values, color="green", linewidth=1.5, label="7-day rolling std")
    plt.xlabel("Date")
    plt.ylabel("Wind Speed (m/s)")
    plt.title("Rolling Mean and Standard Deviation -- Stationarity Check")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("stationarity.png"), dpi=150)
    plt.close()
    print("Done.")


# ==============================================================================
# SECTION 5 — Seasonal Weibull analysis
# ==============================================================================
def section5_seasonal_analysis():
    print("\n" + "=" * 70)
    print("SECTION 5: Seasonal Weibull Analysis")
    print("=" * 70)
    df = pd.read_csv(DATA_PATH_HOURLY)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")

    seasons = {
        "Winter (Q1: Jan-Mar)": df[df.index.month.isin([1, 2, 3])]["U10"],
        "Spring (Q2: Apr-Jun)": df[df.index.month.isin([4, 5, 6])]["U10"],
        "Summer (Q3: Jul-Sep)": df[df.index.month.isin([7, 8, 9])]["U10"],
        "Autumn (Q4: Oct-Dec)": df[df.index.month.isin([10, 11, 12])]["U10"],
    }

    print(f'{"Season":<25} {"Mean":>8} {"Std":>8} {"Min":>8} {"Max":>8} {"k":>8} {"lambda":>8}')
    print("-" * 75)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    colors = ["steelblue", "green", "orange", "red"]

    for i, (season, data) in enumerate(seasons.items()):
        u = data.dropna().values
        u_pos = u[u > 0]
        k, loc, lam = weibull_min.fit(u_pos, floc=0)
        print(f"{season:<25} {u.mean():>8.4f} {u.std():>8.4f} {u.min():>8.4f} {u.max():>8.4f} {k:>8.4f} {lam:>8.4f}")

        x = np.linspace(0, u.max() + 1, 200)
        pdf = weibull_min.pdf(x, k, scale=lam)

        axes[i].hist(u_pos, bins=40, density=True, alpha=0.5, color=colors[i])
        axes[i].plot(x, pdf, color="black", linewidth=2, label=f"Weibull (k={k:.3f}, \u03bb={lam:.3f})")
        axes[i].set_title(season)
        axes[i].set_xlabel("Wind Speed (m/s)")
        axes[i].set_ylabel("Probability Density")
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Seasonal Weibull Fits -- University of Reading 2023", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path("seasonal_analysis.png"), dpi=150)
    plt.close()
    print("Done.")


# ==============================================================================
# SECTION 6 — Higham (2001) numerical method validation
# (Brownian motion, stochastic integrals, Euler-Maruyama, strong/weak
# convergence, Milstein's method). These are self-contained and do NOT
# require the Reading dataset.
# ==============================================================================
def _higham_save(fig_name):
    plt.tight_layout()
    plt.savefig(plot_path(fig_name), dpi=150)
    plt.close()


def section6a_brownian_motion():
    print("\n--- 6a: Brownian Motion ---")
    np.random.seed(100)
    T, N = 1, 500
    dt = T / N
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)
    t_grid = np.linspace(0, T, N + 1)

    M = 1000
    dW_paths = np.sqrt(dt) * np.random.randn(M, N)
    W_paths = np.cumsum(dW_paths, axis=1)
    U = np.exp(np.tile(t_grid[1:], (M, 1)) + 0.5 * W_paths)
    U_mean = U.mean(axis=0)
    exact = np.exp(9 * t_grid[1:] / 8)
    max_err = np.max(np.abs(U_mean - exact))
    print(f"   Max error (mean vs exact): {max_err:.4f}  (Higham 2001 gets 0.0504)")

    plt.figure(figsize=(6, 4))
    plt.plot(t_grid, np.insert(W, 0, 0), color="steelblue")
    plt.xlabel("t"); plt.ylabel("W(t)"); plt.title("Discretised Brownian Path")
    _higham_save("01_brownian_motion.png")


def section6b_stochastic_integrals():
    print("\n--- 6b: Stochastic Integrals (It\u00f4 vs Stratonovich) ---")
    np.random.seed(100)
    T, N = 1, 500
    dt = T / N
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)
    W_prev = np.insert(W[:-1], 0, 0)

    ito = np.sum(W_prev * dW)
    strat = np.sum((0.5 * (W_prev + W) + 0.5 * np.sqrt(dt) * np.random.randn(N)) * dW)
    diff = ito - strat
    print(f"   It\u00f4={ito:.4f}, Stratonovich={strat:.4f}, diff={diff:.4f} (expected \u2248 -0.5)")

    plt.figure(figsize=(6, 4))
    plt.bar(["It\u00f4", "Stratonovich"], [ito, strat], color=["steelblue", "darkorange"])
    plt.title("It\u00f4 vs Stratonovich Stochastic Integrals")
    _higham_save("02_stochastic_integrals.png")


def section6c_euler_maruyama():
    print("\n--- 6c: Euler-Maruyama Method ---")
    np.random.seed(100)
    lam, mu, X0 = 2, 1, 1
    T, N = 1, 2 ** 8
    dt = T / N
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)
    t_grid = np.linspace(dt, T, N)
    X_true = X0 * np.exp((lam - 0.5 * mu ** 2) * t_grid + mu * W)

    R = 1
    Dt = R * dt
    L = N // R
    X_em = np.zeros(L)
    X_temp = X0
    for j in range(L):
        Winc = np.sum(dW[R * j:R * (j + 1)])
        X_temp = X_temp + Dt * lam * X_temp + mu * X_temp * Winc
        X_em[j] = X_temp
    err = abs(X_em[-1] - X_true[-1])
    print(f"   Endpoint error (R=1, finest step): {err:.4f} (Higham 2001-style benchmark \u2248 0.0328)")

    plt.figure(figsize=(6, 4))
    plt.plot(t_grid, X_true, "m-", label="Exact")
    plt.plot(t_grid[::R], X_em, "r--*", label="EM approximation")
    plt.legend(); plt.title("Euler-Maruyama vs Exact Solution")
    _higham_save("03_euler_maruyama.png")


def section6d_strong_convergence():
    print("\n--- 6d: Strong Convergence ---")
    np.random.seed(100)
    lam, mu, X0 = 2, 1, 1
    T, N = 1, 2 ** 9
    dt = T / N
    M = 1000
    Dtvals = dt * (2.0 ** np.arange(5))
    Xerr = np.zeros((M, 5))

    for s in range(M):
        dW = np.sqrt(dt) * np.random.randn(N)
        W = np.cumsum(dW)
        X_true = X0 * np.exp((lam - 0.5 * mu ** 2) + mu * W[-1])
        for p in range(5):
            R = 2 ** p
            Dt_p = R * dt
            L = N // R
            X_temp = X0
            for j in range(L):
                Winc = np.sum(dW[R * j:R * (j + 1)])
                X_temp = X_temp + Dt_p * lam * X_temp + mu * X_temp * Winc
            Xerr[s, p] = abs(X_temp - X_true)

    mean_err = Xerr.mean(axis=0)
    A = np.column_stack([np.ones(5), np.log(Dtvals)])
    _, q = np.linalg.lstsq(A, np.log(mean_err), rcond=None)[0]
    print(f"   Fitted strong convergence order q = {q:.4f} (theory: 0.5)")

    plt.figure(figsize=(6, 4))
    plt.loglog(Dtvals, mean_err, "b*-", label="Simulated")
    plt.loglog(Dtvals, Dtvals ** 0.5, "r--", label="Reference slope 1/2")
    plt.legend(); plt.title("Strong Convergence of Euler-Maruyama")
    _higham_save("04_strong_convergence.png")


def section6e_weak_convergence():
    print("\n--- 6e: Weak Convergence ---")
    np.random.seed(100)
    lam, mu = 2, 0.1
    T = 1
    M = 50000
    Xem = np.zeros(5)
    Dtvals = 2.0 ** (np.arange(1, 6) - 10)

    for p, Dt in enumerate(Dtvals):
        L = int(T / Dt)
        Xtemp = np.ones(M)
        for j in range(L):
            Winc = np.sqrt(Dt) * np.random.randn(M)
            Xtemp = Xtemp + Dt * lam * Xtemp + mu * Xtemp * Winc
        Xem[p] = Xtemp.mean()

    exact = np.exp(lam * T)
    Xerr = np.abs(Xem - exact)
    A = np.column_stack([np.ones(5), np.log(Dtvals)])
    _, q = np.linalg.lstsq(A, np.log(Xerr), rcond=None)[0]
    print(f"   Fitted weak convergence order q = {q:.4f} (theory: 1.0)")

    plt.figure(figsize=(6, 4))
    plt.loglog(Dtvals, Xerr, "b*-", label="Simulated")
    plt.loglog(Dtvals, Dtvals, "r--", label="Reference slope 1")
    plt.legend(); plt.title("Weak Convergence of Euler-Maruyama")
    _higham_save("05_weak_convergence.png")


def section6f_milstein():
    print("\n--- 6f: Milstein's Method ---")
    np.random.seed(100)
    r, K, beta, X0 = 2, 1, 0.25, 0.5
    T, N = 1, 2 ** 11
    dt = T / N
    M = 500
    Rvals = np.array([1, 16, 32, 64, 128])

    dW_all = np.sqrt(dt) * np.random.randn(M, N)
    Xmil = np.zeros((M, 5))

    for p, R in enumerate(Rvals):
        Dt_p = R * dt
        L = N // R
        Xtemp = np.full(M, X0)
        for j in range(L):
            Winc = dW_all[:, R * j:R * (j + 1)].sum(axis=1)
            Xtemp = (
                Xtemp
                + Dt_p * r * Xtemp * (K - Xtemp)
                + beta * Xtemp * Winc
                + 0.5 * beta ** 2 * Xtemp * (Winc ** 2 - Dt_p)
            )
        Xmil[:, p] = Xtemp

    Xref = Xmil[:, 0]
    Xerr = np.abs(Xmil[:, 1:] - Xref[:, None]).mean(axis=0)
    Dtvals = dt * Rvals[1:]
    A = np.column_stack([np.ones(4), np.log(Dtvals)])
    _, q = np.linalg.lstsq(A, np.log(Xerr), rcond=None)[0]
    print(f"   Fitted Milstein strong convergence order q = {q:.4f} (theory: 1.0)")

    plt.figure(figsize=(6, 4))
    plt.loglog(Dtvals, Xerr, "b*-", label="Simulated")
    plt.loglog(Dtvals, Dtvals, "r--", label="Reference slope 1")
    plt.legend(); plt.title("Milstein's Method: Strong Convergence")
    _higham_save("06_milstein.png")


def section6_higham_scripts():
    print("\n" + "=" * 70)
    print("SECTION 6: Higham (2001) Numerical Methods Validation")
    print("=" * 70)
    section6a_brownian_motion()
    section6b_stochastic_integrals()
    section6c_euler_maruyama()
    section6d_strong_convergence()
    section6e_weak_convergence()
    section6f_milstein()


# ==============================================================================
# SECTION 7 — Robustness analysis (+/-10% parameter perturbation)
# ==============================================================================
def section7_robustness(sde_results=None):
    print("\n" + "=" * 70)
    print("SECTION 7: Robustness Analysis (\u00b110% parameter perturbation)")
    print("=" * 70)
    if sde_results is not None:
        k_base, lam_base = sde_results["k"], sde_results["lam"]
    else:
        k_base, lam_base = 1.9133, 2.7140  # fallback if Section 3 was skipped

    perturbations = {
        f"Base (k={k_base:.3f}, \u03bb={lam_base:.3f})": (k_base, lam_base),
        "k +10%": (k_base * 1.1, lam_base),
        "k -10%": (k_base * 0.9, lam_base),
        "\u03bb +10%": (k_base, lam_base * 1.1),
        "\u03bb -10%": (k_base, lam_base * 0.9),
    }

    np.random.seed(42)
    N, dt = 8760, 0.01
    alpha = 1.0
    sigma = np.sqrt(2 * alpha)

    X = np.zeros(N)
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        X[i] = X[i - 1] - alpha * X[i - 1] * dt + sigma * dW
    U_norm = np.clip(1 / (1 + np.exp(-X)), 1e-6, 1 - 1e-6)

    results = {}
    for label, (k, lam) in perturbations.items():
        sim = weibull_min.ppf(U_norm, k, scale=lam)
        results[label] = sim
        print(f"{label}: Mean={sim.mean():.4f}, Std={sim.std():.4f}, Min={sim.min():.4f}, Max={sim.max():.4f}")

    x = np.linspace(0, 10, 300)
    plt.figure(figsize=(10, 5))
    colors = ["black", "steelblue", "darkorange", "green", "red"]
    styles = ["-", "--", "--", ":", ":"]
    for (label, sim), color, style in zip(results.items(), colors, styles):
        kde = gaussian_kde(sim)
        plt.plot(x, kde(x), color=color, linestyle=style, linewidth=1.8, label=label)
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Probability Density")
    plt.title("Robustness Analysis: Effect of Parameter Perturbation on SDE Model I")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_path("robustness.png"), dpi=150)
    plt.close()
    print("Done.")


# ==============================================================================
# SECTION 8 — ARIMA, GARCH, LSTM benchmarks
# Returns real fitted mean/std for each, used to build the comparison table.
# ==============================================================================
def section8_benchmarks():
    print("\n" + "=" * 70)
    print("SECTION 8: ARIMA, GARCH, LSTM Benchmarks")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH_HOURLY)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    u = df["U10"].values

    # --- ARIMA ---
    print("\n--- ARIMA(2,0,2) ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(u, lags=40, ax=axes[0])
    plot_pacf(u, lags=40, ax=axes[1])
    plt.suptitle("ACF and PACF of Hourly Wind Speed")
    plt.tight_layout()
    plt.savefig(plot_path("acf_pacf.png"), dpi=150)
    plt.close()

    arima_model = ARIMA(u, order=(2, 0, 2))
    arima_result = arima_model.fit()
    try:
        # statsmodels >= 0.15 renamed random_state -> rng
        arima_sim = np.abs(
            arima_result.simulate(nsimulations=8760, rng=np.random.default_rng(42))
        )
    except TypeError:
        # older statsmodels versions
        arima_sim = np.abs(arima_result.simulate(nsimulations=8760, random_state=42))
    arima_mean, arima_std = arima_sim.mean(), arima_sim.std()
    print(f"ARIMA simulation - Mean: {arima_mean:.4f}, Std: {arima_std:.4f}")

    plt.figure(figsize=(14, 4))
    plt.plot(arima_sim[:500], color="purple", linewidth=0.8, label="ARIMA(2,0,2) simulation")
    plt.xlabel("Time (hours)"); plt.ylabel("Wind Speed (m/s)")
    plt.title("ARIMA Model Simulation - First 500 hours")
    plt.legend(); plt.tight_layout()
    plt.savefig(plot_path("arima_simulation.png"), dpi=150)
    plt.close()

    # --- GARCH ---
    print("\n--- GARCH(1,1) ---")
    u_scaled = u * 100
    garch_model_obj = arch_model(u_scaled, vol="Garch", p=1, q=1, mean="constant", dist="normal")
    garch_result = garch_model_obj.fit(disp="off")
    sim = garch_model_obj.simulate(garch_result.params, nobs=8760)
    garch_sim = abs(sim["data"].values) / 100
    garch_mean, garch_std = garch_sim.mean(), garch_sim.std()
    print(f"GARCH simulation - Mean: {garch_mean:.4f}, Std: {garch_std:.4f}")

    plt.figure(figsize=(14, 4))
    plt.plot(garch_sim[:500], color="red", linewidth=0.8, label="GARCH(1,1) simulation")
    plt.xlabel("Time (hours)"); plt.ylabel("Wind Speed (m/s)")
    plt.title("GARCH Model Simulation - First 500 hours")
    plt.legend(); plt.tight_layout()
    plt.savefig(plot_path("garch_simulation.png"), dpi=150)
    plt.close()

    # --- LSTM ---
    print("\n--- LSTM ---")
    u_col = df["U10"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    u_scaled2 = scaler.fit_transform(u_col)

    look_back = 24
    X, y = [], []
    for i in range(look_back, len(u_scaled2)):
        X.append(u_scaled2[i - look_back:i, 0])
        y.append(u_scaled2[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lstm_model_obj = Sequential()
    lstm_model_obj.add(LSTM(50, return_sequences=False, input_shape=(look_back, 1)))
    lstm_model_obj.add(Dense(1))
    lstm_model_obj.compile(optimizer="adam", loss="mse")
    lstm_model_obj.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    pred_scaled = lstm_model_obj.predict(X_test)
    lstm_pred = scaler.inverse_transform(pred_scaled).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    lstm_mean, lstm_std = lstm_pred.mean(), lstm_pred.std()
    print(f"LSTM prediction - Mean: {lstm_mean:.4f}, Std: {lstm_std:.4f}")

    plt.figure(figsize=(14, 4))
    plt.plot(actual[:500], color="steelblue", linewidth=0.8, label="Actual")
    plt.plot(lstm_pred[:500], color="orange", linewidth=0.8, label="LSTM predicted")
    plt.xlabel("Time (hours)"); plt.ylabel("Wind Speed (m/s)")
    plt.title("LSTM Model - Predicted vs Actual Wind Speed")
    plt.legend(); plt.tight_layout()
    plt.savefig(plot_path("lstm_prediction.png"), dpi=150)
    plt.close()

    return {
        "arima_mean": arima_mean, "arima_std": arima_std,
        "garch_mean": garch_mean, "garch_std": garch_std,
        "lstm_mean": lstm_mean, "lstm_std": lstm_std,
    }


# ==============================================================================
# SECTION 9 — Benchmark comparison table (built from real outputs above)
# ==============================================================================
def section9_comparison_table(sde_results, benchmark_results):
    print("\n" + "=" * 70)
    print("SECTION 9: Benchmark Comparison Table")
    print("=" * 70)

    u1_mean, u1_std = sde_results["U1_all"].mean(), sde_results["U1_all"].std()
    u2_mean, u2_std = sde_results["U2_all"].mean(), sde_results["U2_all"].std()
    u3_mean, u3_std = sde_results["U3_all"].mean(), sde_results["U3_all"].std()

    rows = [
        ("SDE Model I", u1_mean, u1_std),
        ("SDE Model II", u2_mean, u2_std),
        ("SDE Model III", u3_mean, u3_std),
        ("ARIMA(2,0,2)", benchmark_results["arima_mean"], benchmark_results["arima_std"]),
        ("GARCH(1,1)", benchmark_results["garch_mean"], benchmark_results["garch_std"]),
        ("LSTM", benchmark_results["lstm_mean"], benchmark_results["lstm_std"]),
    ]

    print(f'{"Model":<18} {"Mean (m/s)":>12} {"Std (m/s)":>12}')
    print("-" * 45)
    for name, mean, std in rows:
        print(f"{name:<18} {mean:>12.4f} {std:>12.4f}")


# ==============================================================================
# MAIN — run everything in order
# ==============================================================================
########################################################################
# PART 2: CHAPTER 6 VALIDATION TASKS 1-8
#
# All eight tasks share a common pattern: they load the cached simulation
# output from Section 3 (Dataset/section3_outputs.npz), so Section 3 MUST
# be run at least once before any of Tasks 1-8 can run.
########################################################################

from scipy.stats import kstest, probplot, skew
from scipy.optimize import minimize_scalar


def _cache_path():
    return data_path("section3_outputs.npz")


def _load_cached_section3():
    cpath = _cache_path()
    if not os.path.exists(cpath):
        raise FileNotFoundError(
            f"{cpath} not found. Run section3_weibull_acf_sde_models() first "
            "(it is called automatically by main() before the Task functions)."
        )
    data = np.load(cpath)
    return {
        "k": float(data["k"]),
        "lam": float(data["lam"]),
        "alpha": float(data["alpha"]),
        "mu_W": float(data["mu_W"]),
        "sigma_W": float(data["sigma_W"]),
        "u_observed": data["u_observed"],
        "U1_all": data["U1_all"],
        "U2_all": data["U2_all"],
        "U3_all": data["U3_all"],
    }


# ----------------------------------------------------------------------
# TASK 1 — Goodness of Fit
# ----------------------------------------------------------------------
def _task1_ks_test_vs_weibull(sample, k, lam):
    D, p = kstest(sample, "weibull_min", args=(k, 0, lam))
    return D, p


def _task1_density_rmse(sample, k, lam, x_grid=None):
    if x_grid is None:
        x_grid = np.linspace(0, max(sample.max(), 10), 300)
    kde = gaussian_kde(sample)
    empirical_density = kde(x_grid)
    fitted_density = weibull_min.pdf(x_grid, k, scale=lam)
    rmse = np.sqrt(np.mean((empirical_density - fitted_density) ** 2))
    return rmse


def _task1_make_qq_plot(sample, k, lam, label, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    probplot(sample, dist=weibull_min(k, scale=lam), plot=ax)
    ax.set_title(f"QQ Plot: {label} vs Weibull(k={k:.3f}, lam={lam:.3f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _task1_make_cdf_plot(sample, k, lam, label, save_path):
    x_sorted = np.sort(sample)
    empirical_cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    model_cdf = weibull_min.cdf(x_sorted, k, scale=lam)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_sorted, empirical_cdf, color="steelblue", linewidth=1.5, label="Empirical CDF")
    ax.plot(x_sorted, model_cdf, color="black", linewidth=1.5, linestyle="--", label="Weibull model CDF")
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"Empirical vs Model CDF: {label}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _task1_make_histogram_plot(sample, k, lam, label, save_path):
    x_grid = np.linspace(0, max(sample.max(), 10), 300)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sample, bins=60, density=True, color="steelblue", alpha=0.5, label=f"{label} (simulated/observed)")
    ax.plot(x_grid, weibull_min.pdf(x_grid, k, scale=lam), "k-", linewidth=2, label="Fitted Weibull density")
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Histogram with fitted stationary density: {label}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def task1_goodness_of_fit():
    print("\n" + "=" * 70)
    print("TASK 1: Goodness of Fit of the Stationary Distribution")
    print("=" * 70)
    cached = _load_cached_section3()
    k, lam = cached["k"], cached["lam"]
    print(f"Using fitted Weibull parameters: k={k:.4f}, lambda={lam:.4f}")

    samples = {
        "Observed data": cached["u_observed"],
        "Model I (simulated)": cached["U1_all"].flatten(),
        "Model II (simulated)": cached["U2_all"].flatten(),
        "Model III (simulated)": cached["U3_all"].flatten(),
    }
    tags = {
        "Observed data": "observed",
        "Model I (simulated)": "modelI",
        "Model II (simulated)": "modelII",
        "Model III (simulated)": "modelIII",
    }

    results = []
    for label, sample in samples.items():
        sample = sample[sample > 0]
        if len(sample) > 20000:
            rng = np.random.default_rng(42)
            sample = rng.choice(sample, size=20000, replace=False)

        D, p = _task1_ks_test_vs_weibull(sample, k, lam)
        rmse = _task1_density_rmse(sample, k, lam)
        tag = tags[label]
        _task1_make_histogram_plot(sample, k, lam, label, plot_path(f"task1_hist_{tag}.png"))
        _task1_make_cdf_plot(sample, k, lam, label, plot_path(f"task1_cdf_{tag}.png"))
        _task1_make_qq_plot(sample, k, lam, label, plot_path(f"task1_qq_{tag}.png"))

        print(f"  {label}: KS D={D:.5f}, p={p:.5f}, density RMSE={rmse:.6f}")
        results.append({"Sample": label, "KS_statistic": D, "KS_pvalue": p, "Density_RMSE": rmse})

    summary = pd.DataFrame(results)
    print("\n=== Task 1 Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(data_path("task1_goodness_of_fit_summary.csv"), index=False)
    return summary


# ----------------------------------------------------------------------
# TASK 2 — Time Dependence / ACF Distribution + Deseasonalisation Fix
# ----------------------------------------------------------------------
def _task2_compute_acf_distribution(all_trajectories, max_lag=120, n_use=1000):
    n_use = min(n_use, all_trajectories.shape[0])
    acf_matrix = np.zeros((n_use, max_lag + 1))
    for i in range(n_use):
        acf_matrix[i] = acf(all_trajectories[i], nlags=max_lag, fft=True)
    return acf_matrix


def task2_acf_distribution():
    print("\n" + "=" * 70)
    print("TASK 2: Time Dependence -- ACF Distribution Across Trajectories")
    print("=" * 70)
    cached = _load_cached_section3()
    alpha = cached["alpha"]
    max_lag = 120
    observed_acf = acf(cached["u_observed"], nlags=max_lag, fft=True)

    models = [
        ("Model I", cached["U1_all"], "steelblue"),
        ("Model II", cached["U2_all"], "darkorange"),
        ("Model III", cached["U3_all"], "green"),
    ]

    summary_rows = []
    for label, all_sims, color in models:
        print(f"\nComputing ACF for {label} trajectories...")
        acf_matrix = _task2_compute_acf_distribution(all_sims, max_lag=max_lag)
        acf_lower = np.percentile(acf_matrix, 2.5, axis=0)
        acf_upper = np.percentile(acf_matrix, 97.5, axis=0)
        within = (observed_acf >= acf_lower) & (observed_acf <= acf_upper)
        coverage = within.mean()
        print(f"  {label}: observed ACF within simulated 95% band for {coverage:.1%} of lags")

        lags = np.arange(max_lag + 1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.fill_between(lags, acf_lower, acf_upper, color=color, alpha=0.25, label="95% band")
        ax.plot(lags, acf_matrix.mean(axis=0), color=color, linewidth=2, label="Mean simulated ACF")
        ax.plot(lags, observed_acf, color="black", linewidth=2, linestyle="--", label="Observed ACF")
        ax.plot(lags, np.exp(-alpha * lags), color="grey", linewidth=1.5, linestyle=":", label=f"exp(-\u03b1t), \u03b1={alpha:.4f}")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Time lag [h]"); ax.set_ylabel("Autocorrelation")
        ax.set_title(f"ACF Distribution: {label}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_path(f"task2_acf_distribution_{label.replace(' ', '')}.png"), dpi=150)
        plt.close()

        summary_rows.append({"Model": label, "Observed_ACF_Coverage_95pct": coverage})

    summary = pd.DataFrame(summary_rows)
    print("\n=== Task 2 Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(data_path("task2_acf_coverage_summary.csv"), index=False)
    return summary


def _task2_fit_diurnal_component(series, hour_of_day):
    t = hour_of_day.values
    X = np.column_stack([
        np.sin(2 * np.pi * t / 24), np.cos(2 * np.pi * t / 24),
        np.sin(4 * np.pi * t / 24), np.cos(4 * np.pi * t / 24),
    ])
    X = np.column_stack([np.ones(len(t)), X])
    coeffs, _, _, _ = np.linalg.lstsq(X, series.values, rcond=None)
    seasonal = X @ coeffs
    return seasonal, coeffs


def _task2_build_seasonal_lookup(coeffs):
    def seasonal_fn(hour_of_day):
        t = np.asarray(hour_of_day)
        X = np.column_stack([
            np.ones_like(t, dtype=float),
            np.sin(2 * np.pi * t / 24), np.cos(2 * np.pi * t / 24),
            np.sin(4 * np.pi * t / 24), np.cos(4 * np.pi * t / 24),
        ])
        return X @ coeffs
    return seasonal_fn


def task2_deseasonalize_fix(sde_results):
    print("\n" + "=" * 70)
    print("TASK 2 FIX: Deseasonalised SDE Modelling (All Three Models)")
    print("=" * 70)
    df = load_raw_reading_data()
    series = df[["U10"]].resample("1h").mean().dropna()["U10"]
    n_hours = len(series)
    hour_of_day = pd.Series(series.index.hour, index=series.index)

    print("Fitting diurnal (24h) seasonal component...")
    seasonal, coeffs = _task2_fit_diurnal_component(series, hour_of_day)
    seasonal_fn = _task2_build_seasonal_lookup(coeffs)
    residual = series.values - seasonal

    k, lam = sde_results["k"], sde_results["lam"]
    mu_W, sigma_W = sde_results["mu_W"], sde_results["sigma_W"]

    max_lag = 120
    acf_vals = acf(residual, nlags=max_lag, fft=True)
    lags = np.arange(0, max_lag + 1)

    def exp_decay(tau, a):
        return np.exp(-a * tau)

    popt, _ = curve_fit(exp_decay, lags, acf_vals, p0=[0.05], bounds=(0, np.inf))
    alpha_resid = popt[0]
    print(f"Residual alpha (deseasonalised): {alpha_resid:.6f}")

    hours_array = np.arange(n_hours) % 24
    seasonal_repeat = seasonal_fn(hours_array)
    seasonal_centered = seasonal_repeat - seasonal_repeat.mean()

    observed_acf = acf(series.values, nlags=max_lag, fft=True)
    original_coverage = {"Model I": 0.653, "Model II": 0.678, "Model III": 0.744}

    dt_fine, step_per_hour, n_sim = 0.1, 10, 1000
    n_fine = n_hours * step_per_hour
    u0 = lam * (np.log(2)) ** (1 / k)
    sigma_ou = np.sqrt(2 * alpha_resid)
    b3 = np.sqrt(2 * alpha_resid) * sigma_W

    def b2_model2(uv):
        eps = 1e-9
        uv = np.maximum(uv, eps)
        pw = np.maximum(weibull_min.pdf(uv, k, scale=lam), eps)
        a_val = 1 + 1 / k
        z_val = (uv / lam) ** k
        inc_gam = Gamma(a_val) * (1 - gammainc(a_val, z_val))
        term = lam * inc_gam - mu_W * np.exp(-z_val)
        return np.maximum((2 * alpha_resid / pw) * term, 0.0)

    def sim_model1():
        sims = np.zeros((n_sim, n_hours))
        for s in range(n_sim):
            np.random.seed(s)
            dW = np.random.normal(0, np.sqrt(dt_fine), n_fine)
            X = np.zeros(n_fine)
            for i in range(1, n_fine):
                X[i] = X[i - 1] - alpha_resid * X[i - 1] * dt_fine + sigma_ou * dW[i]
            sims[s] = weibull_min.ppf(np.clip(sp_norm.cdf(X), 1e-9, 1 - 1e-9), k, scale=lam)[::step_per_hour][:n_hours]
        return sims

    def sim_model2():
        sims = np.zeros((n_sim, n_hours))
        for s in range(n_sim):
            np.random.seed(s)
            dW = np.random.normal(0, np.sqrt(dt_fine), n_fine)
            U2 = np.zeros(n_fine); U2[0] = u0
            for i in range(1, n_fine):
                uv = max(U2[i - 1], 0.001)
                a_ = -alpha_resid * (uv - mu_W)
                b_ = np.sqrt(b2_model2(uv))
                U2[i] = max(uv + a_ * dt_fine + b_ * dW[i], 0.001)
            sims[s] = U2[::step_per_hour][:n_hours]
        return sims

    def sim_model3():
        sims = np.zeros((n_sim, n_hours))
        for s in range(n_sim):
            np.random.seed(s)
            dW = np.random.normal(0, np.sqrt(dt_fine), n_fine)
            U3 = np.zeros(n_fine); U3[0] = u0
            for i in range(1, n_fine):
                uv = max(U3[i - 1], 0.001)
                a_ = alpha_resid * sigma_W ** 2 * (k / uv) * ((k - 1) / k - (uv / lam) ** k)
                step = uv + a_ * dt_fine + b3 * dW[i]
                U3[i] = abs(step) if step < 0.01 else step
            sims[s] = U3[::step_per_hour][:n_hours]
        return sims

    model_runners = {"Model I": sim_model1, "Model II": sim_model2, "Model III": sim_model3}
    colors = {"Model I": "steelblue", "Model II": "darkorange", "Model III": "green"}
    summary_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for (label, runner), ax in zip(model_runners.items(), axes):
        print(f"\nSimulating {label} on deseasonalised residual timescale...")
        sims_raw = runner()
        sims_reseasonalized = np.clip(sims_raw + seasonal_centered, 0, None)

        acf_matrix = np.zeros((n_sim, max_lag + 1))
        for i in range(n_sim):
            acf_matrix[i] = acf(sims_reseasonalized[i], nlags=max_lag, fft=True)

        acf_mean = acf_matrix.mean(axis=0)
        acf_lower = np.percentile(acf_matrix, 2.5, axis=0)
        acf_upper = np.percentile(acf_matrix, 97.5, axis=0)
        within = (observed_acf >= acf_lower) & (observed_acf <= acf_upper)
        coverage = within.mean()
        print(f"  {label} NEW coverage: {coverage:.1%} (was {original_coverage[label]:.1%})")
        summary_rows.append({"Model": label, "Original_Coverage": original_coverage[label], "Deseasonalized_Coverage": coverage})

        color = colors[label]
        ax.fill_between(lags, acf_lower, acf_upper, color=color, alpha=0.25, label="95% band")
        ax.plot(lags, acf_mean, color=color, linewidth=2, label="Mean simulated ACF")
        ax.plot(lags, observed_acf, color="black", linewidth=2, linestyle="--", label="Observed")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Time lag [h]")
        ax.set_title(f"{label} (coverage={coverage:.1%})")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Autocorrelation")
    fig.suptitle("Deseasonalised + Reseasonalised: All Three Models", fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_path("task2_deseasonalized_fix_all_models.png"), dpi=150)
    plt.close()

    summary = pd.DataFrame(summary_rows)
    print("\n=== Deseasonalisation Fix Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(data_path("task2_deseasonalization_summary.csv"), index=False)
    return summary


# ----------------------------------------------------------------------
# TASK 3 — Monte Carlo Validation
# ----------------------------------------------------------------------
def _task3_compute_trajectory_stats(all_trajectories, thresholds=(3, 5, 7)):
    n_sim = all_trajectories.shape[0]
    records = []
    for i in range(n_sim):
        traj = all_trajectories[i]
        record = {"mean": traj.mean(), "variance": traj.var(), "skewness": skew(traj), "max": traj.max()}
        for thresh in thresholds:
            record[f"exceedance_p_{thresh}ms"] = (traj > thresh).mean()
        records.append(record)
    return pd.DataFrame(records)


def _task3_compute_observed_stats(observed, thresholds=(3, 5, 7)):
    record = {"mean": observed.mean(), "variance": observed.var(), "skewness": skew(observed), "max": observed.max()}
    for thresh in thresholds:
        record[f"exceedance_p_{thresh}ms"] = (observed > thresh).mean()
    return record


def _task3_percentile_rank(observed_value, simulated_values):
    return (simulated_values < observed_value).mean() * 100


def task3_montecarlo_validation():
    print("\n" + "=" * 70)
    print("TASK 3: Monte Carlo Validation")
    print("=" * 70)
    cached = _load_cached_section3()
    observed = cached["u_observed"]
    observed_stats = _task3_compute_observed_stats(observed)

    models = {"Model I": cached["U1_all"], "Model II": cached["U2_all"], "Model III": cached["U3_all"]}
    thresholds = (3, 5, 7)
    print("Computing per-trajectory statistics for each model...")
    model_stats_dfs = {label: _task3_compute_trajectory_stats(sims, thresholds) for label, sims in models.items()}

    stat_columns = ["mean", "variance", "skewness", "max"] + [f"exceedance_p_{t}ms" for t in thresholds]
    summary_rows = []
    for stat in stat_columns:
        row = {"Statistic": stat, "Observed": observed_stats[stat]}
        for label in models:
            sim_values = model_stats_dfs[label][stat].values
            pct = _task3_percentile_rank(observed_stats[stat], sim_values)
            row[f"{label}_SimMean"] = sim_values.mean()
            row[f"{label}_ObsPercentile"] = pct
        summary_rows.append(row)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
        colors = {"Model I": "steelblue", "Model II": "darkorange", "Model III": "green"}
        for ax, label in zip(axes, models):
            sim_values = model_stats_dfs[label][stat].values
            pct = _task3_percentile_rank(observed_stats[stat], sim_values)
            ax.hist(sim_values, bins=40, color=colors[label], alpha=0.6, density=True)
            ax.axvline(observed_stats[stat], color="black", linewidth=2, linestyle="--", label=f"Observed ({observed_stats[stat]:.3f})")
            ax.set_title(f"{label}: observed at {pct:.1f}th %ile")
            ax.set_xlabel(stat)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Density")
        plt.tight_layout()
        plt.savefig(plot_path(f"task3_montecarlo_{stat}.png"), dpi=150)
        plt.close()

    summary = pd.DataFrame(summary_rows)
    print("\n=== Task 3 Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(data_path("task3_montecarlo_summary.csv"), index=False)

    print("\n=== Flags: statistics outside the 95% range ===")
    any_flags = False
    for _, row in summary.iterrows():
        for label in models:
            pct = row[f"{label}_ObsPercentile"]
            if pct < 2.5 or pct > 97.5:
                print(f"  {label} - {row['Statistic']}: observed at {pct:.1f}th percentile (OUTSIDE 95%)")
                any_flags = True
    if not any_flags:
        print("  None - all observed statistics fall within the simulated 95% range.")
    return summary


# ----------------------------------------------------------------------
# TASK 4 — Extreme-Event Behaviour
# ----------------------------------------------------------------------
def task4_extreme_events():
    print("\n" + "=" * 70)
    print("TASK 4: Extreme-Event Behaviour")
    print("=" * 70)
    cached = _load_cached_section3()
    observed = cached["u_observed"]
    models = {"Model I": cached["U1_all"], "Model II": cached["U2_all"], "Model III": cached["U3_all"]}
    quantiles = [0.90, 0.95, 0.99, 0.999]
    thresholds = [3, 5, 6, 7, 8]
    return_periods = [2, 5, 10, 20, 50]

    print("Computing high quantiles...")
    quantile_rows = []
    quantile_summary = {"Observed": [np.quantile(observed, q) for q in quantiles]}
    for label, all_sims in models.items():
        pooled = all_sims.flatten()
        pooled = pooled[pooled > 0]
        quantile_summary[label] = [np.quantile(pooled, q) for q in quantiles]
        for q in quantiles:
            quantile_rows.append({"Model": label, "Quantile": f"q{int(q*1000)/10}",
                                   "Observed": np.quantile(observed, q), "Simulated": np.quantile(pooled, q)})
    quantile_df = pd.DataFrame(quantile_rows)
    print(quantile_df.to_string(index=False))

    print("\nComputing exceedance probabilities...")
    exceed_rows = []
    for label, all_sims in models.items():
        pooled = all_sims.flatten()
        pooled = pooled[pooled > 0]
        for t in thresholds:
            exceed_rows.append({"Model": label, "Threshold": f"P(>{t}m/s)",
                                 "Observed": (observed > t).mean(), "Simulated": (pooled > t).mean()})
    exceed_df = pd.DataFrame(exceed_rows)
    print(exceed_df.to_string(index=False))

    print("\nComputing return levels from annual maxima distributions...")
    return_level_rows = []
    for label, all_sims in models.items():
        annual_maxima = all_sims.max(axis=1)
        row = {"Model": label}
        for T in return_periods:
            row[f"{T}yr_return_level"] = np.quantile(annual_maxima, 1 - 1 / T)
        return_level_rows.append(row)
    return_level_df = pd.DataFrame(return_level_rows)
    print(return_level_df.to_string(index=False))

    quantile_df.to_csv(data_path("task4_quantile_comparison.csv"), index=False)
    exceed_df.to_csv(data_path("task4_exceedance_comparison.csv"), index=False)
    return_level_df.to_csv(data_path("task4_return_levels.csv"), index=False)
    return quantile_df, exceed_df, return_level_df


# ----------------------------------------------------------------------
# TASK 5 — Prediction Accuracy (Chronological Train/Test Split)
# ----------------------------------------------------------------------
def task5_prediction_accuracy(n_sim=500):
    print("\n" + "=" * 70)
    print("TASK 5: Prediction Accuracy (Chronological Train/Test Split)")
    print("=" * 70)
    df = load_raw_reading_data()
    series = df[["U10"]].resample("1h").mean().dropna()["U10"]
    n = len(series)
    split_idx = int(n * 0.75)
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]
    print(f"Total: {n} | Train: {len(train)} | Test: {len(test)}")

    u = train.dropna()
    u_pos = u.values[u.values > 0]
    k, loc, lam = weibull_min.fit(u_pos, floc=0)
    max_lag = 120
    acf_vals = acf(u.values, nlags=max_lag, fft=True)
    lags = np.arange(0, max_lag + 1)

    def exp_decay(tau, a):
        return np.exp(-a * tau)

    popt, _ = curve_fit(exp_decay, lags, acf_vals, p0=[0.05], bounds=(0, np.inf))
    alpha = popt[0]
    mu_W = lam * Gamma(1 + 1 / k)
    sigma_W = np.sqrt(lam ** 2 * Gamma(1 + 2 / k) - mu_W ** 2)
    sigma_ou = np.sqrt(2 * alpha)
    b3 = np.sqrt(2 * alpha) * sigma_W
    print(f"Fitted on TRAIN only: k={k:.4f}, lambda={lam:.4f}, alpha={alpha:.6f}")

    def b2_model2(uv):
        eps = 1e-9
        uv = np.maximum(uv, eps)
        pw = np.maximum(weibull_min.pdf(uv, k, scale=lam), eps)
        a_val = 1 + 1 / k
        z_val = (uv / lam) ** k
        inc_gam = Gamma(a_val) * (1 - gammainc(a_val, z_val))
        term = lam * inc_gam - mu_W * np.exp(-z_val)
        return np.maximum((2 * alpha / pw) * term, 0.0)

    n_hours = len(test)
    n_fine = n_hours * 10
    u_last = np.clip(train.iloc[-1], 1e-6, None)
    X0 = sp_norm.ppf(np.clip(weibull_min.cdf(u_last, k, scale=lam), 1e-9, 1 - 1e-9))

    U1_all = np.zeros((n_sim, n_hours))
    U2_all = np.zeros((n_sim, n_hours))
    U3_all = np.zeros((n_sim, n_hours))

    print(f"Simulating {n_sim} trajectories x {n_hours} hours for each model...")
    for s in range(n_sim):
        np.random.seed(s)
        dW = np.random.normal(0, np.sqrt(0.1), n_fine)

        X = np.zeros(n_fine); X[0] = X0
        for i in range(1, n_fine):
            X[i] = X[i - 1] - alpha * X[i - 1] * 0.1 + sigma_ou * dW[i]
        U1_all[s] = weibull_min.ppf(np.clip(sp_norm.cdf(X), 1e-9, 1 - 1e-9), k, scale=lam)[::10][:n_hours]

        U2 = np.zeros(n_fine); U2[0] = u_last
        for i in range(1, n_fine):
            uv = max(U2[i - 1], 0.001)
            a_ = -alpha * (uv - mu_W)
            b_ = np.sqrt(b2_model2(uv))
            U2[i] = max(uv + a_ * 0.1 + b_ * dW[i], 0.001)
        U2_all[s] = U2[::10][:n_hours]

        U3 = np.zeros(n_fine); U3[0] = u_last
        for i in range(1, n_fine):
            uv = max(U3[i - 1], 0.001)
            a_ = alpha * sigma_W ** 2 * (k / uv) * ((k - 1) / k - (uv / lam) ** k)
            step = uv + a_ * 0.1 + b3 * dW[i]
            U3[i] = abs(step) if step < 0.01 else step
        U3_all[s] = U3[::10][:n_hours]

    sims = {"Model I": U1_all, "Model II": U2_all, "Model III": U3_all}
    summary_rows = []
    for model_name, paths in sims.items():
        observed_test = test.values
        mc_mean = paths.mean(axis=0)
        rmse = np.sqrt(np.mean((observed_test - mc_mean) ** 2))
        lower = np.percentile(paths, 2.5, axis=0)
        upper = np.percentile(paths, 97.5, axis=0)
        coverage = ((observed_test >= lower) & (observed_test <= upper)).mean()
        print(f"{model_name}: RMSE={rmse:.4f}  Coverage={coverage:.1%}")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(train.index[-100:], train.values[-100:], color="gray", label="Train (last 100h)")
        ax.plot(test.index, test.values, color="black", label="Observed (test)")
        ax.plot(test.index, mc_mean, color="crimson", label="MC mean prediction")
        ax.fill_between(test.index, lower, upper, color="crimson", alpha=0.2, label="95% prediction interval")
        ax.set_title(f"{model_name}: RMSE={rmse:.3f}, Coverage={coverage:.1%}")
        ax.legend(); plt.tight_layout()
        plt.savefig(plot_path(f"task5_{model_name.replace(' ', '')}.png"), dpi=150)
        plt.close()

        summary_rows.append({"Model": model_name, "RMSE": rmse, "Coverage (95% CI)": coverage})

    summary = pd.DataFrame(summary_rows)
    print("\n=== Task 5 Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(data_path("task5_prediction_summary.csv"), index=False)
    return summary


# ----------------------------------------------------------------------
# TASK 6 — Information Criteria (Euler + Ozaki) + Confidence Intervals
# ----------------------------------------------------------------------
def _euler_neg_log_likelihood(alpha, x_prev, x_next, drift_fn, diffusion_fn, dt=1.0):
    f_vals = drift_fn(x_prev, alpha)
    g_vals = np.maximum(diffusion_fn(x_prev, alpha), 1e-6)
    mean_pred = x_prev + f_vals * dt
    var_pred = (g_vals ** 2) * dt
    residual = x_next - mean_pred
    log_density = -0.5 * (np.log(2 * np.pi * var_pred) + (residual ** 2) / var_pred)
    return -np.sum(log_density)


def _ozaki_neg_log_likelihood(alpha, x_prev, x_next, drift_fn, drift_deriv_fn, diffusion_fn, dt=1.0):
    f_vals = drift_fn(x_prev, alpha)
    L_vals = drift_deriv_fn(x_prev, alpha)
    g_vals = np.maximum(diffusion_fn(x_prev, alpha), 1e-6)
    L_safe = np.where(np.abs(L_vals) < 1e-8, 1e-8, L_vals)
    mean_pred = x_prev + (f_vals / L_safe) * (np.exp(L_safe * dt) - 1)
    var_pred = np.maximum(np.abs((g_vals ** 2) * (np.exp(2 * L_safe * dt) - 1) / (2 * L_safe)), 1e-8)
    residual = x_next - mean_pred
    log_density = -0.5 * (np.log(2 * np.pi * var_pred) + (residual ** 2) / var_pred)
    return -np.sum(log_density)


def _fit_alpha(neg_log_lik_fn, args, bounds=(1e-5, 2.0)):
    result = minimize_scalar(neg_log_lik_fn, args=args, bounds=bounds, method="bounded")
    return result.x, -result.fun


def _compute_aic_bic(log_lik, n_params, n_obs):
    return 2 * n_params - 2 * log_lik, n_params * np.log(n_obs) - 2 * log_lik


def _task6_model_setup(data, k, lam, mu_W, sigma_W, min_u_threshold=0.1):
    u_clipped = np.clip(data, 1e-6, None)
    X_transformed = sp_norm.ppf(np.clip(weibull_min.cdf(u_clipped, k, scale=lam), 1e-9, 1 - 1e-9))
    x_prev1, x_next1 = X_transformed[:-1], X_transformed[1:]

    def drift1(x, a): return -a * x
    def drift1_deriv(x, a): return -a * np.ones_like(x)
    def diff1(x, a): return np.sqrt(2 * a) * np.ones_like(x)

    x_prev_full, x_next_full = data[:-1], data[1:]
    valid_mask = x_prev_full >= min_u_threshold
    n_excluded = (~valid_mask).sum()
    x_prev2, x_next2 = x_prev_full[valid_mask], x_next_full[valid_mask]

    def drift2(x, a): return -a * (x - mu_W)
    def drift2_deriv(x, a): return -a * np.ones_like(x)
    def diff2(x, a):
        eps = 1e-6
        xv = np.maximum(x, eps)
        pw = np.maximum(weibull_min.pdf(xv, k, scale=lam), 1e-4)
        a_val = 1 + 1 / k
        z_val = (xv / lam) ** k
        inc_gam = Gamma(a_val) * (1 - gammainc(a_val, z_val))
        term = lam * inc_gam - mu_W * np.exp(-z_val)
        return np.sqrt(np.clip((2 * a / pw) * term, 1e-6, 100))

    def drift3(x, a):
        xv = np.maximum(x, 0.001)
        return a * sigma_W ** 2 * (k / xv) * ((k - 1) / k - (xv / lam) ** k)

    def drift3_deriv(x, a):
        xv = np.maximum(x, 0.001)
        h = 1e-4
        f_plus = a * sigma_W ** 2 * (k / (xv + h)) * ((k - 1) / k - ((xv + h) / lam) ** k)
        f_minus = a * sigma_W ** 2 * (k / (xv - h)) * ((k - 1) / k - ((xv - h) / lam) ** k)
        return (f_plus - f_minus) / (2 * h)

    def diff3(x, a): return np.sqrt(2 * a) * sigma_W * np.ones_like(x)

    return {
        "Model I": {"euler_args": (x_prev1, x_next1, drift1, diff1),
                    "ozaki_args": (x_prev1, x_next1, drift1, drift1_deriv, diff1),
                    "n_obs": len(x_prev1)},
        "Model II": {"euler_args": (x_prev2, x_next2, drift2, diff2),
                     "ozaki_args": (x_prev2, x_next2, drift2, drift2_deriv, diff2),
                     "n_obs": len(x_prev2)},
        "Model III": {"euler_args": (x_prev_full, x_next_full, drift3, diff3),
                      "ozaki_args": (x_prev_full, x_next_full, drift3, drift3_deriv, diff3),
                      "n_obs": len(x_prev_full)},
    }, n_excluded


def task6_information_criteria():
    print("\n" + "=" * 70)
    print("TASK 6: Information Criteria + Euler vs Ozaki + Confidence Intervals")
    print("=" * 70)
    df = load_raw_reading_data()
    series = df[["U10"]].resample("1h").mean().dropna()["U10"]
    data = series.values

    u_pos = data[data > 0]
    k, loc, lam = weibull_min.fit(u_pos, floc=0)
    mu_W = lam * Gamma(1 + 1 / k)
    sigma_W = np.sqrt(lam ** 2 * Gamma(1 + 2 / k) - mu_W ** 2)
    print(f"Weibull: k={k:.4f}, lambda={lam:.4f}")

    setups, n_excluded = _task6_model_setup(data, k, lam, mu_W, sigma_W)

    results, results_ext, results_ci = [], [], []
    for label, setup in setups.items():
        print(f"\nFitting {label}...")
        alpha_e, logL_e = _fit_alpha(_euler_neg_log_likelihood, setup["euler_args"])
        aic_e, bic_e = _compute_aic_bic(logL_e, 1, setup["n_obs"])
        print(f"  Euler: alpha={alpha_e:.6f}, logLik={logL_e:.2f}, AIC={aic_e:.2f}, BIC={bic_e:.2f}")
        results.append({"Model": label, "alpha": alpha_e, "LogLikelihood": logL_e, "AIC": aic_e, "BIC": bic_e, "n_obs": setup["n_obs"]})
        results_ext.append({"Model": label, "Method": "Euler", "alpha": alpha_e, "LogLikelihood": logL_e, "AIC": aic_e, "BIC": bic_e})

        alpha_o, logL_o = _fit_alpha(_ozaki_neg_log_likelihood, setup["ozaki_args"])
        aic_o, bic_o = _compute_aic_bic(logL_o, 1, setup["n_obs"])
        print(f"  Ozaki: alpha={alpha_o:.6f}, logLik={logL_o:.2f}, AIC={aic_o:.2f}, BIC={bic_o:.2f}")
        results_ext.append({"Model": label, "Method": "Ozaki", "alpha": alpha_o, "LogLikelihood": logL_o, "AIC": aic_o, "BIC": bic_o})

        # confidence interval via numerical second derivative (Euler)
        xp, xn, df_fn, dif_fn = setup["euler_args"]
        H = 1e-5
        f0 = _euler_neg_log_likelihood(alpha_e, xp, xn, df_fn, dif_fn)
        f_plus = _euler_neg_log_likelihood(alpha_e + H, xp, xn, df_fn, dif_fn)
        f_minus = _euler_neg_log_likelihood(alpha_e - H, xp, xn, df_fn, dif_fn)
        second_deriv = (f_plus - 2 * f0 + f_minus) / H ** 2
        if second_deriv > 0:
            se = np.sqrt(1 / second_deriv)
            ci_lower, ci_upper = alpha_e - 1.96 * se, alpha_e + 1.96 * se
            print(f"  95% CI = [{ci_lower:.6f}, {ci_upper:.6f}]")
        else:
            se = ci_lower = ci_upper = None
        results_ci.append({"Model": label, "alpha": alpha_e, "SE": se, "CI_lower_95": ci_lower, "CI_upper_95": ci_upper})

    summary = pd.DataFrame(results)
    print("\n=== Information Criteria (Euler) ===")
    print(summary.to_string(index=False))
    print(f"\nBest model by AIC: {summary.loc[summary['AIC'].idxmin(), 'Model']}")
    print(f"Note: Model II excluded {n_excluded} near-zero transitions.")
    summary.to_csv(data_path("task6_information_criteria.csv"), index=False)

    summary_ext = pd.DataFrame(results_ext)
    print("\n=== Euler vs Ozaki AIC Comparison ===")
    print(summary_ext.pivot(index="Model", columns="Method", values="AIC").to_string())
    summary_ext.to_csv(data_path("task6_euler_vs_ozaki.csv"), index=False)

    summary_ci = pd.DataFrame(results_ci)
    print("\n=== 95% Confidence Intervals for alpha ===")
    print(summary_ci.to_string(index=False))
    summary_ci.to_csv(data_path("task6_confidence_intervals.csv"), index=False)

    return summary, summary_ext, summary_ci


# ----------------------------------------------------------------------
# TASK 7 — Computational Efficiency
# ----------------------------------------------------------------------
def task7_computational_efficiency():
    print("\n" + "=" * 70)
    print("TASK 7: Computational Efficiency")
    print("=" * 70)
    df = load_raw_reading_data()
    series = df[["U10"]].resample("1h").mean().dropna()["U10"]
    data = series.values

    u_pos = data[data > 0]
    k, loc, lam = weibull_min.fit(u_pos, floc=0)
    mu_W = lam * Gamma(1 + 1 / k)
    sigma_W = np.sqrt(lam ** 2 * Gamma(1 + 2 / k) - mu_W ** 2)

    setups, n_excluded = _task6_model_setup(data, k, lam, mu_W, sigma_W)

    def time_fit(euler_args, n_repeats=5):
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            minimize_scalar(_euler_neg_log_likelihood, args=euler_args, bounds=(1e-5, 2.0), method="bounded")
            times.append(time.perf_counter() - t0)
        return np.mean(times), np.std(times)

    print("\nTiming parameter estimation (averaged over 5 runs)...")
    timing = {}
    for label, setup in setups.items():
        t_mean, t_std = time_fit(setup["euler_args"])
        timing[label] = t_mean
        print(f"  {label}: {t_mean*1000:.2f} +/- {t_std*1000:.2f} ms")

    alpha_est = 0.032692
    n_sim_timing, n_hours_timing = 50, 8760
    dt_fine, step_per_hour = 0.1, 10
    n_fine = n_hours_timing * step_per_hour

    def sim1():
        sigma_ou = np.sqrt(2 * alpha_est)
        neg = 0
        for s in range(n_sim_timing):
            np.random.seed(s)
            dW = np.random.normal(0, np.sqrt(dt_fine), n_fine)
            X = np.zeros(n_fine)
            for i in range(1, n_fine):
                X[i] = X[i - 1] - alpha_est * X[i - 1] * dt_fine + sigma_ou * dW[i]
        return neg

    def sim2():
        u0 = lam * (np.log(2)) ** (1 / k)
        neg = 0

        def b2m(uv):
            eps = 1e-9
            uv = np.maximum(uv, eps)
            pw = np.maximum(weibull_min.pdf(uv, k, scale=lam), eps)
            a_val = 1 + 1 / k
            z_val = (uv / lam) ** k
            inc_gam = Gamma(a_val) * (1 - gammainc(a_val, z_val))
            term = lam * inc_gam - mu_W * np.exp(-z_val)
            return np.maximum((2 * alpha_est / pw) * term, 0.0)

        for s in range(n_sim_timing):
            np.random.seed(s)
            dW = np.random.normal(0, np.sqrt(dt_fine), n_fine)
            U2 = np.zeros(n_fine); U2[0] = u0
            for i in range(1, n_fine):
                uv = max(U2[i - 1], 0.001)
                a_ = -alpha_est * (uv - mu_W)
                b_ = np.sqrt(b2m(uv))
                step = uv + a_ * dt_fine + b_ * dW[i]
                if step < 0:
                    neg += 1
                U2[i] = max(step, 0.001)
        return neg

    def sim3():
        u0 = lam * (np.log(2)) ** (1 / k)
        b3 = np.sqrt(2 * alpha_est) * sigma_W
        neg = 0
        for s in range(n_sim_timing):
            np.random.seed(s)
            dW = np.random.normal(0, np.sqrt(dt_fine), n_fine)
            U3 = np.zeros(n_fine); U3[0] = u0
            for i in range(1, n_fine):
                uv = max(U3[i - 1], 0.001)
                a_ = alpha_est * sigma_W ** 2 * (k / uv) * ((k - 1) / k - (uv / lam) ** k)
                step = uv + a_ * dt_fine + b3 * dW[i]
                if step < 0.01:
                    neg += 1
                U3[i] = abs(step) if step < 0.01 else step
        return neg

    print(f"\nTiming simulation ({n_sim_timing} trajectories x {n_hours_timing} hours)...")
    t0 = time.perf_counter(); neg1 = sim1(); sim_time1 = time.perf_counter() - t0
    print(f"  Model I: {sim_time1:.2f}s total, {sim_time1/n_sim_timing*1000:.2f} ms/traj, {neg1} negative events")
    t0 = time.perf_counter(); neg2 = sim2(); sim_time2 = time.perf_counter() - t0
    print(f"  Model II: {sim_time2:.2f}s total, {sim_time2/n_sim_timing*1000:.2f} ms/traj, {neg2} negative events")
    t0 = time.perf_counter(); neg3 = sim3(); sim_time3 = time.perf_counter() - t0
    print(f"  Model III: {sim_time3:.2f}s total, {sim_time3/n_sim_timing*1000:.2f} ms/traj, {neg3} negative events")

    total_steps = n_sim_timing * n_hours_timing * step_per_hour
    summary = pd.DataFrame([
        {"Model": "Model I", "ParamEstTime_ms": timing["Model I"] * 1000, "SimTime_per_traj_ms": sim_time1 / n_sim_timing * 1000, "NegativeEvents": neg1, "NegativeEventRate_pct": 100 * neg1 / total_steps},
        {"Model": "Model II", "ParamEstTime_ms": timing["Model II"] * 1000, "SimTime_per_traj_ms": sim_time2 / n_sim_timing * 1000, "NegativeEvents": neg2, "NegativeEventRate_pct": 100 * neg2 / total_steps},
        {"Model": "Model III", "ParamEstTime_ms": timing["Model III"] * 1000, "SimTime_per_traj_ms": sim_time3 / n_sim_timing * 1000, "NegativeEvents": neg3, "NegativeEventRate_pct": 100 * neg3 / total_steps},
    ])
    print("\n=== Task 7 Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(data_path("task7_computational_efficiency.csv"), index=False)
    return summary


# ----------------------------------------------------------------------
# TASK 8 — Quantitative Comparison Metrics / Skill Score
# ----------------------------------------------------------------------
def task8_quantitative_comparison():
    print("\n" + "=" * 70)
    print("TASK 8: Quantitative Comparison Metrics / Skill Score")
    print("=" * 70)
    cached = _load_cached_section3()
    k, lam = cached["k"], cached["lam"]
    observed = cached["u_observed"]
    models = {"Model I": cached["U1_all"], "Model II": cached["U2_all"], "Model III": cached["U3_all"]}
    max_lag = 120
    quantiles = [0.90, 0.95, 0.99]

    def dist_err(pooled):
        sample = pooled[pooled > 0]
        if len(sample) > 20000:
            rng = np.random.default_rng(42)
            sample = rng.choice(sample, size=20000, replace=False)
        D, _ = kstest(sample, "weibull_min", args=(k, 0, lam))
        return D

    def acf_err(all_sims, n_use=200):
        observed_acf = acf(observed, nlags=max_lag, fft=True)
        n_use = min(n_use, all_sims.shape[0])
        acf_matrix = np.array([acf(all_sims[i], nlags=max_lag, fft=True) for i in range(n_use)])
        return np.mean(np.abs(observed_acf - acf_matrix.mean(axis=0)))

    rows = []
    for label, all_sims in models.items():
        print(f"  Processing {label}...")
        pooled = all_sims.flatten()
        d_err = dist_err(pooled)
        a_err = acf_err(all_sims)
        obs_mean, obs_var = observed.mean(), observed.var()
        mean_err = abs(obs_mean - all_sims.mean(axis=1).mean())
        var_err = abs(obs_var - all_sims.var(axis=1).mean())
        extreme_err = abs(observed.max() - all_sims.max(axis=1).mean())
        row = {"Model": label, "Distribution_Error_KS": d_err, "ACF_Error_MAE": a_err,
               "Mean_Error": mean_err, "Variance_Error": var_err, "Extreme_Value_Error": extreme_err}
        for q in quantiles:
            obs_q = np.quantile(observed, q)
            sim_q = np.quantile(pooled[pooled > 0], q)
            row[f"q{int(q*100)}_error"] = abs(obs_q - sim_q)
        rows.append(row)

    summary = pd.DataFrame(rows)
    col_order = ["Model", "Distribution_Error_KS", "ACF_Error_MAE", "Mean_Error", "Variance_Error",
                 "q90_error", "q95_error", "q99_error", "Extreme_Value_Error"]
    summary = summary[col_order]
    print("\n=== Task 8 Error Measures ===")
    print(summary.to_string(index=False))

    metric_cols = [c for c in col_order if c != "Model"]
    reference_row = summary[summary["Model"] == "Model I"][metric_cols].values[0]
    skill_scores = 1 - summary[metric_cols].div(reference_row, axis=1)
    summary["Mean_Skill_Score"] = skill_scores.mean(axis=1)

    print("\n=== Overall Ranking: Mean Skill Score (higher = better) ===")
    ranking = summary[["Model", "Mean_Skill_Score"]].sort_values("Mean_Skill_Score", ascending=False)
    print(ranking.to_string(index=False))

    summary.to_csv(data_path("task8_quantitative_comparison.csv"), index=False)
    return summary


########################################################################
# PART 3: CHAPTER 7 -- HYBRID SDE+LSTM, CHRONOS, REJECTION SAMPLING,
# TURBINE EXCEEDANCE PROBABILITIES
#
# These are more expensive (LSTM training, or in Chronos's case a large
# pretrained model download) and are kept separate from the default run
# order -- call run_chapter7_extras() explicitly, or run individual
# functions.
########################################################################

def _load_hour_of_day():
    df = load_raw_reading_data()
    df_hourly = df[["U10"]].resample("1h").mean().dropna()
    return df_hourly.index.hour.values


def _to_ou_space(u, k, lam):
    F = np.clip(weibull_min.cdf(u, k, scale=lam), 1e-9, 1 - 1e-9)
    return sp_norm.ppf(F)


def _from_ou_space(X, k, lam):
    F = np.clip(sp_norm.cdf(X), 1e-9, 1 - 1e-9)
    return weibull_min.ppf(F, k, scale=lam)


def hybrid_sde_lstm(lookback=24, train_frac=0.75):
    """Deterministic hybrid: LSTM predicts the OU innovation epsilon(t)."""
    print("\n" + "=" * 70)
    print("HYBRID SDE + LSTM MODEL (deterministic residual correction)")
    print("=" * 70)
    from tensorflow.keras import layers, models as tfmodels, callbacks

    np.random.seed(42)
    tf.random.set_seed(42)

    cached = _load_cached_section3()
    k, lam, alpha = cached["k"], cached["lam"], cached["alpha"]
    u_observed = cached["u_observed"]
    hours = _load_hour_of_day()
    assert len(hours) == len(u_observed), "Length mismatch between hours and observed series."

    X = _to_ou_space(u_observed, k, lam)
    phi = np.exp(-alpha * 1.0)
    eps = X[1:] - phi * X[:-1]
    print(f"phi = exp(-alpha) = {phi:.5f}")

    def make_windows(eps_, hours_, lb):
        hours_aligned = hours_[1:]
        sin_h = np.sin(2 * np.pi * hours_aligned / 24)
        cos_h = np.cos(2 * np.pi * hours_aligned / 24)
        Xf, y = [], []
        for t in range(lb, len(eps_)):
            feat = np.stack([eps_[t - lb:t], sin_h[t - lb:t], cos_h[t - lb:t]], axis=-1)
            Xf.append(feat); y.append(eps_[t])
        return np.array(Xf), np.array(y)

    def make_windows_lstm_only(u_, hours_, lb):
        sin_h = np.sin(2 * np.pi * hours_ / 24)
        cos_h = np.cos(2 * np.pi * hours_ / 24)
        Xf, y = [], []
        for t in range(lb, len(u_)):
            feat = np.stack([u_[t - lb:t], sin_h[t - lb:t], cos_h[t - lb:t]], axis=-1)
            Xf.append(feat); y.append(u_[t])
        return np.array(Xf), np.array(y)

    Xf_hybrid, y_hybrid = make_windows(eps, hours, lookback)
    Xf_lstm, y_lstm = make_windows_lstm_only(u_observed, hours, lookback)
    n = min(len(Xf_hybrid), len(Xf_lstm))
    Xf_hybrid, y_hybrid = Xf_hybrid[-n:], y_hybrid[-n:]
    Xf_lstm, y_lstm = Xf_lstm[-n:], y_lstm[-n:]

    split = int(n * train_frac)
    print(f"Train/test split: {split} train, {n - split} test")

    def build_lstm(input_shape):
        m = tfmodels.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(50, activation="tanh"),
            layers.Dense(20, activation="relu"),
            layers.Dense(1),
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    es = callbacks.EarlyStopping(patience=8, restore_best_weights=True)

    print("\nTraining hybrid (residual) LSTM...")
    model_hybrid = build_lstm((lookback, 3))
    model_hybrid.fit(Xf_hybrid[:split], y_hybrid[:split], validation_split=0.1,
                      epochs=100, batch_size=32, callbacks=[es], verbose=0)
    eps_pred_test = model_hybrid.predict(Xf_hybrid[split:], verbose=0).flatten()

    print("Training pure LSTM baseline...")
    model_lstm = build_lstm((lookback, 3))
    model_lstm.fit(Xf_lstm[:split], y_lstm[:split], validation_split=0.1,
                    epochs=100, batch_size=32, callbacks=[es], verbose=0)
    u_pred_lstm_only = model_lstm.predict(Xf_lstm[split:], verbose=0).flatten()

    test_start_in_eps = split + lookback
    X_prev_test = X[test_start_in_eps: test_start_in_eps + len(eps_pred_test)]
    u_true_test = u_observed[test_start_in_eps + 1: test_start_in_eps + 1 + len(eps_pred_test)]

    u_pred_sde_only = _from_ou_space(phi * X_prev_test, k, lam)
    u_pred_hybrid = _from_ou_space(phi * X_prev_test + eps_pred_test, k, lam)

    m = min(len(u_true_test), len(u_pred_lstm_only))
    u_true_test = u_true_test[:m]
    u_pred_sde_only = u_pred_sde_only[:m]
    u_pred_hybrid = u_pred_hybrid[:m]
    u_pred_lstm_only = u_pred_lstm_only[:m]

    def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))
    rmse_sde = rmse(u_true_test, u_pred_sde_only)
    rmse_lstm = rmse(u_true_test, u_pred_lstm_only)
    rmse_hybrid = rmse(u_true_test, u_pred_hybrid)

    print("\n" + "=" * 50)
    print("ONE-STEP-AHEAD TEST-SET RMSE (m/s)")
    print("=" * 50)
    print(f"SDE-only  : {rmse_sde:.4f}")
    print(f"LSTM-only : {rmse_lstm:.4f}")
    print(f"Hybrid    : {rmse_hybrid:.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_n = min(300, m)
    ax.plot(u_true_test[:plot_n], color="black", linewidth=1.2, label="Observed")
    ax.plot(u_pred_sde_only[:plot_n], color="steelblue", linewidth=1, alpha=0.7, label="SDE-only")
    ax.plot(u_pred_hybrid[:plot_n], color="crimson", linewidth=1, alpha=0.8, label="Hybrid")
    ax.set_xlabel("Test hour"); ax.set_ylabel("Wind speed [m/s]")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_path("hybrid_forecast_comparison.png"), dpi=150)
    plt.close()

    np.savez(data_path("hybrid_outputs.npz"),
             u_true_test=u_true_test, u_pred_sde_only=u_pred_sde_only,
             u_pred_hybrid=u_pred_hybrid, u_pred_lstm_only=u_pred_lstm_only,
             rmse_sde=rmse_sde, rmse_lstm=rmse_lstm, rmse_hybrid=rmse_hybrid)
    print(f"Saved results to {data_path('hybrid_outputs.npz')}")
    return {"rmse_sde": rmse_sde, "rmse_lstm": rmse_lstm, "rmse_hybrid": rmse_hybrid}


def hybrid_distribution_validation():
    print("\n" + "=" * 70)
    print("HYBRID DISTRIBUTIONAL + ACF VALIDATION")
    print("=" * 70)
    hyb_path = data_path("hybrid_outputs.npz")
    if not os.path.exists(hyb_path):
        raise FileNotFoundError(f"{hyb_path} not found -- run hybrid_sde_lstm() first.")
    hyb = np.load(hyb_path)
    cached = _load_cached_section3()
    k, lam, alpha = cached["k"], cached["lam"], cached["alpha"]

    u_true = hyb["u_true_test"]
    u_sde = hyb["u_pred_sde_only"]
    u_lstm = hyb["u_pred_lstm_only"]
    u_hybrid = hyb["u_pred_hybrid"]

    results_gof = {}
    print(f"\nGoodness of fit vs target Weibull (k={k:.4f}, lambda={lam:.4f}):")
    for name, series in [("Observed(test)", u_true), ("SDE-only", u_sde), ("LSTM-only", u_lstm), ("Hybrid", u_hybrid)]:
        series_pos = series[series > 0]
        D, p = kstest(series_pos, "weibull_min", args=(k, 0, lam))
        results_gof[name] = (D, p)
        print(f"  {name}: D={D:.4f}, p={p:.4f}")

    max_lag = min(120, len(u_true) - 1)
    lags = np.arange(0, max_lag + 1)
    acf_true = acf(u_true, nlags=max_lag, fft=True)
    acf_sde = acf(u_sde, nlags=max_lag, fft=True)
    acf_lstm = acf(u_lstm, nlags=max_lag, fft=True)
    acf_hybrid = acf(u_hybrid, nlags=max_lag, fft=True)

    mae_sde = np.mean(np.abs(acf_sde - acf_true))
    mae_lstm = np.mean(np.abs(acf_lstm - acf_true))
    mae_hybrid = np.mean(np.abs(acf_hybrid - acf_true))
    print(f"\nACF MAE: SDE-only={mae_sde:.4f}, LSTM-only={mae_lstm:.4f}, Hybrid={mae_hybrid:.4f}")

    np.savez(data_path("hybrid_distribution_validation.npz"),
             ks_sde=results_gof["SDE-only"], ks_lstm=results_gof["LSTM-only"], ks_hybrid=results_gof["Hybrid"],
             acf_mae_sde=mae_sde, acf_mae_lstm=mae_lstm, acf_mae_hybrid=mae_hybrid)
    return results_gof, {"sde": mae_sde, "lstm": mae_lstm, "hybrid": mae_hybrid}


def diebold_mariano_test(e1, e2, h=1, power=2):
    e1, e2 = np.asarray(e1), np.asarray(e2)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    d_bar = np.mean(d)
    max_lag = max(1, h - 1)
    var_d = np.var(d, ddof=0)
    for lag in range(1, max_lag + 1):
        var_d += 2 * np.cov(d[lag:], d[:-lag])[0, 1]
    var_d = max(var_d, 1e-12)
    dm_stat = d_bar / np.sqrt(var_d / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_adj = dm_stat * hln
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_adj), df=n - 1))
    return dm_stat_adj, p_value


def run_dm_test():
    print("\n" + "=" * 70)
    print("DIEBOLD-MARIANO SIGNIFICANCE TEST")
    print("=" * 70)
    hyb_path = data_path("hybrid_outputs.npz")
    if not os.path.exists(hyb_path):
        raise FileNotFoundError(f"{hyb_path} not found -- run hybrid_sde_lstm() first.")
    data = np.load(hyb_path)
    u_true, u_sde, u_lstm, u_hybrid = data["u_true_test"], data["u_pred_sde_only"], data["u_pred_lstm_only"], data["u_pred_hybrid"]
    e_sde, e_lstm, e_hybrid = u_true - u_sde, u_true - u_lstm, u_true - u_hybrid

    dm1, p1 = diebold_mariano_test(e_hybrid, e_sde)
    dm2, p2 = diebold_mariano_test(e_hybrid, e_lstm)
    dm3, p3 = diebold_mariano_test(e_lstm, e_sde)
    print(f"Hybrid vs SDE-only:  DM={dm1:.4f}, p={p1:.4f}")
    print(f"Hybrid vs LSTM-only: DM={dm2:.4f}, p={p2:.4f}")
    print(f"LSTM-only vs SDE:    DM={dm3:.4f}, p={p3:.4f}")

    np.savez(data_path("dm_test_results.npz"),
             dm_hybrid_vs_sde=dm1, p_hybrid_vs_sde=p1,
             dm_hybrid_vs_lstm=dm2, p_hybrid_vs_lstm=p2,
             dm_lstm_vs_sde=dm3, p_lstm_vs_sde=p3)
    return {"hybrid_vs_sde": (dm1, p1), "hybrid_vs_lstm": (dm2, p2), "lstm_vs_sde": (dm3, p3)}


def model3_rejection_sampling(n_sim_target=1000, max_attempts=3000):
    """Tests whether full-trajectory rejection sampling (the original paper's
    boundary approach) is viable for Model III on this dataset's parameters."""
    print("\n" + "=" * 70)
    print("MODEL III: REJECTION SAMPLING INVESTIGATION")
    print("=" * 70)
    cached = _load_cached_section3()
    k, lam, alpha, sigma_W = cached["k"], cached["lam"], cached["alpha"], cached["sigma_W"]
    U3_all_old, u_observed = cached["U3_all"], cached["u_observed"]

    dt_fine, step_per_hour, N_fine = 0.1, 10, 8760 * 10
    u0 = lam * (np.log(2)) ** (1 / k)
    b3 = np.sqrt(2 * alpha) * sigma_W

    def simulate_trajectory(seed):
        rng = np.random.default_rng(seed)
        dW = rng.normal(0, np.sqrt(dt_fine), N_fine)
        U3 = np.zeros(N_fine); U3[0] = u0
        went_negative = False
        for i in range(1, N_fine):
            uv = max(U3[i - 1], 0.001)
            a_ = alpha * sigma_W ** 2 * (k / uv) * ((k - 1) / k - (uv / lam) ** k)
            step = uv + a_ * dt_fine + b3 * dW[i]
            if step < 0:
                went_negative = True
            U3[i] = abs(step) if step < 0.01 else step
        return U3[::step_per_hour], went_negative

    print(f"Target: {n_sim_target} valid trajectories, max {max_attempts} attempts")
    valid_trajectories, total_attempts, rejected, seed = [], 0, 0, 0
    while len(valid_trajectories) < n_sim_target and total_attempts < max_attempts:
        traj, went_negative = simulate_trajectory(seed)
        total_attempts += 1
        seed += 1
        if went_negative:
            rejected += 1
            continue
        valid_trajectories.append(traj)

    rejection_rate = rejected / total_attempts * 100 if total_attempts else 0.0
    print(f"\n{total_attempts} attempts, {rejected} rejected. Rejection rate: {rejection_rate:.2f}% (paper reports 9.57%)")

    if len(valid_trajectories) > 0:
        U3_all_new = np.array(valid_trajectories)
        old_flat_pos = U3_all_old.flatten()
        old_flat_pos = old_flat_pos[old_flat_pos > 0]
        new_flat_pos = U3_all_new.flatten()
        new_flat_pos = new_flat_pos[new_flat_pos > 0]
        D_old, p_old = kstest(old_flat_pos, "weibull_min", args=(k, 0, lam))
        D_new, p_new = kstest(new_flat_pos, "weibull_min", args=(k, 0, lam))
        print(f"Reflection (old): D={D_old:.4f}, p={p_old:.4f}")
        print(f"Rejection (new):  D={D_new:.4f}, p={p_new:.4f}")
    else:
        D_old = p_old = D_new = p_new = None
        print("No valid trajectories found within max_attempts (expected if the "
              "expected zero-crossing rate per year is high).")

    np.savez(data_path("model3_rejection_sampling_results.npz"),
             total_attempts=total_attempts, rejected=rejected, rejection_rate=rejection_rate,
             D_old=D_old, p_old=p_old, D_new=D_new, p_new=p_new)
    return {"total_attempts": total_attempts, "rejected": rejected, "rejection_rate": rejection_rate}


def exceedance_probabilities():
    print("\n" + "=" * 70)
    print("TURBINE-RELEVANT EXCEEDANCE PROBABILITIES")
    print("=" * 70)
    cached = _load_cached_section3()
    k, lam = cached["k"], cached["lam"]
    u_obs = cached["u_observed"]
    U1_all = cached["U1_all"]

    thresholds = np.array([2.0, 3.0, 5.0, 6.0, 7.0, 8.0])
    labels = ["2.0 m/s (light breeze)", "3.0 m/s (turbine cut-in)", "5.0 m/s (moderate)",
              "6.0 m/s (fresh breeze)", "7.0 m/s (strong breeze)", "8.0 m/s (near gale)"]

    print(f"{'Threshold':<28}{'Observed':>12}{'Simulated':>15}{'Weibull MLE':>15}")
    rows = []
    for thresh, label in zip(thresholds, labels):
        p_obs = np.mean(u_obs > thresh)
        p_sim = np.mean(U1_all > thresh)
        p_weib = 1 - weibull_min.cdf(thresh, k, scale=lam)
        print(f"{label:<28}{p_obs:>12.4f}{p_sim:>15.4f}{p_weib:>15.4f}")
        rows.append({"threshold": label, "observed": p_obs, "simulated": p_sim, "weibull_mle": p_weib})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(data_path("exceedance_probabilities.csv"), index=False)
    return df_out


def run_chapter7_extras():
    """Runs the (slower) Chapter 7 original-contribution work: hybrid model,
    distributional validation, DM significance test, Model III rejection
    sampling investigation, and turbine exceedance probabilities.
    Chronos is intentionally excluded here since it requires an additional
    `pip install chronos-forecasting` and downloads a pretrained model from
    the internet; call chronos_benchmark() directly if you want to run it.
    """
    print("\n" + "#" * 70)
    print("# CHAPTER 7: HYBRID SDE-LSTM MODEL AND EXTENSIONS")
    print("#" * 70)
    hybrid_sde_lstm()
    hybrid_distribution_validation()
    run_dm_test()
    model3_rejection_sampling()
    exceedance_probabilities()


def chronos_benchmark(context_len=168, train_frac=0.75, batch_size=50, num_samples=20):
    """Zero-shot benchmark against Amazon's Chronos time series foundation
    model. REQUIRES: pip install chronos-forecasting torch
    This downloads a pretrained model from Hugging Face on first run.
    """
    print("\n" + "=" * 70)
    print("ZERO-SHOT FOUNDATION MODEL BENCHMARK (Chronos)")
    print("=" * 70)
    try:
        from chronos import ChronosPipeline
    except ImportError:
        raise ImportError(
            "chronos-forecasting is not installed. Run: "
            "pip install chronos-forecasting torch --break-system-packages"
        )

    cached = _load_cached_section3()
    k, lam, alpha, u_observed = cached["k"], cached["lam"], cached["alpha"], cached["u_observed"]
    n = len(u_observed)
    split = int(n * train_frac)
    test_indices = list(range(max(split, context_len), n))
    print(f"Test set: {len(test_indices)} one-step-ahead forecasts (context={context_len}h)")

    print("Loading Chronos (amazon/chronos-t5-small)...")
    pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32)

    point_forecasts = np.zeros(len(test_indices))
    all_samples = np.zeros((len(test_indices), num_samples))

    for batch_start in range(0, len(test_indices), batch_size):
        batch_indices = test_indices[batch_start: batch_start + batch_size]
        contexts = [torch.tensor(u_observed[t - context_len:t], dtype=torch.float32) for t in batch_indices]
        forecast = pipeline.predict(contexts, prediction_length=1, num_samples=num_samples)
        forecast_np = forecast.numpy()[:, :, 0]
        for j, idx in enumerate(range(batch_start, batch_start + len(batch_indices))):
            all_samples[idx] = forecast_np[j]
            point_forecasts[idx] = np.median(forecast_np[j])
        print(f"  {batch_start + len(batch_indices)}/{len(test_indices)} done")

    u_true_test = np.array([u_observed[t] for t in test_indices])
    rmse_chronos = np.sqrt(np.mean((u_true_test - point_forecasts) ** 2))
    print(f"\nChronos RMSE: {rmse_chronos:.4f}")

    pooled_pos = all_samples.flatten()
    pooled_pos = pooled_pos[pooled_pos > 0]
    D, p = kstest(pooled_pos, "weibull_min", args=(k, 0, lam))
    print(f"KS D={D:.4f}, p={p:.4f}")

    max_lag = min(120, len(u_true_test) - 1)
    acf_true = acf(u_true_test, nlags=max_lag, fft=True)
    acf_chronos = acf(point_forecasts, nlags=max_lag, fft=True)
    acf_mae = np.mean(np.abs(acf_chronos - acf_true))
    print(f"ACF MAE: {acf_mae:.4f}")

    np.savez(data_path("chronos_benchmark_results.npz"),
             u_true_test=u_true_test, point_forecasts=point_forecasts,
             rmse_chronos=rmse_chronos, ks_stat=D, ks_p=p, acf_mae=acf_mae)
    return {"rmse": rmse_chronos, "ks_stat": D, "ks_p": p, "acf_mae": acf_mae}


########################################################################
# MAIN
########################################################################
def run_validation_tasks(sde_results):
    """Runs Chapter 6 Tasks 1-8 (requires section3 to have been run)."""
    print("\n" + "#" * 70)
    print("# CHAPTER 6: MODEL VALIDATION TASKS 1-8")
    print("#" * 70)
    task1_goodness_of_fit()
    task2_acf_distribution()
    task2_deseasonalize_fix(sde_results)
    task3_montecarlo_validation()
    task4_extreme_events()
    task5_prediction_accuracy()
    task6_information_criteria()
    task7_computational_efficiency()
    task8_quantitative_comparison()


def main(run_validation=True, run_chapter7=False):
    """
    Parameters
    ----------
    run_validation : bool
        If True (default), also runs the Chapter 6 validation Tasks 1-8.
        These reuse the cached simulation output from Section 3, so no
        extra SDE simulation is needed, but they do add some runtime
        (Task 2's deseasonalisation fix and Task 5 each simulate 500-1000
        more trajectories).
    run_chapter7 : bool
        If True, also runs the Chapter 7 original-contribution work
        (hybrid SDE+LSTM, DM test, Model III rejection sampling
        investigation, exceedance probabilities). This trains several
        LSTM models and is the slowest optional part of the pipeline.
        Chronos is NOT included here -- call chronos_benchmark()
        separately if you have chronos-forecasting installed.
    """
    if not os.path.exists(DATA_PATH_RAW):
        raise FileNotFoundError(
            f"Raw data file not found at {DATA_PATH_RAW}.\n"
            "Download the University of Reading 2023 dataset and place it there "
            "before running this script (see the header of this file for details)."
        )

    section1_hourly_aggregate()
    section2_hourly_plot()  # MUST run before Sections 4, 5, 8
    sde_results = section3_weibull_acf_sde_models()
    section4_stationarity_test()
    section5_seasonal_analysis()
    section6_higham_scripts()
    section7_robustness(sde_results)
    benchmark_results = section8_benchmarks()
    section9_comparison_table(sde_results, benchmark_results)

    if run_validation:
        run_validation_tasks(sde_results)

    if run_chapter7:
        run_chapter7_extras()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE. See ./plots/ and ./Dataset/ for all outputs.")
    print("=" * 70)


if __name__ == "__main__":
    main(run_validation=True, run_chapter7=False)
