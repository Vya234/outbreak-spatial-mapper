"""
Phase 2 - Step 2: The Forecasting Model
-----------------------------------------
Takes the daily centroid/spread-radius/case-count table built by
phase2_data_engineering.py and produces a forecast for a future day:

  1. TREND REGRESSION (geographic drift)
     Fits a simple polynomial (default: linear) to centroid_lat,
     centroid_lng and spread_radius_km against day. This is what
     "moves" the outbreak polygon on the map.

  2. SIR MODEL (case-count growth)
     Fits Susceptible-Infected-Recovered parameters (beta, gamma) to
     the observed cumulative case_count curve. This is what tells you
     HOW BAD it's getting, independent of geography.

  3. BLEND
     The SIR-predicted case count is used to scale the regression's
     radius prediction, so an accelerating outbreak (SIR says cases
     are about to spike) produces a proportionally larger predicted
     polygon than pure geometric drift would suggest on its own.

Why linear regression + SIR instead of a neural net: with ~14 days of
data from a single synthetic outbreak there are only ~7 usable
training windows (see Step 1) -- nowhere near enough for a model like
an LSTM to learn anything beyond noise. A 2-parameter SIR fit and a
2-parameter linear fit are appropriately sized to that data volume,
and both are interpretable (you can report R^2, residuals, beta/gamma)
in a way a black-box model isn't.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit

from data_engineering import compute_daily_features, load_cases


# ---------------------------------------------------------------------
# 1. Geographic trend regression
# ---------------------------------------------------------------------

@dataclass
class TrendModel:
    """Holds fitted polynomial coefficients + residual std for one feature."""
    coeffs: np.ndarray
    degree: int
    residual_std: float

    def predict(self, day: float) -> float:
        return float(np.polyval(self.coeffs, day))


def fit_trend(daily_features: pd.DataFrame, column: str, degree: int = 1) -> TrendModel:
    """
    Fit a degree-N polynomial to `column` vs `day`.

    degree=1 (linear) is the sane default given how little data there
    is -- only bump this to 2 if you have enough days to justify it
    (rule of thumb: at least ~4x as many days as coefficients).
    """
    days = daily_features["day"].to_numpy(dtype=float)
    values = daily_features[column].to_numpy(dtype=float)

    coeffs = np.polyfit(days, values, deg=degree)
    predicted = np.polyval(coeffs, days)
    residual_std = float(np.std(values - predicted))

    return TrendModel(coeffs=coeffs, degree=degree, residual_std=residual_std)


def forecast_geography(daily_features: pd.DataFrame, target_day: int) -> dict:
    """Predict centroid_lat, centroid_lng, spread_radius_km for target_day."""
    lat_model = fit_trend(daily_features, "centroid_lat", degree=1)
    lng_model = fit_trend(daily_features, "centroid_lng", degree=1)
    radius_model = fit_trend(daily_features, "spread_radius_km", degree=1)

    return {
        "day": target_day,
        "centroid_lat": lat_model.predict(target_day),
        "centroid_lng": lng_model.predict(target_day),
        "spread_radius_km": max(0.0, radius_model.predict(target_day)),
        "radius_uncertainty_km": radius_model.residual_std,
    }


# ---------------------------------------------------------------------
# 2. SIR model fit
# ---------------------------------------------------------------------

def _sir_odes(y, t, beta, gamma, N):
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def _simulate_cumulative_infected(t, beta, gamma, N, I0):
    """Run the SIR ODEs and return cumulative infections (I + R) at each t."""
    S0 = N - I0
    R0 = 0.0
    sol = odeint(_sir_odes, [S0, I0, R0], t, args=(beta, gamma, N))
    I_curve, R_curve = sol[:, 1], sol[:, 2]
    return I_curve + R_curve


@dataclass
class SIRModel:
    beta: float
    gamma: float
    N: int
    I0: float
    t0: float  # day the I0/S0/R0 initial condition corresponds to

    def predict_cumulative_cases(self, days) -> np.ndarray:
        """
        Predict cumulative infections at each day in `days`.

        odeint needs the fit's own start day (t0) as the first point
        in the time array to know where the initial condition applies
        -- passing it a single unrelated future day (e.g. just [21])
        makes it return the initial condition unintegrated instead of
        actually running the model forward. So we always prepend t0,
        integrate, then drop that row before returning.
        """
        days = np.atleast_1d(np.asarray(days, dtype=float))
        t = np.concatenate(([self.t0], days)) if days[0] != self.t0 else days
        curve = _simulate_cumulative_infected(t, self.beta, self.gamma, self.N, self.I0)
        return curve[1:] if t[0] == self.t0 and len(t) > len(days) else curve


def fit_sir(daily_features: pd.DataFrame, population_estimate: int) -> SIRModel:
    """
    Fit beta (infection rate) and gamma (recovery rate) to the observed
    cumulative case_count curve via least squares.

    population_estimate (N): SIR needs a total susceptible population
    size. This is a genuine modeling assumption you have to state and
    justify in your report -- e.g. estimated hostel/campus population
    within the outbreak's catchment area. It is NOT something the data
    itself tells you. Swap the default before treating outputs as real.
    """
    days = daily_features["day"].to_numpy(dtype=float)
    observed = daily_features["case_count"].to_numpy(dtype=float)
    I0 = max(observed[0], 1.0)

    def model_fn(t, beta, gamma):
        return _simulate_cumulative_infected(t, beta, gamma, population_estimate, I0)

    # Reasonable starting guesses + bounds so the optimizer doesn't wander
    # into beta/gamma values that make the ODE numerically unstable.
    (beta_fit, gamma_fit), _ = curve_fit(
        model_fn,
        days,
        observed,
        p0=[0.4, 0.1],
        bounds=([1e-4, 1e-4], [5.0, 5.0]),
        maxfev=5000,
    )

    return SIRModel(
        beta=float(beta_fit), gamma=float(gamma_fit), N=population_estimate, I0=I0, t0=float(days[0])
    )


def forecast_case_count(sir_model: SIRModel, target_day: int) -> float:
    """Predict cumulative case count on target_day."""
    curve = sir_model.predict_cumulative_cases(np.array([target_day], dtype=float))
    return float(curve[0])


# ---------------------------------------------------------------------
# 3. Blended forecast (what the frontend will actually draw)
# ---------------------------------------------------------------------

def forecast_outbreak(
    daily_features: pd.DataFrame,
    target_day: int,
    population_estimate: int,
) -> dict:
    """
    Combine the geographic trend and the SIR case-count forecast into
    a single prediction: predicted centroid, and a radius that's been
    scaled up/down based on how much SIR expects the case count to
    change relative to the most recent observed day.

    Scaling logic: outbreak AREA roughly tracks case count (more cases
    -> more ground covered), and radius ~ sqrt(area). So:
        radius_scaled = radius_trend * sqrt(predicted_cases / last_observed_cases)
    """
    geo = forecast_geography(daily_features, target_day)

    sir = fit_sir(daily_features, population_estimate)
    predicted_cases = forecast_case_count(sir, target_day)

    last_observed_day = daily_features["day"].max()
    last_observed_cases = daily_features.loc[
        daily_features["day"] == last_observed_day, "case_count"
    ].iloc[0]

    case_scale = np.sqrt(max(predicted_cases, 1) / max(last_observed_cases, 1))
    scaled_radius = geo["spread_radius_km"] * case_scale

    return {
        "target_day": target_day,
        "centroid_lat": geo["centroid_lat"],
        "centroid_lng": geo["centroid_lng"],
        "radius_km_trend_only": geo["spread_radius_km"],
        "radius_km_sir_scaled": scaled_radius,
        "radius_uncertainty_km": geo["radius_uncertainty_km"],
        "predicted_case_count": predicted_cases,
        "sir_beta": sir.beta,
        "sir_gamma": sir.gamma,
        "sir_population_assumed": population_estimate,
    }


# ---------------------------------------------------------------------
# 4. Demo / smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    cases = load_cases("/tmp/demo_export.csv")
    daily = compute_daily_features(cases, cumulative=True)

    result = forecast_outbreak(daily, target_day=21, population_estimate=3000)

    print("7-day-ahead forecast (day 21, from day 14 data):")
    for k, v in result.items():
        print(f"  {k}: {v}")