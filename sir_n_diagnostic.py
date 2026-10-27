"""
Reproducible diagnostic for forecasting_engine.optimize_sir_parameters.

Runs the optimiser from several starting values of N on:
  1. the 11-day demo series from forecasting_engine.py,
  2. a simulated SIR epidemic with known parameters (N=5000, beta=0.4, gamma=0.1, 40 days),
  3. the first 30 daily aggregate values of data/processed_dengue_mh.csv (as in test_forecasting_1.py),
     if that file exists.

Every result is written to research_outputs/sir_n_diagnostic.csv so that any number quoted in the
README or report can be checked against this file. Quote the numbers from YOUR run of this script.

Usage:  python3 sir_n_diagnostic.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import odeint

from forecasting_engine import optimize_sir_parameters, sir_model_ode

START_VALUES = [5_000, 50_000, 100_000]
OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

series = {"demo series (11 days)": np.array([1, 2, 5, 12, 25, 45, 70, 110, 150, 180, 200], dtype=float)}

t = np.arange(0, 40)
sol = odeint(sir_model_ode, [4999.0, 1.0, 0.0], t, args=(5000, 0.4, 0.1))
cumulative = sol[:, 1] + sol[:, 2]
series["simulated SIR, true N=5000, beta=0.4, gamma=0.1 (40 days)"] = np.diff(np.concatenate([[1.0], cumulative]))

dengue_path = Path("data/processed_dengue_mh.csv")
if dengue_path.exists():
    d = pd.read_csv(dengue_path)
    series["dengue Maharashtra, first 30 aggregated days"] = d.groupby("Timeline_Day").size().values[:30].astype(float)
else:
    print(f"NOTE: {dengue_path} not found; dengue diagnostic skipped.")

rows = []
for name, daily in series.items():
    for n0 in START_VALUES:
        r = optimize_sir_parameters(daily, initial_population=n0)
        rows.append(dict(series=name, n_days=len(daily), initial_N=n0, optimized_N=r["optimized_N"],
                         change_in_N=round(r["optimized_N"] - n0, 2), optimized_beta=r["optimized_beta"],
                         optimized_gamma=r["optimized_gamma"], optimizer_success=r["success"]))

res = pd.DataFrame(rows)
res.to_csv(OUT / "sir_n_diagnostic.csv", index=False)
pd.set_option("display.width", 220, "display.max_colwidth", 60)
print(res.to_string(index=False))
print(f"\nSaved {OUT / 'sir_n_diagnostic.csv'}")