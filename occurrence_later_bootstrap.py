from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PRED = Path("research_outputs/later_validation/paired_predictions.csv")
CSV = "data/Final_data.csv"
OUT = Path("research_outputs/later_validation")
N_BOOT = 2000
SEED = 2028

BASE = "persistence (own 12-month count)"
LR = "Logistic Regression"
GBM = "GBM"

pred = pd.read_csv(PRED)
raw = pd.read_csv(CSV)

for c in ["state_ut", "district"]:
    raw[c] = raw[c].astype("string").str.strip()

dist = (
    raw.groupby(["state_ut", "district"])
       .agg(lat=("Latitude", "first"), lon=("Longitude", "first"))
       .reset_index()
)
state_by_d = dist["state_ut"].to_numpy()
pred["state_ut"] = state_by_d[pred["d"].astype(int).to_numpy()]

assert pred.groupby("d")["state_ut"].nunique().max() == 1
assert pred["state_ut"].nunique() == 36

rng = np.random.default_rng(SEED)
results = []

for (disease, period, eligibility), g in pred.groupby(
    ["disease", "test_period", "eligibility"], sort=False
):
    states = np.sort(g["state_ut"].unique())
    blocks = {s: g[g["state_ut"] == s] for s in states}

    y0 = g["y"].to_numpy()
    if np.unique(y0).size < 2:
        continue

    observed = {
        "GBM_minus_persistence":
            average_precision_score(y0, g[GBM]) -
            average_precision_score(y0, g[BASE]),
        "LogReg_minus_persistence":
            average_precision_score(y0, g[LR]) -
            average_precision_score(y0, g[BASE]),
        "GBM_minus_LogReg":
            average_precision_score(y0, g[GBM]) -
            average_precision_score(y0, g[LR]),
    }

    boot = {k: [] for k in observed}
    attempts = 0
    while min(map(len, boot.values())) < N_BOOT and attempts < N_BOOT * 10:
        attempts += 1
        sampled = rng.choice(states, size=len(states), replace=True)
        b = pd.concat([blocks[s] for s in sampled], ignore_index=True)
        y = b["y"].to_numpy()

        if np.unique(y).size < 2:
            continue

        ap_base = average_precision_score(y, b[BASE])
        ap_lr = average_precision_score(y, b[LR])
        ap_gbm = average_precision_score(y, b[GBM])

        boot["GBM_minus_persistence"].append(ap_gbm - ap_base)
        boot["LogReg_minus_persistence"].append(ap_lr - ap_base)
        boot["GBM_minus_LogReg"].append(ap_gbm - ap_lr)

    for comparison, delta in observed.items():
        values = np.asarray(boot[comparison])
        if len(values) == 0:
            continue

        low, high = np.quantile(values, [0.025, 0.975])
        results.append({
            "disease": disease,
            "test_period": period,
            "eligibility": eligibility,
            "comparison": comparison,
            "observed_delta": delta,
            "CI_low": low,
            "CI_high": high,
            "fraction_bootstrap_delta_positive": np.mean(values > 0),
            "valid_replicates": len(values),
            "states": len(states),
            "districts": g["d"].nunique(),
            "positives": int(g["y"].sum()),
        })

        print(
            f"{disease} | {period} | {eligibility} | {comparison}: "
            f"delta={delta:+.4f}, 95% CI [{low:+.4f}, {high:+.4f}], "
            f"positive fraction={np.mean(values > 0):.3f}, "
            f"replicates={len(values)}"
        )

out = pd.DataFrame(results)
path = OUT / "state_bootstrap_metrics.csv"
out.to_csv(path, index=False)
print(f"\nSaved: {path}")
print("Intervals are conditional on fitted models; pointwise, not multiplicity-adjusted.")
