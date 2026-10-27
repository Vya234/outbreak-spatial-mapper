from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PRED = Path("research_outputs/logistic_comparison/paired_predictions.csv")
CSV = "data/Final_data.csv"
OUT = Path("research_outputs/logistic_comparison")
N_BOOT = 2000
SEED = 2027

df = pd.read_csv(PRED)
raw = pd.read_csv(CSV)

for c in ["state_ut", "district"]:
    raw[c] = raw[c].astype("string").str.strip()

# Recreate the exact district-ID order used by the modeling script.
dist = (
    raw.groupby(["state_ut", "district"])
       .agg(lat=("Latitude", "first"), lon=("Longitude", "first"))
       .reset_index()
)
state_by_district = dist["state_ut"].to_numpy()

df["state_ut"] = state_by_district[df["d"].astype(int).to_numpy()]

assert df.groupby("d")["state_ut"].nunique().max() == 1
assert df["state_ut"].nunique() == 36

rng = np.random.default_rng(SEED)
results = []

for (disease, year, eligibility), g in df.groupby(
    ["disease", "test_year", "eligibility"], sort=False
):
    states = np.sort(g["state_ut"].unique())
    blocks = {s: g[g.state_ut == s] for s in states}

    y0 = g["y"].to_numpy()
    if np.unique(y0).size < 2:
        continue

    p_gbm0 = g["GBM"].to_numpy()
    p_lr0 = g["Logistic Regression"].to_numpy()
    p_base0 = g["persistence (own 12-month count)"].to_numpy()

    ap_gbm0 = average_precision_score(y0, p_gbm0)
    ap_lr0 = average_precision_score(y0, p_lr0)
    ap_base0 = average_precision_score(y0, p_base0)

    # Positive delta means GBM has higher average precision.
    observed = ap_gbm0 - ap_lr0

    deltas = []
    attempts = 0
    while len(deltas) < N_BOOT and attempts < N_BOOT * 5:
        attempts += 1
        sampled = rng.choice(states, size=len(states), replace=True)
        parts = [blocks[s] for s in sampled]
        b = pd.concat(parts, ignore_index=True)

        y = b["y"].to_numpy()
        if np.unique(y).size < 2:
            continue

        ap_gbm = average_precision_score(y, b["GBM"])
        ap_lr = average_precision_score(y, b["Logistic Regression"])
        deltas.append(ap_gbm - ap_lr)

    if not deltas:
        print("Skipped:", disease, year, eligibility)
        continue

    deltas = np.asarray(deltas)
    low, high = np.quantile(deltas, [0.025, 0.975])

    results.append({
        "disease": disease,
        "test_year": year,
        "eligibility": eligibility,
        "states": len(states),
        "districts": g["d"].nunique(),
        "GBM_PR_AUC": ap_gbm0,
        "LogReg_PR_AUC": ap_lr0,
        "Persistence_PR_AUC": ap_base0,
        "GBM_minus_LogReg": observed,
        "CI_low": low,
        "CI_high": high,
        "fraction_bootstrap_delta_positive": np.mean(deltas > 0),
        "valid_replicates": len(deltas),
    })

    print(
        f"{disease} | {year} | {eligibility}: "
        f"GBM-LR={observed:+.4f}, "
        f"95% CI [{low:+.4f}, {high:+.4f}], "
        f"fraction positive={np.mean(deltas > 0):.3f}, "
        f"states={len(states)}"
    )

out = pd.DataFrame(results)
path = OUT / "logreg_state_bootstrap.csv"
out.to_csv(path, index=False)
print(f"\nSaved: {path}")
print("Positive delta favors GBM; negative delta favors logistic regression.")
print("Intervals are conditional on the fitted models and evaluated years.")
