from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PRED = Path("research_outputs/later_validation/paired_predictions.csv")
COHORT = Path("research_outputs/fixed_district_prehistory/cohort_districts.csv")
OUT = Path("research_outputs/fixed_district_prehistory")
OUT.mkdir(parents=True, exist_ok=True)

BASE = "persistence (own 12-month count)"
MODELS = [BASE, "Logistic Regression", "GBM"]
COMPARISONS = [
    ("GBM", BASE),
    ("Logistic Regression", BASE),
    ("GBM", "Logistic Regression"),
]
N_BOOT = 2000
SEED = 42

pred = pd.read_csv(PRED)
cohort = pd.read_csv(COHORT)

pred = pred[
    pred["eligibility"].eq("all districts")
    & pred["disease"].isin(["Acute Diarrhoeal Disease", "Dengue"])
    & pred["test_period"].isin(["2020", "2021", "2022_H1"])
].copy()

# District IDs are mapped to states using the saved cohort membership file.
# Check each district has exactly one state label.
# Rebuild the complete district-to-state mapping using the same sorted
# district order as the original validation script.
source = pd.read_csv("data/Final_data.csv")
for c in ["state_ut", "district"]:
    source[c] = source[c].astype("string").str.strip()

districts = (
    source.groupby(["state_ut", "district"])
          .agg(lat=("Latitude", "first"), lon=("Longitude", "first"))
          .reset_index()
)
districts = districts.sort_values(
    ["state_ut", "district"], kind="stable"
).reset_index(drop=True)
districts["d"] = np.arange(len(districts))

mapping = districts[["d", "state_ut"]].copy()

pred = pred.merge(mapping, on="d", how="left", validate="many_to_one")
if pred["state_ut"].isna().any():
    raise ValueError("Some prediction district IDs have no state mapping.")

# Use the saved prehistory-only cohort for each disease, period, and threshold.
cohort_keys = ["disease", "test_period", "cohort", "district_id"]
cohort["d"] = cohort["district_id"].astype(int)

rng = np.random.default_rng(SEED)
results = []

for (disease, period, cohort_name), selected in cohort.groupby(
    ["disease", "test_period", "cohort"]
):
    ids = set(selected["d"].astype(int))
    sub = pred[
        pred["disease"].eq(disease)
        & pred["test_period"].eq(period)
        & pred["d"].isin(ids)
    ].copy()

    sub = sub.dropna(subset=["y"] + MODELS + ["state_ut"])
    if sub.empty or sub["y"].nunique() < 2:
        print("Skipping insufficient data:", disease, period, cohort_name)
        continue

    # Aggregate rows by state. Resample states, keeping all their district-month
    # rows together. Repeated states in a bootstrap draw are repeated clusters.
    states = sub["state_ut"].dropna().unique()
    state_rows = {
        s: sub.index[sub["state_ut"].eq(s)].to_numpy()
        for s in states
    }

    # Compute point estimates on the original evaluation set.
    point = {
        model: average_precision_score(sub["y"], sub[model])
        for model in MODELS
    }

    # Bootstrap paired differences, using the same resampled states for all models.
    boot_diffs = {
        (a, b): [] for a, b in COMPARISONS
    }

    for _ in range(N_BOOT):
        sampled_states = rng.choice(states, size=len(states), replace=True)
        pieces = []
        for s in sampled_states:
            pieces.append(sub.loc[state_rows[s]])
        sample = pd.concat(pieces, ignore_index=True)

        if sample["y"].nunique() < 2:
            continue

        scores = {
            model: average_precision_score(sample["y"], sample[model])
            for model in MODELS
        }
        for a, b in COMPARISONS:
            boot_diffs[(a, b)].append(scores[a] - scores[b])

    for a, b in COMPARISONS:
        vals = np.asarray(boot_diffs[(a, b)])
        if len(vals) == 0:
            lo = hi = np.nan
        else:
            lo, hi = np.quantile(vals, [0.025, 0.975])

        results.append({
            "disease": disease,
            "test_period": period,
            "cohort": cohort_name,
            "comparison": f"{a} minus {b}",
            "model_A_PR_AUC": point[a],
            "model_B_PR_AUC": point[b],
            "difference": point[a] - point[b],
            "CI_2.5%": lo,
            "CI_97.5%": hi,
            "bootstrap_reps_used": len(vals),
            "states": len(states),
            "districts": sub["d"].nunique(),
            "positive_rows": int(sub["y"].sum()),
        })

result = pd.DataFrame(results)
result.to_csv(OUT / "bootstrap_state_cluster.csv", index=False)

print("\nSTATE-CLUSTER BOOTSTRAP RESULTS")
print(result.round(4).to_string(index=False))
print(f"\nSaved to {OUT / 'bootstrap_state_cluster.csv'}")
print(
    "\nInterpretation: a 95% interval entirely above zero favors model A; "
    "an interval entirely below zero favors model B. Intervals crossing zero "
    "do not establish a clear difference. These are pointwise intervals."
)
