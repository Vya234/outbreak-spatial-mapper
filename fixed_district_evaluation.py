from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CSV = "data/Final_data.csv"
PRED_FILE = "research_outputs/later_validation/paired_predictions.csv"
OUT = Path("research_outputs/fixed_district")
OUT.mkdir(parents=True, exist_ok=True)

DISEASES = ["Acute Diarrhoeal Disease", "Dengue"]
PERIODS = ["2020", "2021", "2022_H1"]
BASE = "persistence (own 12-month count)"
MODELS = [BASE, "Logistic Regression", "GBM"]

# Load the original dataset and reproduce the original district IDs.
df = pd.read_csv(CSV)
for c in ["state_ut", "district", "Disease"]:
    df[c] = df[c].astype("string").str.strip()

df["month"] = pd.to_datetime(
    dict(year=df.year, month=df.mon, day=df.day)
).dt.to_period("M")

months = pd.period_range("2009-05", "2022-06", freq="M")
month_idx = {m: i for i, m in enumerate(months)}
df["t"] = df["month"].map(month_idx)

if df["t"].isna().any():
    raise ValueError("Dates outside the original modeled date range.")

dist = (
    df.groupby(["state_ut", "district"])
      .agg(lat=("Latitude", "first"), lon=("Longitude", "first"))
      .reset_index()
)
did = {
    (s, d): i
    for i, (s, d) in enumerate(zip(dist.state_ut, dist.district))
}
df["d"] = [did[(s, d)] for s, d in zip(df.state_ut, df.district)]

# Any-record presence, not confirmed disease-free status.
D = len(dist)
T = len(months)
A = np.zeros((D, T), dtype=np.uint8)
A[df.d.to_numpy(), df.t.astype(int).to_numpy()] = 1

pred = pd.read_csv(PRED_FILE)
required = {
    "d", "t", "y", BASE, "GBM", "Logistic Regression",
    "disease", "test_period", "eligibility"
}
missing = required - set(pred.columns)
if missing:
    raise ValueError(f"Missing columns in prediction file: {sorted(missing)}")

# Keep only the all-district predictions; every model then shares identical rows.
pred = pred[
    pred["eligibility"].eq("all districts")
    & pred["disease"].isin(DISEASES)
    & pred["test_period"].isin(PERIODS)
].copy()

# Validate IDs and outcomes against the source data's constructed panel.
if pred["d"].min() < 0 or pred["d"].max() >= D:
    raise ValueError("Prediction district IDs do not match the reconstructed panel.")

results = []
coverage = []

for disease in DISEASES:
    for period in PERIODS:
        sub = pred[
            pred["disease"].eq(disease)
            & pred["test_period"].eq(period)
        ].copy()

        if sub.empty:
            print(f"WARNING: no predictions for {disease}, {period}")
            continue

        start = {
            "2020": pd.Period("2020-01", freq="M"),
            "2021": pd.Period("2021-01", freq="M"),
            "2022_H1": pd.Period("2022-01", freq="M"),
        }[period]
        end = {
            "2020": pd.Period("2020-12", freq="M"),
            "2021": pd.Period("2021-12", freq="M"),
            "2022_H1": pd.Period("2022-06", freq="M"),
        }[period]

        lo_t, hi_t = month_idx[start], month_idx[end]

        # District has any recorded disease in the pre-test period.
        pre_any = A[:, :lo_t].sum(axis=1)
        # District has any recorded disease during the test period.
        test_any = A[:, lo_t:hi_t + 1].sum(axis=1)

        cohort_any = (pre_any > 0) & (test_any > 0)
        cohort_6 = (pre_any > 0) & (test_any > 0) & (pre_any >= 6)

        for cohort_name, cohort_mask in [
            ("fixed: any record before and during test", cohort_any),
            ("fixed: >=6 any-record months before and during test", cohort_6),
        ]:
            cohort_ids = set(np.flatnonzero(cohort_mask))
            matched = sub[sub["d"].isin(cohort_ids)].copy()

            # Verify the same target district-month rows are used for all models.
            matched = matched.dropna(subset=["y"] + MODELS)
            if matched.empty:
                continue

            positives = int(matched["y"].sum())
            negatives = int(len(matched) - positives)
            valid = matched["y"].nunique() == 2

            coverage.append({
                "disease": disease,
                "test_period": period,
                "cohort": cohort_name,
                "districts": matched["d"].nunique(),
                "district_month_rows": len(matched),
                "positives": positives,
                "negatives": negatives,
                "base_rate": matched["y"].mean(),
                "months_in_period": hi_t - lo_t + 1,
            })

            for model in MODELS:
                results.append({
                    "disease": disease,
                    "test_period": period,
                    "cohort": cohort_name,
                    "model": model,
                    "districts": matched["d"].nunique(),
                    "district_month_rows": len(matched),
                    "positives": positives,
                    "base_rate": matched["y"].mean(),
                    "PR_AUC": (
                        average_precision_score(matched["y"], matched[model])
                        if valid else np.nan
                    ),
                    "ROC_AUC": (
                        roc_auc_score(matched["y"], matched[model])
                        if valid else np.nan
                    ),
                })

res = pd.DataFrame(results)
cov = pd.DataFrame(coverage)

res.to_csv(OUT / "metrics.csv", index=False)
cov.to_csv(OUT / "coverage.csv", index=False)

print("\nFIXED-DISTRICT COHORT COVERAGE")
print(cov.round(4).to_string(index=False) if not cov.empty else "No rows")

print("\nMODEL METRICS")
print(res.round(4).to_string(index=False) if not res.empty else "No rows")

print(f"\nSaved files to {OUT}/metrics.csv and {OUT}/coverage.csv")
print(
    "\nCAVEAT: Cohorts require any-disease records during the test period. "
    "This is a retrospective sensitivity analysis and may induce selection "
    "bias; it does not establish complete surveillance."
)
