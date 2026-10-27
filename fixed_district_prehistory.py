from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CSV = "data/Final_data.csv"
PRED_FILE = "research_outputs/later_validation/paired_predictions.csv"
OUT = Path("research_outputs/fixed_district_prehistory")
OUT.mkdir(parents=True, exist_ok=True)

DISEASES = ["Acute Diarrhoeal Disease", "Dengue"]
PERIODS = {
    "2020": ("2020-01", "2020-12"),
    "2021": ("2021-01", "2021-12"),
    "2022_H1": ("2022-01", "2022-06"),
}
BASE = "persistence (own 12-month count)"
MODELS = [BASE, "Logistic Regression", "GBM"]

# Load data and reproduce the district IDs from the original validation script.
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
    raise ValueError("Found dates outside the modeled month range.")

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

D, T = len(dist), len(months)
any_record = np.zeros((D, T), dtype=np.uint8)
any_record[df.d.to_numpy(), df.t.astype(int).to_numpy()] = 1

pred = pd.read_csv(PRED_FILE)
required = {
    "d", "t", "y", BASE, "GBM", "Logistic Regression",
    "disease", "test_period", "eligibility"
}
missing = required - set(pred.columns)
if missing:
    raise ValueError(f"Missing prediction columns: {sorted(missing)}")

# Only the original all-district prediction rows are used.
pred = pred[
    pred["eligibility"].eq("all districts")
    & pred["disease"].isin(DISEASES)
    & pred["test_period"].isin(PERIODS)
].copy()

if pred.empty:
    raise ValueError("No matching prediction rows found.")

if pred["d"].min() < 0 or pred["d"].max() >= D:
    raise ValueError("District IDs do not match the original district mapping.")

# Check source labels against the target-disease district-month panel.
for disease in DISEASES:
    source = df[df["Disease"].eq(disease)]
    truth = set(zip(source["d"].astype(int), source["t"].astype(int)))
    part = pred[pred["disease"].eq(disease)]
    for row in part.itertuples(index=False):
        expected = int((int(row.d), int(row.t)) in truth)
        if int(row.y) != expected:
            raise ValueError(
                f"Outcome mismatch for {disease}, district {row.d}, month {row.t}"
            )

metrics = []
coverage = []
cohort_members = []

for disease in DISEASES:
    for period, (start_str, end_str) in PERIODS.items():
        sub = pred[
            pred["disease"].eq(disease)
            & pred["test_period"].eq(period)
        ].copy()

        start = pd.Period(start_str, freq="M")
        end = pd.Period(end_str, freq="M")
        lo_t, hi_t = month_idx[start], month_idx[end]

        # Use ONLY pre-test any-disease-record history to select districts.
        pre_months = any_record[:, :lo_t].sum(axis=1)
        cohorts = [
            ("prehistory: >=1 any-record month", pre_months >= 1),
            ("prehistory: >=6 any-record months", pre_months >= 6),
        ]

        for cohort_name, mask in cohorts:
            ids = set(np.flatnonzero(mask))
            matched = sub[sub["d"].isin(ids)].copy()
            matched = matched.dropna(subset=["y"] + MODELS)

            if matched.empty:
                continue

            # Keep only rows inside the declared test window.
            matched = matched[
                matched["t"].between(lo_t, hi_t)
            ].copy()

            if matched.empty:
                continue

            # Save the actual selected district list for reproducibility.
            for d in sorted(set(matched["d"].astype(int))):
                cohort_members.append({
                    "disease": disease,
                    "test_period": period,
                    "cohort": cohort_name,
                    "district_id": d,
                    "state_ut": dist.iloc[d]["state_ut"],
                    "district": dist.iloc[d]["district"],
                    "pretest_any_record_months": int(pre_months[d]),
                })

            positives = int(matched["y"].sum())
            n = len(matched)
            valid = matched["y"].nunique() == 2

            coverage.append({
                "disease": disease,
                "test_period": period,
                "cohort": cohort_name,
                "districts": matched["d"].nunique(),
                "district_month_rows": n,
                "positives": positives,
                "base_rate": matched["y"].mean(),
                "test_months": hi_t - lo_t + 1,
                "min_pretest_record_months": int(
                    matched["d"].map(
                        lambda d: pre_months[int(d)]
                    ).min()
                ),
            })

            for model in MODELS:
                metrics.append({
                    "disease": disease,
                    "test_period": period,
                    "cohort": cohort_name,
                    "model": model,
                    "districts": matched["d"].nunique(),
                    "district_month_rows": n,
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

metrics_df = pd.DataFrame(metrics)
coverage_df = pd.DataFrame(coverage)
members_df = pd.DataFrame(cohort_members)

metrics_df.to_csv(OUT / "metrics.csv", index=False)
coverage_df.to_csv(OUT / "coverage.csv", index=False)
members_df.to_csv(OUT / "cohort_districts.csv", index=False)

print("\nPREHISTORY-ONLY COHORT COVERAGE")
print(coverage_df.round(4).to_string(index=False))

print("\nMODEL METRICS")
print(metrics_df.round(4).to_string(index=False))

print("\nGBM AND LOGISTIC REGRESSION DIFFERENCES")
if not metrics_df.empty:
    wide = metrics_df.pivot(
        index=["disease", "test_period", "cohort"],
        columns="model",
        values="PR_AUC",
    ).reset_index()
    wide["GBM_minus_persistence"] = wide["GBM"] - wide[BASE]
    wide["LogReg_minus_persistence"] = wide["Logistic Regression"] - wide[BASE]
    wide["GBM_minus_LogReg"] = wide["GBM"] - wide["Logistic Regression"]
    print(wide.round(4).to_string(index=False))
    wide.to_csv(OUT / "model_comparison.csv", index=False)

print(f"\nSaved results to {OUT}/")
print(
    "Note: cohort selection uses pre-test any-disease records only. "
    "This avoids selecting districts based on future records, but does not "
    "correct incomplete or disease-specific reporting."
)
