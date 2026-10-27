from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DATA = "data/Final_data.csv"
PRED = "research_outputs/later_validation/paired_predictions.csv"
OUT = Path("research_outputs/common_pre2020_cohort")
OUT.mkdir(parents=True, exist_ok=True)

BASE = "persistence (own 12-month count)"
MODELS = [BASE, "Logistic Regression", "GBM"]
COMPARISONS = [
    ("GBM", BASE),
    ("Logistic Regression", BASE),
    ("GBM", "Logistic Regression"),
]
PERIODS = ["2020", "2021", "2022_H1"]
DISEASES = ["Acute Diarrhoeal Disease", "Dengue"]
N_BOOT = 2000
SEED = 42

# Recreate district IDs in the same sorted order as the original model script.
df = pd.read_csv(DATA)
for c in ["state_ut", "district", "Disease"]:
    df[c] = df[c].astype("string").str.strip()

df["month"] = pd.to_datetime(
    dict(year=df.year, month=df.mon, day=df.day)
).dt.to_period("M")

months = pd.period_range("2009-05", "2022-06", freq="M")
month_idx = {m: i for i, m in enumerate(months)}
df["t"] = df["month"].map(month_idx)

if df["t"].isna().any():
    raise ValueError("Some source dates fall outside the modeled month range.")

districts = (
    df.groupby(["state_ut", "district"])
      .agg(lat=("Latitude", "first"), lon=("Longitude", "first"))
      .reset_index()
)
districts["d"] = np.arange(len(districts))
D = len(districts)

df["d"] = [
    i for i in (
        df[["state_ut", "district"]]
        .apply(tuple, axis=1)
        .map({
            (r.state_ut, r.district): int(r.d)
            for r in districts.itertuples(index=False)
        })
    )
]

# A[d,t] = 1 if any disease was recorded for this district-month.
A = np.zeros((D, len(months)), dtype=np.uint8)
A[df["d"].to_numpy(), df["t"].astype(int).to_numpy()] = 1

cutoff = month_idx[pd.Period("2020-01", freq="M")]
pre2020_count = A[:, :cutoff].sum(axis=1)

# Same fixed district lists for every test period; selection uses pre-2020 data only.
cohorts = {
    "common: >=1 any-record month before 2020": pre2020_count >= 1,
    "common: >=6 any-record months before 2020": pre2020_count >= 6,
}

pred = pd.read_csv(PRED)
pred = pred[
    pred["eligibility"].eq("all districts")
    & pred["disease"].isin(DISEASES)
    & pred["test_period"].isin(PERIODS)
].copy()

if pred["d"].min() < 0 or pred["d"].max() >= D:
    raise ValueError("Prediction district IDs do not match the reconstructed IDs.")

pred = pred.merge(
    districts[["d", "state_ut"]],
    on="d",
    how="left",
    validate="many_to_one",
)

if pred["state_ut"].isna().any():
    raise ValueError("Some prediction rows could not be mapped to a state.")

# Verify y against the source's target-disease records.
for disease in DISEASES:
    src = df[df["Disease"].eq(disease)]
    positive_pairs = set(zip(src["d"].astype(int), src["t"].astype(int)))
    rows = pred[pred["disease"].eq(disease)]
    for r in rows.itertuples(index=False):
        expected = int((int(r.d), int(r.t)) in positive_pairs)
        if int(r.y) != expected:
            raise ValueError(
                f"Outcome mismatch: {disease}, d={r.d}, t={r.t}"
            )

rng = np.random.default_rng(SEED)
metrics = []
boot_results = []
coverage = []
member_rows = []

for cohort_name, mask in cohorts.items():
    selected_ids = set(np.flatnonzero(mask))
    selected_districts = districts[districts["d"].isin(selected_ids)]

    # Save one common cohort membership file.
    for r in selected_districts.itertuples(index=False):
        member_rows.append({
            "cohort": cohort_name,
            "district_id": int(r.d),
            "state_ut": r.state_ut,
            "district": r.district,
            "pre2020_any_record_months": int(pre2020_count[int(r.d)]),
        })

    for disease in DISEASES:
        for period in PERIODS:
            sub = pred[
                pred["disease"].eq(disease)
                & pred["test_period"].eq(period)
                & pred["d"].isin(selected_ids)
            ].copy()
            sub = sub.dropna(subset=["y"] + MODELS + ["state_ut"])

            if sub.empty:
                continue

            # Ensure the exact same district IDs are represented for each period.
            present_ids = set(sub["d"].astype(int))
            if present_ids != selected_ids:
                raise ValueError(
                    f"District coverage mismatch for {cohort_name}, "
                    f"{disease}, {period}: expected {len(selected_ids)}, "
                    f"found {len(present_ids)}."
                )

            y = sub["y"].astype(int)
            if y.nunique() < 2:
                raise ValueError(
                    f"Only one class in {disease} {period} {cohort_name}."
                )

            scores = {
                m: average_precision_score(y, sub[m])
                for m in MODELS
            }
            roc = {
                m: roc_auc_score(y, sub[m])
                for m in MODELS
            }

            states = sub["state_ut"].unique()
            state_indices = {
                state: sub.index[sub["state_ut"].eq(state)].to_numpy()
                for state in states
            }

            coverage.append({
                "cohort": cohort_name,
                "disease": disease,
                "test_period": period,
                "districts": len(present_ids),
                "states": len(states),
                "district_month_rows": len(sub),
                "positive_rows": int(y.sum()),
                "base_rate": float(y.mean()),
            })

            for model in MODELS:
                metrics.append({
                    "cohort": cohort_name,
                    "disease": disease,
                    "test_period": period,
                    "model": model,
                    "PR_AUC": scores[model],
                    "ROC_AUC": roc[model],
                    "districts": len(present_ids),
                    "states": len(states),
                    "positive_rows": int(y.sum()),
                })

            boot = {pair: [] for pair in COMPARISONS}
            for _ in range(N_BOOT):
                chosen = rng.choice(states, size=len(states), replace=True)
                pieces = [sub.loc[state_indices[s]] for s in chosen]
                sample = pd.concat(pieces, ignore_index=True)
                sy = sample["y"].astype(int)

                if sy.nunique() < 2:
                    continue

                bs = {
                    m: average_precision_score(sy, sample[m])
                    for m in MODELS
                }
                for a, b in COMPARISONS:
                    boot[(a, b)].append(bs[a] - bs[b])

            for a, b in COMPARISONS:
                vals = np.asarray(boot[(a, b)])
                lo, hi = np.quantile(vals, [0.025, 0.975])
                boot_results.append({
                    "cohort": cohort_name,
                    "disease": disease,
                    "test_period": period,
                    "comparison": f"{a} minus {b}",
                    "model_A_PR_AUC": scores[a],
                    "model_B_PR_AUC": scores[b],
                    "difference": scores[a] - scores[b],
                    "CI_2.5%": lo,
                    "CI_97.5%": hi,
                    "bootstrap_reps_used": len(vals),
                    "states": len(states),
                    "districts": len(present_ids),
                    "positive_rows": int(y.sum()),
                })

pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
pd.DataFrame(boot_results).to_csv(OUT / "bootstrap_state_cluster.csv", index=False)
pd.DataFrame(coverage).to_csv(OUT / "coverage.csv", index=False)
pd.DataFrame(member_rows).to_csv(OUT / "cohort_districts.csv", index=False)

print("\nFIXED COHORT COVERAGE")
print(pd.DataFrame(coverage).round(4).to_string(index=False))

print("\nMODEL METRICS")
print(pd.DataFrame(metrics).round(4).to_string(index=False))

print("\nSTATE-CLUSTER BOOTSTRAP")
print(pd.DataFrame(boot_results).round(4).to_string(index=False))

print(f"\nSaved all files under: {OUT}")
print(
    "\nCaution: outcomes are recorded disease occurrences, not verified "
    "disease absence. Bootstrap intervals are pointwise, not adjusted "
    "for multiple comparisons."
)
