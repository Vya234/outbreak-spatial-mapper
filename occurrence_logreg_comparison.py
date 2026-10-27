from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

CSV = "data/Final_data.csv"
OUT = Path("research_outputs/logistic_comparison")
OUT.mkdir(parents=True, exist_ok=True)

DISEASES = ["Acute Diarrhoeal Disease", "Dengue"]
TEST_YEARS = [2016, 2017, 2018, 2019]
ELIGIBILITY = [
    ("all districts", 0),
    (">=6 recorded months (training)", 6),
    (">=12 recorded months (training)", 12),
]

BASE = "persistence (own 12-month count)"
FEATURES = [
    "sin", "cos", "st1", "st_any3",
    "any3", "own1", "own3", "own12",
]

df = pd.read_csv(CSV)
for c in ["state_ut", "district", "Disease"]:
    df[c] = df[c].astype("string").str.strip()

df["Date"] = pd.to_datetime(
    dict(year=df.year, month=df.mon, day=df.day)
)
months = pd.period_range("2009-05", "2022-06", freq="M")
T = len(months)
month_idx = {m: i for i, m in enumerate(months)}
df["t"] = df["Date"].dt.to_period("M").map(month_idx)

dist = (
    df.groupby(["state_ut", "district"])
      .agg(lat=("Latitude", "first"), lon=("Longitude", "first"))
      .reset_index()
)
D = len(dist)
did = {
    (s, d): i
    for i, (s, d) in enumerate(zip(dist.state_ut, dist.district))
}
df["d"] = [did[(s, d)] for s, d in zip(df.state_ut, df.district)]
state_codes = dist.state_ut.astype("category").cat.codes.values


def roll_sum(M, w):
    c = np.cumsum(np.pad(M, ((0, 0), (1, 0))), axis=1)
    return np.stack([
        c[:, m + 1] - c[:, max(0, m + 1 - w)]
        for m in range(M.shape[1])
    ], axis=1)


def build_panel(disease):
    Y = np.zeros((D, T))
    s = df[df.Disease == disease]
    Y[s.d.values, s.t.values] = 1

    A = np.zeros((D, T))
    A[df.d.values, df.t.values] = 1

    own3 = roll_sum(Y, 3)
    own12 = roll_sum(Y, 12)
    any3 = roll_sum(A, 3)

    st1 = np.zeros((D, T))
    st_any3 = np.zeros((D, T))

    for sc in np.unique(state_codes):
        idx = np.where(state_codes == sc)[0]
        st1[idx] = Y[idx].sum(0, keepdims=True) - Y[idx]
        st_any3[idx] = any3[idx].sum(0, keepdims=True) - any3[idx]

    frames = []
    for m in range(T - 1):
        target_month = months[m + 1].month
        frames.append(pd.DataFrame({
            "d": np.arange(D),
            "t": m + 1,
            "y": Y[:, m + 1],
            "sin": np.sin(2 * np.pi * target_month / 12),
            "cos": np.cos(2 * np.pi * target_month / 12),
            "st1": st1[:, m],
            "st_any3": st_any3[:, m],
            "any3": any3[:, m],
            "own1": Y[:, m],
            "own3": own3[:, m],
            "own12": own12[:, m],
        }))

    return pd.concat(frames, ignore_index=True), Y


def make_gbm():
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=200,
        random_state=0,
    )


def make_logreg():
    # Scaling is fitted on training rows only through the pipeline.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs"),
    )


results = []
predictions = []

for disease in DISEASES:
    P, Y = build_panel(disease)

    for year in TEST_YEARS:
        lo_t = month_idx[pd.Period(f"{year}-01", "M")]
        hi_t = month_idx[pd.Period(f"{year}-12", "M")]

        tr = P[P.t < lo_t]
        te = P[(P.t >= lo_t) & (P.t <= hi_t)].copy()

        # Eligibility uses training-period outcomes only.
        recorded_months = Y[:, :lo_t].sum(axis=1)

        # Fit both models on exactly the same training rows/features.
        gbm = make_gbm().fit(tr[FEATURES], tr.y)
        logreg = make_logreg().fit(tr[FEATURES], tr.y)

        te["GBM"] = gbm.predict_proba(te[FEATURES])[:, 1]
        te["Logistic Regression"] = logreg.predict_proba(
            te[FEATURES]
        )[:, 1]
        te[BASE] = te.own12 + 0.01 * te.own3

        for label, k in ELIGIBILITY:
            eligible = np.where(recorded_months >= k)[0]
            sub = te[te.d.isin(eligible)].copy()

            # Save paired predictions for a later paired bootstrap.
            predictions.append(
                sub[["d", "t", "y", BASE, "GBM", "Logistic Regression"]]
                .assign(
                    disease=disease,
                    test_year=year,
                    eligibility=label,
                )
            )

            valid = sub.y.nunique() == 2
            for model in [BASE, "GBM", "Logistic Regression"]:
                ap = (
                    average_precision_score(sub.y, sub[model])
                    if valid else np.nan
                )
                roc = (
                    roc_auc_score(sub.y, sub[model])
                    if valid else np.nan
                )
                results.append({
                    "disease": disease,
                    "test_year": year,
                    "eligibility": label,
                    "districts": sub.d.nunique(),
                    "positives": int(sub.y.sum()),
                    "base_rate": sub.y.mean(),
                    "model": model,
                    "PR_AUC": ap,
                    "ROC_AUC": roc,
                })

res = pd.DataFrame(results)
res.to_csv(OUT / "yearly_metrics.csv", index=False)

pred = pd.concat(predictions, ignore_index=True)
pred.to_csv(OUT / "paired_predictions.csv", index=False)

wide = res.pivot(
    index=["disease", "test_year", "eligibility"],
    columns="model",
    values="PR_AUC",
).reset_index()

wide["GBM_minus_persistence"] = wide["GBM"] - wide[BASE]
wide["LogReg_minus_persistence"] = wide["Logistic Regression"] - wide[BASE]
wide["GBM_minus_LogReg"] = wide["GBM"] - wide["Logistic Regression"]

wide.to_csv(OUT / "model_comparison.csv", index=False)

print("\nPR-AUC comparison by disease, year, and eligibility")
print("Compare models within each row; base rates vary between rows.\n")
print(wide.round(4).to_string(index=False))
print(f"\nSaved results to: {OUT}")
