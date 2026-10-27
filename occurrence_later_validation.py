from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

CSV = "data/Final_data.csv"
OUT = Path("research_outputs/later_validation")
OUT.mkdir(parents=True, exist_ok=True)

DISEASES = ["Acute Diarrhoeal Disease", "Dengue"]
PERIODS = [
    ("2020", "2020-01", "2020-12"),
    ("2021", "2021-01", "2021-12"),
    ("2022_H1", "2022-01", "2022-06"),
]
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
    dict(year=df["year"], month=df["mon"], day=df["day"])
)

months = pd.period_range("2009-05", "2022-06", freq="M")
T = len(months)
month_idx = {m: i for i, m in enumerate(months)}
df["t"] = df["Date"].dt.to_period("M").map(month_idx)

if df["t"].isna().any():
    raise ValueError("Some dataset dates fall outside the modeled month range.")

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
state_codes = dist.state_ut.astype("category").cat.codes.to_numpy()


def roll_sum(M, w):
    c = np.cumsum(np.pad(M, ((0, 0), (1, 0))), axis=1)
    return np.stack([
        c[:, m + 1] - c[:, max(0, m + 1 - w)]
        for m in range(M.shape[1])
    ], axis=1)


def build_panel(disease):
    Y = np.zeros((D, T))
    s = df[df.Disease == disease]
    Y[s.d.to_numpy(), s.t.astype(int).to_numpy()] = 1

    A = np.zeros((D, T))
    A[df.d.to_numpy(), df.t.astype(int).to_numpy()] = 1

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
        max_depth=4, learning_rate=0.05,
        max_iter=200, random_state=0
    )


def make_logreg():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
    )


metrics = []
predictions = []

for disease in DISEASES:
    P, Y = build_panel(disease)

    for period_label, start_str, end_str in PERIODS:
        lo_t = month_idx[pd.Period(start_str, freq="M")]
        hi_t = month_idx[pd.Period(end_str, freq="M")]

        # Expanding training window: all target months before this period.
        tr = P[P.t < lo_t]
        te_all = P[(P.t >= lo_t) & (P.t <= hi_t)].copy()

        if tr.empty or te_all.empty:
            raise ValueError(f"Empty train/test split: {disease}, {period_label}")

        recorded_months = Y[:, :lo_t].sum(axis=1)

        gbm = make_gbm().fit(tr[FEATURES], tr.y)
        lr = make_logreg().fit(tr[FEATURES], tr.y)

        te_all["GBM"] = gbm.predict_proba(te_all[FEATURES])[:, 1]
        te_all["Logistic Regression"] = lr.predict_proba(
            te_all[FEATURES]
        )[:, 1]
        te_all[BASE] = te_all.own12 + 0.01 * te_all.own3

        for label, k in ELIGIBILITY:
            eligible = np.flatnonzero(recorded_months >= k)
            sub = te_all[te_all.d.isin(eligible)].copy()

            predictions.append(
                sub[["d", "t", "y", BASE, "GBM", "Logistic Regression"]]
                .assign(
                    disease=disease,
                    test_period=period_label,
                    eligibility=label,
                )
            )

            valid = sub.y.nunique() == 2
            for model in [BASE, "Logistic Regression", "GBM"]:
                metrics.append({
                    "disease": disease,
                    "test_period": period_label,
                    "start_month": start_str,
                    "end_month": end_str,
                    "eligibility": label,
                    "districts": sub.d.nunique(),
                    "months_per_district": hi_t - lo_t + 1,
                    "positives": int(sub.y.sum()),
                    "base_rate": sub.y.mean(),
                    "model": model,
                    "PR_AUC": (
                        average_precision_score(sub.y, sub[model])
                        if valid else np.nan
                    ),
                    "ROC_AUC": (
                        roc_auc_score(sub.y, sub[model])
                        if valid else np.nan
                    ),
                })

res = pd.DataFrame(metrics)
res.to_csv(OUT / "yearly_metrics.csv", index=False)

pred = pd.concat(predictions, ignore_index=True)
pred.to_csv(OUT / "paired_predictions.csv", index=False)

wide = res.pivot(
    index=["disease", "test_period", "eligibility"],
    columns="model",
    values="PR_AUC",
).reset_index()

wide["GBM_minus_persistence"] = wide["GBM"] - wide[BASE]
wide["LogReg_minus_persistence"] = wide["Logistic Regression"] - wide[BASE]
wide["GBM_minus_LogReg"] = wide["GBM"] - wide["Logistic Regression"]
wide.to_csv(OUT / "model_comparison.csv", index=False)

print("\nLater-period validation: PR-AUC")
print("Compare models within each disease/period/eligibility group.")
print("2022_H1 covers January through June only.\n")
print(wide.round(4).to_string(index=False))

print("\nOutcome counts and evaluation coverage:")
print(
    res[res.model == BASE][[
        "disease", "test_period", "eligibility",
        "districts", "months_per_district", "positives", "base_rate"
    ]].to_string(index=False)
)

print(f"\nSaved results to: {OUT}")
