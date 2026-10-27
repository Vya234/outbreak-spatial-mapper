"""
Robustness check for occurrence_model.py: does next-month reported-occurrence prediction hold up
when evaluation is restricted to districts with a stronger reporting footprint?

DESIGN
- Same rolling-origin folds (test years 2016-2019), same features, same baseline and the same three
  GBM feature sets as occurrence_model.py (season+state, +own history, +neighbours).
- Models are trained on ALL districts exactly as in occurrence_model.py; only the TEST rows are filtered.
- A district is eligible in a test year if it had >= k recorded months of the target disease in the
  TRAINING period only (months before that test year). Eligibility never uses the test year or later.
- Thresholds k = 6 and 12 are exploratory choices, not established standards.

CAVEATS
- Base rates differ between eligibility sets, so PR-AUC is NOT comparable across sets. Compare each
  model with the baseline and with the base rate WITHIN the same set (see 'lift').
- Small sets (few districts / positives) give very noisy estimates; they are flagged LOW-N. The cut-off
  (fewer than 30 positives in any test year) is a practical warning rule, not a statistical standard.
- A district with many past reports is not necessarily fully monitored, and unreported months still
  cannot be confirmed as true negatives. This tests robustness to reporting footprint only.
- The code for the panel is copied from occurrence_model.py so that script and its saved outputs stay unchanged.

Usage:  python occurrence_robustness.py data/Final_data.csv research_outputs
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/Final_data.csv"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "research_outputs")
OUT.mkdir(parents=True, exist_ok=True)

DISEASES = ["Acute Diarrhoeal Disease", "Dengue"]
TEST_YEARS = [2016, 2017, 2018, 2019]
K_NEIGHBOURS = 5
ELIGIBILITY = [("all districts", 0), (">=6 recorded months (training)", 6), (">=12 recorded months (training)", 12)]
LOW_N_POSITIVES = 30          # practical warning rule (not a statistical standard): LOW-N if any year has fewer positives

# ---------------------------------------------------------------- load (same as occurrence_model.py)
df = pd.read_csv(CSV)
for c in ["state_ut", "district", "Disease"]:
    df[c] = df[c].astype("string").str.strip()
df["Date"] = pd.to_datetime(dict(year=df.year, month=df.mon, day=df.day))
months = pd.period_range("2009-05", "2022-06", freq="M")
T = len(months)
month_idx = {m: i for i, m in enumerate(months)}
df["t"] = df["Date"].dt.to_period("M").map(month_idx)

dist = (df.groupby(["state_ut", "district"])
          .agg(lat=("Latitude", "first"), lon=("Longitude", "first")).reset_index())
D = len(dist)
did = {(s, d): i for i, (s, d) in enumerate(zip(dist.state_ut, dist.district))}
df["d"] = [did[(s, d)] for s, d in zip(df.state_ut, df.district)]
state_codes = dist.state_ut.astype("category").cat.codes.values

la, lo = np.radians(dist.lat.values), np.radians(dist.lon.values)
a = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
     + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
dd = 2 * 6371 * np.arcsin(np.sqrt(a))
np.fill_diagonal(dd, np.inf)
NBR = np.argsort(dd, axis=1)[:, :K_NEIGHBOURS]


def roll_sum(M, w):
    c = np.cumsum(np.pad(M, ((0, 0), (1, 0))), axis=1)
    return np.stack([c[:, m + 1] - c[:, max(0, m + 1 - w)] for m in range(M.shape[1])], axis=1)


def build_panel(disease):
    Y = np.zeros((D, T))
    s = df[df.Disease == disease]
    Y[s.d.values, s.t.values] = 1
    A = np.zeros((D, T))
    A[df.d.values, df.t.values] = 1
    own3, own12, any3 = roll_sum(Y, 3), roll_sum(Y, 12), roll_sum(A, 3)
    nb1, nb3 = Y[NBR].mean(axis=1), own3[NBR].mean(axis=1)
    st1, st_any3 = np.zeros((D, T)), np.zeros((D, T))
    for sc in np.unique(state_codes):
        idx = np.where(state_codes == sc)[0]
        st1[idx] = Y[idx].sum(0, keepdims=True) - Y[idx]
        st_any3[idx] = any3[idx].sum(0, keepdims=True) - any3[idx]
    frames = []
    for m in range(T - 1):
        tm = months[m + 1].month
        frames.append(pd.DataFrame(dict(
            d=np.arange(D), t=m + 1, y=Y[:, m + 1],
            sin=np.sin(2 * np.pi * tm / 12), cos=np.cos(2 * np.pi * tm / 12),
            st1=st1[:, m], st_any3=st_any3[:, m], any3=any3[:, m],
            own1=Y[:, m], own3=own3[:, m], own12=own12[:, m], nb1=nb1[:, m], nb3=nb3[:, m])))
    return pd.concat(frames, ignore_index=True), Y


SETS = {
    "Season only": ["sin", "cos"],

    "Season + st1": [
        "sin", "cos", "st1"
    ],

    "Season + state 3-month": [
        "sin", "cos", "st_any3"
    ],

    "Season + both state features": [
        "sin", "cos", "st1", "st_any3"
    ],

    "Season + state + own history": [
        "sin", "cos", "st1", "st_any3",
        "any3", "own1", "own3", "own12"
    ],
}


def fit_gbm(X, y):
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                          random_state=0).fit(X, y)


rows = []
for disease in DISEASES:
    P, Y = build_panel(disease)
    for yr in TEST_YEARS:
        lo_t = month_idx[pd.Period(f"{yr}-01", "M")]
        hi_t = month_idx[pd.Period(f"{yr}-12", "M")]
        tr, te = P[P.t < lo_t], P[(P.t >= lo_t) & (P.t <= hi_t)].copy()
        recorded_months_in_training = Y[:, :lo_t].sum(axis=1)       # TRAINING period only
        models = {n: fit_gbm(tr[c], tr.y) for n, c in SETS.items()}
        te["persistence (own 12-month count)"] = te.own12.values + 0.01 * te.own3.values
        for n, c in SETS.items():
            te[n] = models[n].predict_proba(te[c])[:, 1]
        for label, k in ELIGIBILITY:
            eligible = np.where(recorded_months_in_training >= k)[0]
            sub = te[te.d.isin(eligible)]
            for model in ["persistence (own 12-month count)", *SETS]:
                ok = sub.y.sum() > 0 and sub.y.sum() < len(sub)
                rows.append(dict(
                    disease=disease, test_year=yr, eligibility=label, eligible_districts=len(eligible),
                    positives=int(sub.y.sum()), base_rate=round(sub.y.mean(), 4), model=model,
                    pr_auc=round(average_precision_score(sub.y, sub[model]), 4) if ok else np.nan,
                    roc_auc=round(roc_auc_score(sub.y, sub[model]), 4) if ok else np.nan))

res = pd.DataFrame(rows)
res.to_csv(OUT / "occurrence_robustness_by_year.csv", index=False)

# ---------------------------------------------------------------- summary
BASE = "persistence (own 12-month count)"
out = []
for (disease, elig), g in res.groupby(["disease", "eligibility"], sort=False):
    meta = g[g.model == BASE]
    wide = g.pivot(index="test_year", columns="model", values="pr_auc")
    row = dict(
        disease=disease, eligibility=elig,
        districts=f"{meta.eligible_districts.min()}-{meta.eligible_districts.max()}",
        positives_per_year=f"{meta.positives.min()}-{meta.positives.max()}",
        base_rate=round(meta.base_rate.mean(), 3),
        LOW_N="LOW-N" if meta.positives.min() < LOW_N_POSITIVES else "")
    for model, short in [
        (BASE, "baseline"),
        ("Season only", "season-only"),
        ("Season + st1", "season+st1"),
        ("Season + state 3-month", "state-3-month"),
        ("Season + both state features", "both-state"),
        ("Season + state + own history", "own-history"),
    ]:
        row[f"PR-AUC {short}"] = round(wide[model].mean(), 3)
    row["Own-history lift over base rate"] = round(wide["Season + state + own history"].mean() / meta.base_rate.mean(), 1)

    row["Years own-history > baseline (PR-AUC)"] = (f"{int((wide['Season + state + own history'] > wide[BASE]).sum())}"f"/{wide.shape[0]}")
    out.append(row)
summary = pd.DataFrame(out)
summary.to_csv(OUT / "occurrence_robustness_summary.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 30)
print("Robustness to training-period reporting footprint (mean over test years 2016-2019)")
print("PR-AUC is NOT comparable across eligibility sets (base rates differ); compare within a set.\n")
print(summary.to_string(index=False))
print(f"\nSaved {OUT}/occurrence_robustness_by_year.csv and occurrence_robustness_summary.csv")