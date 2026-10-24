"""
Outbreak-occurrence forecasting on EpiClim (district-month panel).

TARGET   y(d, m+1) = 1 if district d has >= 1 reported outbreak of the disease in month m+1.
INPUTS   only information from months <= m (no future data, no same-month climate).
SPLITS   rolling-origin: for each test year, train on all earlier months, test on that year.
         2020-2022 are excluded: reporting collapses (151/367/188 records vs 572-1113 per year in 2010-2019).

Usage:   python occurrence_model.py data/Final_data.csv research_outputs

IMPORTANT INTERPRETATION CAVEAT
EpiClim rows are *reported* outbreaks. "No row" means "no report", not "no outbreak".
A district's own history therefore partly captures its tendency to report. Importance
scores must be described as predictors of *reported* occurrence, not as causes of spread.
Neighbours = 5 nearest district centroids (coordinates are fixed per district).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/Final_data.csv"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "research_outputs")
OUT.mkdir(parents=True, exist_ok=True)

DISEASES = ["Acute Diarrhoeal Disease", "Dengue"]
TEST_YEARS = [2016, 2017, 2018, 2019]
K_NEIGHBOURS = 5
TOP_K = 20          # districts a health officer could realistically follow up each month

# ---------------------------------------------------------------- load
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
    A[df.d.values, df.t.values] = 1                      # any disease (activity / reporting proxy)
    own3, own12, any3 = roll_sum(Y, 3), roll_sum(Y, 12), roll_sum(A, 3)
    nb1, nb3 = Y[NBR].mean(axis=1), own3[NBR].mean(axis=1)
    st1, st_any3 = np.zeros((D, T)), np.zeros((D, T))
    for sc in np.unique(state_codes):
        idx = np.where(state_codes == sc)[0]
        st1[idx] = Y[idx].sum(0, keepdims=True) - Y[idx]
        st_any3[idx] = any3[idx].sum(0, keepdims=True) - any3[idx]
    frames = []
    for m in range(T - 1):                               # features at m  ->  target at m+1
        tm = months[m + 1].month
        frames.append(pd.DataFrame(dict(
            d=np.arange(D), t=m + 1, y=Y[:, m + 1],
            sin=np.sin(2 * np.pi * tm / 12), cos=np.cos(2 * np.pi * tm / 12),
            st1=st1[:, m], st_any3=st_any3[:, m], any3=any3[:, m],
            own1=Y[:, m], own3=own3[:, m], own12=own12[:, m], nb1=nb1[:, m], nb3=nb3[:, m])))
    return pd.concat(frames, ignore_index=True)


SETS = {
    "season+state":     ["sin", "cos", "st1", "st_any3"],
    "+own history":     ["sin", "cos", "st1", "st_any3", "any3", "own1", "own3", "own12"],
    "+neighbours":      ["sin", "cos", "st1", "st_any3", "any3", "own1", "own3", "own12", "nb1", "nb3"],
}


def topk_metrics(test, score):
    t = test.assign(s=score)
    rec, prec = [], []
    for _, g in t.groupby("t"):
        if g.y.sum() == 0:
            continue
        top = g.nlargest(TOP_K, "s")
        rec.append(top.y.sum() / g.y.sum())
        prec.append(top.y.mean())
    return float(np.mean(rec)), float(np.mean(prec))


def fit_gbm(X, y):
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=0)
    return m.fit(X, y)


rows, imp_rows = [], []
for disease in DISEASES:
    P = build_panel(disease)
    for yr in TEST_YEARS:
        lo_t = month_idx[pd.Period(f"{yr}-01", "M")]
        hi_t = month_idx[pd.Period(f"{yr}-12", "M")]
        tr, te = P[P.t < lo_t], P[(P.t >= lo_t) & (P.t <= hi_t)]
        base = tr.y.mean()
        scores = {
            "prevalence": np.full(len(te), base),
            "persistence (own 12-month count)": te.own12.values + 0.01 * te.own3.values,
        }
        models = {}
        for name, cols in SETS.items():
            models[name] = fit_gbm(tr[cols], tr.y)
            scores[f"GBM {name}"] = models[name].predict_proba(te[cols])[:, 1]
        for name, s in scores.items():
            rec, prec = topk_metrics(te, s)
            is_prob = name.startswith("GBM") or name == "prevalence"
            rows.append(dict(
                disease=disease, test_year=yr, model=name, n_test=len(te), positives=int(te.y.sum()),
                base_rate=round(te.y.mean(), 4),
                pr_auc=round(average_precision_score(te.y, s), 4),
                roc_auc=round(roc_auc_score(te.y, s), 4) if np.std(s) > 0 else 0.5,
                brier=round(brier_score_loss(te.y, s), 4) if is_prob else np.nan,
                recall_at_k=round(rec, 3), precision_at_k=round(prec, 3)))
        if yr == TEST_YEARS[-1]:                         # importance on the latest fold, full model
            cols = SETS["+neighbours"]
            pi = permutation_importance(models["+neighbours"], te[cols], te.y, n_repeats=5,
                                        random_state=0, scoring="average_precision")
            for c, mu, sd in zip(cols, pi.importances_mean, pi.importances_std):
                imp_rows.append(dict(disease=disease, test_year=yr, feature=c,
                                     pr_auc_drop=round(mu, 4), std=round(sd, 4)))

res = pd.DataFrame(rows)
res.to_csv(OUT / "occurrence_rolling_results.csv", index=False)
pd.DataFrame(imp_rows).to_csv(OUT / "occurrence_permutation_importance.csv", index=False)

pd.set_option("display.width", 200)
summ = (res.groupby(["disease", "model"])[["pr_auc", "roc_auc", "recall_at_k", "precision_at_k", "base_rate"]]
          .mean().round(3))
print(f"Mean over test years {TEST_YEARS}; top-k = {TOP_K} districts per month\n")
print(summ.to_string())
print("\nPermutation importance (latest fold, drop in PR-AUC when shuffled):")
print(pd.DataFrame(imp_rows).sort_values(["disease", "pr_auc_drop"], ascending=[True, False]).to_string(index=False))
print(f"\nSaved to {OUT}/occurrence_rolling_results.csv and occurrence_permutation_importance.csv")