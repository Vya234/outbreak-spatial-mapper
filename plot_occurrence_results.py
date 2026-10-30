"""
Figures for the outbreak-occurrence experiment. Reads the CSVs written by occurrence_model.py,
so every plotted number comes from your own run.

Usage:  python plot_occurrence_results.py research_outputs
Writes: occurrence_fig_models.png, occurrence_fig_importance.png (in the same folder)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "research_outputs")
res = pd.read_csv(OUT / "occurrence_rolling_results.csv")
imp = pd.read_csv(OUT / "occurrence_permutation_importance.csv")

MODELS = [
    ("persistence (own 12-month count)", "Own 12-month\ncount (baseline)", "#9e9e9e"),
    ("GBM season+state",                 "GBM: season\n+ state activity", "#90caf9"),
    ("GBM +own history",                 "GBM: + own\ndistrict history", "#1976d2"),
    ("GBM +neighbours",                  "GBM: + neighbouring\ndistricts", "#0d47a1"),
]
DISEASES = list(res.disease.unique())
years = sorted(res.test_year.unique())

# ---------------------------------------------------------------- Figure 1: model comparison
fig, axes = plt.subplots(len(DISEASES), 2, figsize=(11, 3.9 * len(DISEASES)))
axes = np.atleast_2d(axes)
for r, disease in enumerate(DISEASES):
    d = res[res.disease == disease]
    base = d.base_rate.mean()
    for c, (metric, title) in enumerate([("pr_auc", "PR-AUC (higher is better)"),
                                         ("recall_at_k", "Recall in top-20 districts per month")]):
        ax = axes[r, c]
        means = [d[d.model == m][metric].mean() for m, _, _ in MODELS]
        sds = [d[d.model == m][metric].std(ddof=0) for m, _, _ in MODELS]
        ax.bar(range(len(MODELS)), means, yerr=sds, capsize=4, color=[col for _, _, col in MODELS],
               edgecolor="black", linewidth=0.5)
        for i, v in enumerate(means):
            ax.text(i, v + max(sds[i], 0.002) + 0.004 * max(means) / 0.1, f"{v:.3f}", ha="center", fontsize=9)
        if metric == "pr_auc":
            ax.axhline(base, color="red", ls="--", lw=1.2, label=f"Chance level (base rate = {base:.3f})")
            ax.legend(loc="upper left", fontsize=8, frameon=False)
        else:
            ax.axhline(20 / 795, color="red", ls="--", lw=1.2, label="Chance level (20 of 795 districts)")
            ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([lab for _, lab, _ in MODELS], fontsize=8)
        ax.set_title(f"{disease}: {title}", fontsize=10)
        ax.set_ylim(0, max(np.array(means) + np.array(sds)) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Next-month outbreak occurrence: rolling-origin evaluation, test years {years[0]}–{years[-1]}\n"
             "(mean ± SD over folds; folds share training data and are not independent)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "occurrence_fig_models.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: permutation importance
LABELS = {
    "own12": "Own district: reports in last 12 months", "own3": "Own district: reports in last 3 months",
    "own1": "Own district: report last month", "any3": "Own district: any-disease reports, 3 months",
    "nb1": "Neighbouring districts: last month", "nb3": "Neighbouring districts: last 3 months",
    "st1": "Other districts in state: last month", "st_any3": "State activity, any disease, 3 months",
    "sin": "Season (sin of month)", "cos": "Season (cos of month)",
}
# Stacked panels, larger fonts: the figure is printed at about 17 cm wide, so text sizes here are what you see on paper.
fig, axes = plt.subplots(len(DISEASES), 1, figsize=(8.5, 4.7 * len(DISEASES)))
axes = np.atleast_1d(axes)
for ax, disease in zip(axes, DISEASES):
    d = imp[imp.disease == disease].sort_values("pr_auc_drop")
    colors = ["#c62828" if v < 0 else "#1976d2" for v in d.pr_auc_drop]
    ax.barh([LABELS.get(f, f) for f in d.feature], d.pr_auc_drop, xerr=d["std"], color=colors,
            edgecolor="black", linewidth=0.5, capsize=3, height=0.7)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_title(disease, fontsize=13, fontweight="bold")
    ax.set_xlabel("Drop in PR-AUC when feature is shuffled", fontsize=11)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.tick_params(axis="x", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Permutation importance, test year {int(imp.test_year.iloc[0])}\n"
             "(predictors of REPORTED occurrence, not causes of spread)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "occurrence_fig_importance.png", dpi=200)
plt.close(fig)
print("wrote", OUT / "occurrence_fig_models.png", "and", OUT / "occurrence_fig_importance.png")

# ---------------------------------------------------------------- paired per-fold check (for the report text)
print("\nPer-fold paired comparison: GBM '+own history' vs 'own 12-month count' baseline")
for disease in DISEASES:
    d = res[res.disease == disease]
    p = d[d.model == "persistence (own 12-month count)"].set_index("test_year")
    g = d[d.model == "GBM +own history"].set_index("test_year")
    print(f"  {disease}: PR-AUC won in {int((g.pr_auc > p.pr_auc).sum())}/{len(g)} folds, "
          f"top-20 recall won in {int((g.recall_at_k > p.recall_at_k).sum())}/{len(g)} folds")