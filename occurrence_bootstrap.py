
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

INPUT = Path(
    "research_outputs/bootstrap/occurrence_bootstrap_predictions.csv"
)
OUT = Path("research_outputs/bootstrap")
N_BOOT = 2000
SEED = 42

BASE = "persistence (own 12-month count)"
MODEL = "Season + state + own history"

df = pd.read_csv(INPUT)

required = [
    "d", "y", BASE, MODEL, "disease", "test_year", "eligibility"
]
missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df[required].isna().any().any():
    raise ValueError("Missing values found in required columns.")

rng = np.random.default_rng(SEED)
results = []

groups = df.groupby(
    ["disease", "test_year", "eligibility"],
    sort=False
)

for (disease, year, eligibility), g in groups:
    # Each district must have one row per month in this test-year group.
    districts = np.sort(g["d"].unique())
    n_districts = len(districts)

    # Organize all months belonging to each district.
    blocks = {
        d: g[g["d"] == d].sort_values("t")
        for d in districts
    }

    y_all = g["y"].to_numpy()
    p_base_all = g[BASE].to_numpy()
    p_model_all = g[MODEL].to_numpy()

    if np.unique(y_all).size < 2:
        print(f"Skipping {disease}, {year}, {eligibility}: "
              "only one outcome class.")
        continue

    base_ap = average_precision_score(y_all, p_base_all)
    model_ap = average_precision_score(y_all, p_model_all)
    observed_delta = model_ap - base_ap

    boot_delta = []
    attempts = 0
    max_attempts = N_BOOT * 3

    while len(boot_delta) < N_BOOT and attempts < max_attempts:
        attempts += 1

        # Resample whole districts with replacement.
        sampled = rng.choice(
            districts, size=n_districts, replace=True
        )

        # Repeated district draws are repeated clusters, not deduplicated.
        y_parts = []
        base_parts = []
        model_parts = []

        for d in sampled:
            b = blocks[d]
            y_parts.append(b["y"].to_numpy())
            base_parts.append(b[BASE].to_numpy())
            model_parts.append(b[MODEL].to_numpy())

        y = np.concatenate(y_parts)
        p_base = np.concatenate(base_parts)
        p_model = np.concatenate(model_parts)

        # PR-AUC is undefined as a useful ranking measure if
        # the resample contains only positives or only negatives.
        if np.unique(y).size < 2:
            continue

        ap_base = average_precision_score(y, p_base)
        ap_model = average_precision_score(y, p_model)
        boot_delta.append(ap_model - ap_base)

    if len(boot_delta) < N_BOOT:
        print(f"WARNING: {disease}, {year}, {eligibility}: "
              f"only {len(boot_delta)} valid replicates.")

    boot_delta = np.asarray(boot_delta)

    lo, hi = np.quantile(boot_delta, [0.025, 0.975])

    results.append({
        "disease": disease,
        "test_year": year,
        "eligibility": eligibility,
        "districts": n_districts,
        "positives": int(g["y"].sum()),
        "base_rate": g["y"].mean(),
        "baseline_PR_AUC": base_ap,
        "model_PR_AUC": model_ap,
        "observed_delta": observed_delta,
        "bootstrap_CI_2.5%": lo,
        "bootstrap_CI_97.5%": hi,
        "bootstrap_median_delta": np.median(boot_delta),
        "bootstrap_fraction_delta_positive": np.mean(boot_delta > 0),
        "valid_bootstrap_replicates": len(boot_delta),
    })

    print(
        f"{disease} | {year} | {eligibility}: "
        f"delta={observed_delta:.4f}, "
        f"95% CI [{lo:.4f}, {hi:.4f}], "
        f"P*(delta>0)={np.mean(boot_delta > 0):.3f}, "
        f"districts={n_districts}"
    )

result_df = pd.DataFrame(results)
result_df.to_csv(OUT / "district_bootstrap_by_year.csv", index=False)

print("\nSaved:", OUT / "district_bootstrap_by_year.csv")
print("\nInterpretation:")
print("- CI entirely above zero: evidence of a positive difference under this bootstrap.")
print("- CI includes zero: uncertainty remains about the direction of the difference.")
print("- These intervals are conditional on the fitted models and evaluated years.")
