from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

INPUT = Path(
    "research_outputs/bootstrap/occurrence_bootstrap_predictions.csv"
)
OUT = Path("research_outputs/bootstrap")
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
SEED = 2026

BASE = "persistence (own 12-month count)"
MODEL = "Season + state + own history"

df = pd.read_csv(INPUT)

required = [
    "d", "t", "y", BASE, MODEL,
    "disease", "test_year", "eligibility", "state_ut"
]
missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")
if df[required].isna().any().any():
    raise ValueError("Missing values found in required columns.")

# Critical mapping check: each district must belong to exactly one state.
if df.groupby("d")["state_ut"].nunique().max() != 1:
    raise ValueError("A district ID maps to multiple states.")

rng = np.random.default_rng(SEED)
results = []

groups = df.groupby(
    ["disease", "test_year", "eligibility"], sort=False
)

for (disease, year, eligibility), g in groups:
    states = np.sort(g["state_ut"].unique())
    n_states = len(states)
    n_districts = g["d"].nunique()

    # Preserve every district-month row within each state cluster.
    blocks = {
        state: g[g["state_ut"] == state].sort_values(["d", "t"])
        for state in states
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
    max_attempts = N_BOOT * 5

    while len(boot_delta) < N_BOOT and attempts < max_attempts:
        attempts += 1

        # Sample states/UTs with replacement.
        sampled_states = rng.choice(
            states, size=n_states, replace=True
        )

        # Repeated state draws are retained as repeated clusters.
        y_parts = []
        base_parts = []
        model_parts = []

        for state in sampled_states:
            b = blocks[state]
            y_parts.append(b["y"].to_numpy())
            base_parts.append(b[BASE].to_numpy())
            model_parts.append(b[MODEL].to_numpy())

        y = np.concatenate(y_parts)
        p_base = np.concatenate(base_parts)
        p_model = np.concatenate(model_parts)

        if np.unique(y).size < 2:
            continue

        ap_base = average_precision_score(y, p_base)
        ap_model = average_precision_score(y, p_model)
        boot_delta.append(ap_model - ap_base)

    if len(boot_delta) < N_BOOT:
        print(f"WARNING: {disease}, {year}, {eligibility}: "
              f"only {len(boot_delta)} valid replicates.")

    if not boot_delta:
        continue

    boot_delta = np.asarray(boot_delta)
    lo, hi = np.quantile(boot_delta, [0.025, 0.975])

    results.append({
        "disease": disease,
        "test_year": year,
        "eligibility": eligibility,
        "states": n_states,
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
        f"95% state-bootstrap CI [{lo:.4f}, {hi:.4f}], "
        f"P*(delta>0)={np.mean(boot_delta > 0):.3f}, "
        f"states={n_states}, districts={n_districts}"
    )

result_df = pd.DataFrame(results)
output = OUT / "state_bootstrap_by_year.csv"
result_df.to_csv(output, index=False)

print(f"\nSaved: {output}")
print("\nInterpretation:")
print("- CI entirely above zero: positive difference supported under this bootstrap.")
print("- CI includes zero: uncertainty remains about the direction of the difference.")
print("- These intervals are conditional on the fitted models and evaluated years.")
print("- Only 36 state/UT clusters exist overall; interpret intervals cautiously.")
