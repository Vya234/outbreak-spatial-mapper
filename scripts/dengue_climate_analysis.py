import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv("data/Final_data.csv")

df = df[
    df["state_ut"].astype(str).str.strip().str.lower().eq("maharashtra")
    & df["Disease"].astype(str).str.strip().str.lower().eq("dengue")
].copy()

df["Date"] = pd.to_datetime(
    dict(
        year=pd.to_numeric(df["year"], errors="coerce"),
        month=pd.to_numeric(df["mon"], errors="coerce"),
        day=pd.to_numeric(df["day"], errors="coerce"),
    ),
    errors="coerce",
)

for col in ["Cases", "Temp", "preci", "LAI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Exact duplicates are excluded from the analysis copy only.
dedup_cols = [c for c in df.columns if c != "Unnamed: 0"]
df = df.drop_duplicates(subset=dedup_cols).copy()
df["Month"] = df["Date"].dt.month

print("Records:", len(df))
print("Districts:", df["district"].nunique())
print("Date range:", df["Date"].min(), "to", df["Date"].max())

print("\n=== POOLED SPEARMAN CORRELATIONS ===")
pooled = []

for feature in ["Temp", "preci", "LAI"]:
    subset = df[["Cases", feature]].dropna()
    if len(subset) >= 10:
        rho, p = spearmanr(subset["Cases"], subset[feature])
        pooled.append({
            "Feature": feature,
            "N": len(subset),
            "Spearman_rho": rho,
            "p_value": p,
        })

pooled_df = pd.DataFrame(pooled)
print(pooled_df.round(4).to_string(index=False))
pooled_df.to_csv(OUT / "dengue_pooled_climate_correlations.csv", index=False)

print("\n=== WITHIN-DISTRICT CORRELATIONS ===")
rows = []

for district, group in df.groupby("district"):
    for feature in ["Temp", "preci", "LAI"]:
        subset = group[["Cases", feature]].dropna()

        if len(subset) < 8 or subset[feature].nunique() < 2:
            continue

        rho, p = spearmanr(subset["Cases"], subset[feature])
        rows.append({
            "district": district,
            "Feature": feature,
            "N": len(subset),
            "Spearman_rho": rho,
            "p_value": p,
        })

district_df = pd.DataFrame(rows)
if not district_df.empty:
    district_df.to_csv(
        OUT / "dengue_district_climate_correlations.csv",
        index=False
    )

    print("District-feature tests:", len(district_df))
    print("\nStrongest positive associations (exploratory):")
    print(
        district_df.sort_values("Spearman_rho", ascending=False)
        .head(10).round(4).to_string(index=False)
    )

    print("\nStrongest negative associations (exploratory):")
    print(
        district_df.sort_values("Spearman_rho")
        .head(10).round(4).to_string(index=False)
    )

print("\nSaved pooled and district correlation results.")
print(
    "Interpretation caution: these are exploratory associations, not "
    "causal effects or evidence of predictive performance. Sparse "
    "observations, reporting differences, seasonality and multiple "
    "testing may affect the results."
)
