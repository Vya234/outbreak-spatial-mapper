import pandas as pd
from pathlib import Path

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("data/Final_data.csv")
df = df[
    df["state_ut"].astype(str).str.strip().str.lower().eq("maharashtra")
    & df["Disease"].astype(str).str.strip().eq("Dengue")
].copy()

for col in ["Cases", "Temp", "preci", "LAI", "year", "mon", "day"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["date"] = pd.to_datetime(
    dict(year=df["year"], month=df["mon"], day=df["day"]),
    errors="coerce",
)

# Resolve repeated observations for each district-date.
df = (
    df.groupby(["district", "date"], as_index=False)
    .agg(
        Cases=("Cases", "median"),
        Temp=("Temp", "median"),
        preci=("preci", "median"),
        LAI=("LAI", "median"),
    )
)

results = []

for district, group in df.groupby("district"):
    for climate in ["Temp", "preci", "LAI"]:
        pair = group[["Cases", climate]].dropna()
        rho = (
            pair["Cases"].corr(pair[climate], method="spearman")
            if len(pair) >= 8
            and pair["Cases"].nunique() > 1
            and pair[climate].nunique() > 1
            else float("nan")
        )
        results.append({
            "district": district,
            "climate_variable": climate,
            "n_observations": len(pair),
            "spearman_rho": rho,
            "eligible_min_8": len(pair) >= 8,
        })

res = pd.DataFrame(results)
res.to_csv(OUT / "mh_dengue_district_correlations.csv", index=False)

print("=== DISTRICT-LEVEL CORRELATIONS ===")
for climate in ["Temp", "preci", "LAI"]:
    subset = res[res["climate_variable"] == climate]
    valid = subset["spearman_rho"].dropna()
    print(f"\n{climate}:")
    print("Districts total:", len(subset))
    print("Districts with calculable correlation:", len(valid))
    print("Positive correlations:", int((valid > 0).sum()))
    print("Negative correlations:", int((valid < 0).sum()))
    if len(valid):
        print("Median district correlation:", round(valid.median(), 3))
        print("Range:", round(valid.min(), 3), "to", round(valid.max(), 3))

print("\nSaved:", OUT / "mh_dengue_district_correlations.csv")
print("Note: correlations are exploratory; small samples and multiple testing matter.")
