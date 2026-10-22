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
    errors="coerce"
)

# Collapse repeated district-date records using the median.
# This prevents duplicate rows from receiving extra weight.
df = (
    df.groupby(["district", "date"], as_index=False)
      .agg(
          Cases=("Cases", "median"),
          Temp=("Temp", "median"),
          preci=("preci", "median"),
          LAI=("LAI", "median"),
      )
)

print("Unique district-date rows:", len(df))
print("Districts:", df["district"].nunique())

print("\n=== SPEARMAN CORRELATIONS WITH RECORDED CASES ===")
results = []

for climate in ["Temp", "preci", "LAI"]:
    pair = df[["Cases", climate]].dropna()
    rho = pair["Cases"].corr(pair[climate], method="spearman")

    results.append({
        "climate_variable": climate,
        "n_observations": len(pair),
        "spearman_rho": rho,
    })

    print(
        f"{climate}: n={len(pair)}, "
        f"Spearman rho={rho:.3f}" if pd.notna(rho)
        else f"{climate}: n={len(pair)}, correlation unavailable"
    )

pd.DataFrame(results).to_csv(
    OUT / "mh_dengue_climate_case_correlations.csv",
    index=False
)

print("\nSaved:", OUT / "mh_dengue_climate_case_correlations.csv")
print(
    "\nCaution: these are pooled exploratory correlations. "
    "They do not control for district, season, reporting coverage, "
    "or time trends and do not establish causation."
)
