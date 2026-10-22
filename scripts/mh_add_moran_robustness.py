import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran

DATA = "research_outputs/mh_add_spatial_raw.csv"
BOUNDARIES = "data/boundaries/maharashtra_districts.geojson"

df = pd.read_csv(DATA)
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["district", "Cases"]).copy()
df["district"] = df["district"].replace({"Sholapur": "Solapur"})

summary = (
    df.groupby("district")
    .agg(
        median_cases=("Cases", "median"),
        observations=("Cases", "size")
    )
    .reset_index()
)

gdf = gpd.read_file(BOUNDARIES)
gdf["district"] = gdf["district"].str.strip()
summary["district"] = summary["district"].str.strip()

for minimum in [1, 5, 8, 10, 15]:
    subset = summary[summary["observations"] >= minimum].copy()
    joined = gdf.merge(subset, on="district", how="inner")
    joined = joined.sort_values("district").reset_index(drop=True)

    print(f"\nMinimum observations per district: {minimum}")
    print("Districts:", len(joined))

    if len(joined) < 3 or joined["median_cases"].nunique() < 2:
        print("Insufficient data for Moran's I.")
        continue

    w = Queen.from_dataframe(joined, use_index=False)
    print("Islands:", w.islands)

    # Moran's I requires a defined spatial-neighbor structure.
    if w.islands:
        print("Skipping: disconnected districts require a separate decision.")
        continue

    w.transform = "R"
    np.random.seed(42)
    result = Moran(
        joined["median_cases"].to_numpy(),
        w,
        permutations=9999
    )

    print(f"Moran's I: {result.I:.6f}")
    print(f"Expected I: {result.EI:.6f}")
    print(f"Permutation p-value: {result.p_sim:.6f}")

print("\nNote: each threshold changes the set of districts.")
print("These are sensitivity checks, not independent confirmations.")
