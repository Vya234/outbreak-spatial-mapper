import re
import unicodedata
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import Queen
from esda.moran import Moran

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = "research_outputs/mh_add_spatial_raw.csv"
BOUNDARY_PATH = "data/boundaries/maharashtra_districts.geojson"
OUT_PATH = "research_outputs/mh_add_moran_summary_robustness.csv"


def normalize(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = re.sub(r"\bdistrict\b", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


# Load and clean records
data = pd.read_csv(DATA_PATH)
data["Cases"] = pd.to_numeric(data["Cases"], errors="coerce")
data = data.dropna(subset=["district", "Cases"]).copy()
data["district"] = data["district"].replace({"Sholapur": "Solapur"})

# Compare alternative summaries of the same reported case records
summary = data.groupby("district").agg(
    median_cases=("Cases", "median"),
    mean_cases=("Cases", "mean"),
    max_cases=("Cases", "max"),
    observations=("Cases", "size"),
).reset_index()

polygons = gpd.read_file(BOUNDARY_PATH)
summary["district_key"] = summary["district"].map(normalize)
polygons["district_key"] = polygons["district"].map(normalize)

unmatched = sorted(
    set(summary["district_key"]) - set(polygons["district_key"])
)
if unmatched:
    raise ValueError(f"Unmatched districts: {unmatched}")

gdf = polygons.merge(
    summary.drop(columns="district"),
    on="district_key",
    how="inner",
).sort_values("district_key").reset_index(drop=True)

# Queen contiguity, matching the original analysis
w = Queen.from_dataframe(gdf, use_index=False)
w.transform = "R"

print("Districts included:", len(gdf))
print("Disconnected districts (islands):", w.islands)
print("\nGLOBAL MORAN'S I BY DISTRICT SUMMARY")
print("-" * 66)

results = []

for variable in ["median_cases", "mean_cases", "max_cases"]:
    y = gdf[variable].to_numpy(dtype=float)

    np.random.seed(42)
    moran = Moran(y, w, permutations=9999)

    row = {
        "summary_statistic": variable,
        "districts": len(y),
        "moran_I": moran.I,
        "expected_I": moran.EI,
        "permutation_p": moran.p_sim,
        "z_score": moran.z_sim,
    }
    results.append(row)

    print(f"\nStatistic: {variable}")
    print(f"  Moran's I:       {moran.I:.6f}")
    print(f"  Expected I:      {moran.EI:.6f}")
    print(f"  Permutation p:   {moran.p_sim:.6f}")
    print(f"  Permutation z:   {moran.z_sim:.6f}")

pd.DataFrame(results).to_csv(OUT_PATH, index=False)
print(f"\nSaved results to {OUT_PATH}")
