import os
import re
import unicodedata
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = "research_outputs/mh_add_spatial_raw.csv"
BOUNDARY_PATH = "data/boundaries/maharashtra_districts.geojson"
OUT_DIR = "research_outputs"

os.makedirs(OUT_DIR, exist_ok=True)

def normalize(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = re.sub(r"\bdistrict\b", "", value)
    return re.sub(r"[^a-z0-9]", "", value)

# 1. Load data and boundaries
data = pd.read_csv(DATA_PATH)
polygons = gpd.read_file(BOUNDARY_PATH)

required = {"district", "Cases"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Dataset missing columns: {missing}")

data["Cases"] = pd.to_numeric(data["Cases"], errors="coerce")
data = data.dropna(subset=["district", "Cases"]).copy()

# Normalize known spelling variant
data["district"] = data["district"].replace({"Sholapur": "Solapur"})

# 2. One observation per district: median reported cases
summary = (
    data.groupby("district", as_index=False)
    .agg(
        median_cases=("Cases", "median"),
        mean_cases=("Cases", "mean"),
        total_recorded_cases=("Cases", "sum"),
        observations=("Cases", "size"),
    )
)
summary["district_key"] = summary["district"].map(normalize)
polygons["district_key"] = polygons["district"].map(normalize)

# Stop if any dataset district cannot be joined
unmatched = sorted(
    set(summary["district_key"]) - set(polygons["district_key"])
)
if unmatched:
    raise ValueError(f"Unmatched dataset districts: {unmatched}")

# Retain districts represented in the dataset
gdf = polygons.merge(summary, on="district_key", how="inner")
gdf = gdf.drop(columns=["district_y"], errors="ignore")

# The original polygon name is in district_x after the merge
if "district_x" in gdf.columns:
    gdf = gdf.rename(columns={"district_x": "boundary_district"})
elif "district" in gdf.columns:
    gdf = gdf.rename(columns={"district": "boundary_district"})

gdf["district"] = gdf["district_key"].map(
    summary.set_index("district_key")["district"]
)

gdf = gdf.sort_values("district").reset_index(drop=True)

print("Dataset districts:", summary["district"].nunique())
print("Matched polygons:", len(gdf))
print("Districts excluded from boundary file:", len(polygons) - len(gdf))
print("Districts:", ", ".join(gdf["district"].tolist()))

# 3. Queen-contiguity weights
w = Queen.from_dataframe(gdf, use_index=False)
print("\nQueen weights:")
print("Districts in graph:", w.n)
print("Disconnected districts (islands):", w.islands)
print("Components:", len(w.component_labels))
print("Neighbors per district:")
for i, name in enumerate(gdf["district"]):
    print(f"  {name}: {len(w.neighbors[i])}")

# Moran's I requires variation in the variable
y = gdf["median_cases"].to_numpy(dtype=float)
if len(y) < 3 or np.allclose(y, y[0]):
    raise ValueError("Insufficient variation or districts for Moran's I.")

# 4. Global Moran's I, 9,999 permutations
np.random.seed(42)
w.transform = "R"
global_moran = Moran(y, w, permutations=9999)

print("\nGLOBAL MORAN'S I")
print(f"I: {global_moran.I:.6f}")
print(f"Expected I under null: {global_moran.EI:.6f}")
print(f"Permutation p-value: {global_moran.p_sim:.6f}")
print(f"Permutation z-score: {global_moran.z_sim:.6f}")

# 5. Local Moran's I / LISA
local = Moran_Local(y, w, permutations=9999)

gdf["local_moran_I"] = local.Is
gdf["local_p_sim"] = local.p_sim
gdf["spatial_lag_z"] = local.z_sim
gdf["significant_05"] = gdf["local_p_sim"] < 0.05

# Quadrant codes: 1 HH, 2 LH, 3 LL, 4 HL
labels = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
gdf["lisa_cluster"] = [
    labels.get(int(q), "Not classified") if p < 0.05 else "Not significant"
    for q, p in zip(local.q, local.p_sim)
]

# 6. Save results
columns = [
    "district", "median_cases", "mean_cases", "total_recorded_cases",
    "observations", "local_moran_I", "local_p_sim",
    "significant_05", "lisa_cluster"
]
gdf[columns].to_csv(f"{OUT_DIR}/mh_add_lisa_results.csv", index=False)
gdf.drop(columns="geometry").to_csv(
    f"{OUT_DIR}/mh_add_spatial_joined.csv", index=False
)

# 7. Plot LISA clusters
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(
    column="lisa_cluster",
    categorical=True,
    legend=True,
    edgecolor="black",
    linewidth=0.4,
    ax=ax,
    missing_kwds={"color": "lightgrey"},
)
ax.set_title(
    "Maharashtra Acute Diarrhoeal Disease\n"
    "Exploratory LISA clusters of district median reported cases"
)
ax.set_axis_off()
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/mh_add_lisa_map.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nLISA cluster counts:")
print(gdf["lisa_cluster"].value_counts().to_string())
print("\nSaved:")
print(f"  {OUT_DIR}/mh_add_lisa_results.csv")
print(f"  {OUT_DIR}/mh_add_spatial_joined.csv")
print(f"  {OUT_DIR}/mh_add_lisa_map.png")
