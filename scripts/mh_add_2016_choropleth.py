import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

DATA_PATH = OUT / "mh_add_district_date.csv"
BOUNDARY_PATH = Path("data/boundaries/maharashtra_districts.geojson")

def normalize(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = re.sub(r"\bdistrict\b", "", value)
    return re.sub(r"[^a-z0-9]", "", value)

# Load district-date records and select 2016
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["district", "date", "Cases"]).copy()

df = df[df["date"].dt.year == 2016].copy()
df["district"] = df["district"].replace({"Sholapur": "Solapur"})

summary = (
    df.groupby("district", as_index=False)
      .agg(
          median_cases=("Cases", "median"),
          observations=("Cases", "size"),
      )
)

summary["district_key"] = summary["district"].map(normalize)

# Load boundaries
polygons = gpd.read_file(BOUNDARY_PATH)
polygons["district_key"] = polygons["district"].map(normalize)

unmatched = sorted(
    set(summary["district_key"]) - set(polygons["district_key"])
)
if unmatched:
    raise ValueError(f"Unmatched districts: {unmatched}")

gdf = polygons.merge(
    summary[["district_key", "median_cases", "observations"]],
    on="district_key",
    how="left",
)

gdf["has_data"] = gdf["median_cases"].notna()

# Export the exact data behind the map
gdf[["district", "median_cases", "observations", "has_data"]].to_csv(
    OUT / "mh_add_2016_choropleth_data.csv",
    index=False,
)

fig, ax = plt.subplots(figsize=(10, 11))

gdf[gdf["has_data"]].plot(
    column="median_cases",
    cmap="YlOrRd",
    legend=True,
    edgecolor="black",
    linewidth=0.5,
    legend_kwds={"label": "Median recorded ADD cases", "shrink": 0.7},
    ax=ax,
)

gdf[~gdf["has_data"]].plot(
    color="lightgrey",
    edgecolor="black",
    linewidth=0.5,
    ax=ax,
)

ax.set_title(
    "Maharashtra Acute Diarrhoeal Disease\n"
    "District-wise Median Recorded Cases, 2016"
)
ax.text(
    0.02, 0.02,
    "Grey = no recorded observation in 2016\n"
    "District summaries are based on sparse reporting",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="bottom",
    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
)
ax.set_axis_off()

fig.tight_layout()
map_path = OUT / "mh_add_2016_choropleth.png"
fig.savefig(map_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Districts in boundary file:", len(gdf))
print("Districts with observations:", int(gdf["has_data"].sum()))
print("Districts without observations:", int((~gdf["has_data"]).sum()))
print("\nSaved:")
print(" ", map_path)
print(" ", OUT / "mh_add_2016_choropleth_data.csv")
