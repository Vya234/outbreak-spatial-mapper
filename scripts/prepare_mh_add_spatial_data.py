import pandas as pd
from pathlib import Path

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("data/Final_data.csv", low_memory=False)

# Correct disease label
df["state_clean"] = df["state_ut"].astype(str).str.strip()
df["disease_clean"] = df["Disease"].astype(str).str.strip().str.lower()

subset = df[
    df["state_clean"].str.lower().eq("maharashtra")
    & df["disease_clean"].eq("acute diarrhoeal disease")
].copy()

for col in ["Latitude", "Longitude", "Cases"]:
    subset[col] = pd.to_numeric(subset[col], errors="coerce")

print("Raw Maharashtra ADD rows:", len(subset))
print("Distinct districts:", subset["district"].nunique())
print("Missing coordinates:", subset[["Latitude", "Longitude"]].isna().any(axis=1).sum())

# Examine whether each district has stable coordinates.
coord_check = (
    subset.groupby("district")
    .agg(
        observations=("district", "size"),
        latitude_values=("Latitude", "nunique"),
        longitude_values=("Longitude", "nunique"),
        latitude=("Latitude", "median"),
        longitude=("Longitude", "median"),
    )
    .reset_index()
)

print("\nDistricts with changing latitude:",
      int((coord_check["latitude_values"] > 1).sum()))
print("Districts with changing longitude:",
      int((coord_check["longitude_values"] > 1).sum()))

print("\nCoordinate summary by district:")
print(coord_check.sort_values("observations", ascending=False).head(20).to_string(index=False))

# Preserve original coordinates and values for later analysis.
subset.to_csv(OUT / "mh_add_spatial_raw.csv", index=False)
coord_check.to_csv(OUT / "mh_add_district_coordinates.csv", index=False)

print("\nSaved:")
print(OUT / "mh_add_spatial_raw.csv")
print(OUT / "mh_add_district_coordinates.csv")
