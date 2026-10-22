import geopandas as gpd
import pandas as pd
import re
import unicodedata

data_path = "research_outputs/mh_add_spatial_raw.csv"
boundary_path = "data/boundaries/maharashtra_districts.geojson"

data = pd.read_csv(data_path)
boundaries = gpd.read_file(boundary_path)

def normalize(name):
    name = unicodedata.normalize("NFKD", str(name))
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = re.sub(r"\bdistrict\b", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name

data_names = sorted(data["district"].dropna().unique())
boundary_names = sorted(boundaries["district"].dropna().unique())

data_lookup = {normalize(n): n for n in data_names}
boundary_lookup = {normalize(n): n for n in boundary_names}

matched = sorted(set(data_lookup) & set(boundary_lookup))
data_only = sorted(set(data_lookup) - set(boundary_lookup))
boundary_only = sorted(set(boundary_lookup) - set(data_lookup))

print("Dataset district count:", len(data_names))
print("Boundary district count:", len(boundary_names))
print("Matched district count:", len(matched))

print("\nMATCHED DISTRICTS:")
for key in matched:
    print(f"{data_lookup[key]}  <->  {boundary_lookup[key]}")

print("\nDATASET DISTRICTS WITHOUT A MATCH:")
for key in data_only:
    print(data_lookup[key])

print("\nBOUNDARY DISTRICTS WITHOUT A MATCH:")
for key in boundary_only:
    print(boundary_lookup[key])
