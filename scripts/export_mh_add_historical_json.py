import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

OUT = Path("research_outputs")
source = OUT / "mh_add_district_date.csv"
destination = Path("mh_add_historical.json")

def normalize(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = re.sub(r"\bdistrict\b", "", value)
    return re.sub(r"[^a-z0-9]", "", value)

df = pd.read_csv(source)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["district", "date", "Cases"]).copy()
df["district"] = df["district"].replace({"Sholapur": "Solapur"})
df["year"] = df["date"].dt.year

summary = (
    df.groupby(["district", "year"], as_index=False)
      .agg(
          median_cases=("Cases", "median"),
          observations=("Cases", "size"),
      )
)

records = {}
for row in summary.itertuples(index=False):
    key = normalize(row.district)
    records.setdefault(key, {})[str(int(row.year))] = {
        "district": row.district,
        "median_cases": float(row.median_cases),
        "observations": int(row.observations),
    }

payload = {
    "disease": "Acute Diarrhoeal Disease",
    "state": "Maharashtra",
    "aggregation": "District-year median of recorded district-date case counts",
    "missing_means": "No recorded observations; not zero cases",
    "districts": records,
}

destination.write_text(json.dumps(payload, separators=(",", ":")))
print("Saved:", destination)
print("Districts with data:", len(records))
print("Years covered:", sorted({
    year
    for district in records.values()
    for year in district
}))
print("File size:", destination.stat().st_size, "bytes")
