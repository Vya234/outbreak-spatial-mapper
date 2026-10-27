
import pandas as pd
from pathlib import Path

SOURCE = Path("data/Final_data.csv")
OUTPUT = Path("data_corrected/Final_data_standardized.csv")
AUDIT = Path("research_outputs/district_name_standardization_audit.csv")

df = pd.read_csv(SOURCE)
df.columns = df.columns.str.strip()

if "district" not in df.columns:
    raise ValueError("The source file has no district column.")

original = df["district"].astype("string").str.strip()
normalized = original.str.lower()

# Keep a record of the changes without modifying the source file.
changed = normalized.eq("sholapur").fillna(False)

audit = pd.DataFrame({
    "original_district": original[changed],
    "standardized_district": "Solapur",
})

df["district"] = original
df.loc[changed, "district"] = "Solapur"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
AUDIT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT, index=False)
audit.to_csv(AUDIT, index=False)

print("Original source:", SOURCE)
print("Standardized copy:", OUTPUT)
print("Audit log:", AUDIT)
print("Rows standardized:", int(changed.sum()))
print("\nDistrict counts after standardization:")
print(
    df["district"].value_counts()
      .reindex(["Sholapur", "Solapur"])
      .dropna()
      .to_string()
)
print("\nOriginal source file was not modified.")
