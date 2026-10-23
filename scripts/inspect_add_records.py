import pandas as pd

df = pd.read_csv("data/Final_data.csv")

print("Dataset shape:", df.shape)
print("\nDisease labels:")
print(df["Disease"].value_counts().to_string())

mask = (
    df["state_ut"].astype(str).str.strip().str.lower()
    .isin(["maharashtra", "west bengal"])
    & df["Disease"].astype(str).str.contains(
        "diarr", case=False, na=False
    )
)

print("\nMatching Maharashtra / West Bengal records:")
print(df.loc[mask].to_string(index=False))
print("\nTotal matching rows:", int(mask.sum()))