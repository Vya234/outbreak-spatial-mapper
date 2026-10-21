import pandas as pd
from dataset_loader import OutbreakDatasetLoader

def test_real_data_pipeline():
    loader = OutbreakDatasetLoader(data_dir="data")
    try:
        df = loader.load_dataset("processed_dengue_mh.csv")
        print(f"Successfully loaded real dataset with {len(df)} records!")
        print(f"Timeline spans from Day {df['Timeline_Day'].min()} to Day {df['Timeline_Day'].max()}")
        print(f"Geographic spread bounds:")
        print(f"  Latitude range: {df['Latitude'].min()} to {df['Latitude'].max()}")
        print(f"  Longitude range: {df['Longitude'].min()} to {df['Longitude'].max()}")
    except Exception as e:
        print(f"Pipeline validation failed: {e}")

if __name__ == "__main__":
    test_real_data_pipeline()