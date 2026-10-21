import pandas as pd
import os

class OutbreakDatasetLoader:
    """
    Manages loading and schema validation for real-world epidemiological datasets 
    (such as HED and EpiClim) to feed into the forecasting and spatial engines.
    """
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir

    def load_dataset(self, filename: str) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset {filename} not found in {self.data_dir}")
        
        df = pd.read_csv(filepath)
        required_columns = {"Point_ID", "Latitude", "Longitude", "Timeline_Day"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Dataset missing required schema columns. Expected: {required_columns}")
            
        return df
        
    def get_available_datasets(self) -> list:
        if not os.path.exists(self.data_dir):
            return []
        return [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

if __name__ == "__main__":
    print("DatasetLoader module ready.")
