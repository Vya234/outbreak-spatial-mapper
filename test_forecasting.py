import pandas as pd
import numpy as np
from forecasting_engine import optimize_sir_parameters
from dataset_loader import OutbreakDatasetLoader

def test_sir_optimization():
    loader = OutbreakDatasetLoader(data_dir="data")
    df = loader.load_dataset("processed_dengue_mh.csv")
    
    daily_aggregated = df.groupby('Timeline_Day').size().values
    print("Aggregated daily points over", len(daily_aggregated), "days.")
    
    # Run optimizer on the first 30 days of the real outbreak curve
    sample_curve = daily_aggregated[:30]
    result = optimize_sir_parameters(sample_curve, initial_population=50000)
    print("Optimization Results on Real EpiClim Data:", result)

if __name__ == "__main__":
    test_sir_optimization()