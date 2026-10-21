import pandas as pd
import numpy as np
import os

class EpiClimAdapter:
    """
    Adapts the EpiClim Final_data.csv dataset into the schema required by 
    the OutbreakIQ spatial-temporal pipeline (Point_ID, Latitude, Longitude, Timeline_Day).
    """
    def __init__(self, filepath: str = "data/Final_data.csv"):
        if not os.path.exists(filepath) and os.path.exists("Final_data.csv"):
            filepath = "Final_data.csv"
        self.filepath = filepath
        if os.path.exists(filepath):
            self.df = pd.read_csv(filepath)
        else:
            self.df = None

    def filter_outbreak(self, state: str = None, district: str = None, disease: str = None) -> pd.DataFrame:
        if self.df is None:
            raise FileNotFoundError("Final_data.csv dataset not found in data/ directory.")
            
        filtered = self.df.copy()
        if state:
            filtered = filtered[filtered['state_ut'].str.contains(state, case=False, na=False)]
        if district:
            filtered = filtered[filtered['district'].str.contains(district, case=False, na=False)]
        if disease:
            filtered = filtered[filtered['Disease'].str.contains(disease, case=False, na=False)]
            
        return filtered

    def export_to_pipeline_format(self, state: str = "Maharashtra", disease: str = "Dengue", output_filename: str = "processed_dengue_mh.csv") -> str:
        """
        Extracts multi-week records for a specific region/disease and expands them 
        into case-point coordinates across a chronological Timeline_Day sequence.
        """
        subset = self.filter_outbreak(state=state, disease=disease)
        if subset.empty:
            print(f"No records found for {disease} in {state}. Returning default sample sequence.")
            subset = self.df.head(20)
            
        subset = subset.sort_values(by=['year', 'mon', 'day']).reset_index(drop=True)
        
        records = []
        point_id = 1
        
        for idx, row in subset.iterrows():
            timeline_day = idx + 1
            
            try:
                val = row['Cases']
                cases = int(val) if pd.notnull(val) and str(val).isdigit() else 1
            except Exception:
                cases = 1
                
            base_lat = float(row['Latitude']) if pd.notnull(row['Latitude']) else 19.0760
            base_lon = float(row['Longitude']) if pd.notnull(row['Longitude']) else 72.8777
            
            for i in range(min(cases, 50)): # Cap cluster expansion per record for performance
                lat = base_lat + np.random.normal(0, 0.02)
                lon = base_lon + np.random.normal(0, 0.02)
                records.append({
                    "Point_ID": point_id,
                    "Latitude": round(lat, 5),
                    "Longitude": round(lon, 5),
                    "Timeline_Day": timeline_day
                })
                point_id += 1
                
        out_df = pd.DataFrame(records)
        output_path = os.path.join("data", output_filename) if os.path.exists("data") else output_filename
        out_df.to_csv(output_path, index=False)
        print(f"Successfully exported {len(out_df)} coordinate points to {output_path}")
        return output_path

if __name__ == "__main__":
    adapter = EpiClimAdapter()
    if adapter.df is not None:
        adapter.export_to_pipeline_format(state="Maharashtra", disease="Dengue", output_filename="processed_dengue_mh.csv")