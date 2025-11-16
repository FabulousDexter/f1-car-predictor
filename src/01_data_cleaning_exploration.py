# Load your data
import pandas as pd

def clean_laps_data(df_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the laps dataset.
    """
    df_laps_clean = df_laps.copy()
    
    # Convert LapTime to timedelta
    df_laps_clean['LapTime'] = pd.to_timedelta(df_laps['LapTime'], errors='coerce')

    # Convert LapNumber and TyreLife to integers
    df_laps_clean['LapNumber'] = pd.to_numeric(df_laps_clean['LapNumber'], errors='coerce').astype('Int64')
    df_laps_clean['TyreLife'] = pd.to_numeric(df_laps_clean['TyreLife'], errors='coerce').astype('Int64') 
    
    return df_laps_clean
    
def clean_results_data(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Clean  race results dataset.
    """
    df_results_clean = df_results.copy()
    
    # Convert Time columns to appropriate formats 
    race_results_time_cols = ['Q1', 'Q2', 'Q3', 'Time']
    for col in race_results_time_cols:
        df_results_clean[col] = pd.to_timedelta(df_results[col], errors='coerce')
        
    # Drop Missing Team Names in Race Results
    df_results_clean = df_results_clean.dropna(subset=['TeamName'])

    # Convert Position, GridPosition, and Points to integers
    df_results_clean['Position'] = pd.to_numeric(df_results_clean['Position'], errors='coerce').astype('Int64')
    df_results_clean['GridPosition'] = pd.to_numeric(df_results_clean['GridPosition'], errors='coerce').astype('Int64')
    df_results_clean['Points'] = pd.to_numeric(df_results_clean['Points'], errors='coerce').astype('Int64')

    return df_results_clean

def print_cleaning_summary(df: pd.DataFrame, name: str):
        # Summarize cleaning results
    print(f"=== CLEANED {name} DATASET ===")
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nThe first 5 rows:")
    print(df.head(5))

if __name__ == "__main__":
    # Load raw datasets
    df_laps = pd.read_csv(f"data/f1_all_session_race_laps.csv")
    df_results = pd.read_csv(f"data/f1_all_session_race_results.csv")
    
    # Clean datasets
    df_laps_clean = clean_laps_data(df_laps)
    df_results_clean = clean_results_data(df_results)
    
    # Save cleaned datasets
    df_laps_clean.to_csv("data/f1_all_session_race_laps_clean.csv", index=False)
    df_results_clean.to_csv("data/f1_all_session_race_results_clean.csv", index=False)
    
    print("Data cleaning completed and cleaned files saved.")
    
    print_cleaning_summary(df_laps_clean, "LAPS")
    print_cleaning_summary(df_results_clean, "RACES")
