import pandas as pd
import numpy as np
import yaml

def create_model_features():
    """
    Create model features from cleaned race results data.
    """
    # --- 1. Load Config and Data ---
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("config.yaml not found.")
        exit()

    try:
        results_df = pd.read_csv(config['data']['raw_results_cleaned'])
    except FileNotFoundError:
        print(f"Error: File not found at {config['data']['raw_results_cleaned']}")
        exit()

    # --- Standardize Team Names ---
    team_name_mapping = {
        'Red Bull': 'Red Bull Racing',          
        'RB': 'RB F1 Team',                
        'Racing Bulls': 'RB F1 Team',           
        'AlphaTauri': 'RB F1 Team',           
        'Toro Rosso': 'RB F1 Team',             
        'Alpine F1 Team': 'Alpine',
        'Kick Sauber': 'Sauber',
        'Alfa Romeo': 'Sauber',
    }
    
    results_df['TeamName'] = results_df['TeamName'].replace(team_name_mapping)

    print("Loaded and pre-processed data successfully.")
    print(f"Total rows loaded: {len(results_df)}")
    print(f"Session types: {results_df['SessionType'].unique()}")

    # --- 2. Create Base Table ---
    base_df = results_df[results_df['SessionType'] == 'R'].copy()
    print(f"Race rows before filtering: {len(base_df)}")
    
    base_df = base_df[['Year', 'RoundNumber', 'DriverNumber', 'FullName', 'TeamName', 'Position', 'GridPosition']]
    
    # Check Position column
    print(f"NaN values in Position: {base_df['Position'].isna().sum()}")
    print(f"Position data type: {base_df['Position'].dtype}")
    
    # Only drop if Position is actually NaN (not just non-numeric strings)
    base_df = base_df[base_df['Position'].notna()]
    
    # Convert Position to int, handling any strings
    try:
        base_df['Position'] = pd.to_numeric(base_df['Position'], errors='coerce')
        base_df = base_df.dropna(subset=['Position'])
        base_df['Position'] = base_df['Position'].astype(int)
    except Exception as e:
        print(f"Error converting Position to int: {e}")
        print(f"Sample Position values: {base_df['Position'].head(20)}")
    
    print(f"Created base table. Shape: {base_df.shape}")

    if len(base_df) == 0:
        print("ERROR: Base table is empty! Check your data.")
        return

    # --- 3. Create Features ---

    # Feature 3b: Average Practice Position
    practice_sessions = ['Practice 1', 'Practice 2', 'Practice 3']
    practice_df = results_df[results_df['SessionType'].isin(practice_sessions)].copy()
    practice_avg_df = practice_df.groupby(['Year', 'RoundNumber', 'DriverNumber'])['Position'].mean().reset_index()
    practice_avg_df.rename(columns={'Position': 'AvgPracticePosition'}, inplace=True)

    # Feature 3c: Championship Points
    print("Calculating championship features...")
    race_results_df = results_df[results_df['SessionType'] == 'R'].copy()
    race_results_df.sort_values(by=['Year', 'RoundNumber'], inplace=True)

    driver_points_map = {}
    team_points_map = {}
    driver_points_list = []
    team_points_list = []

    # Group by race (Year + RoundNumber)
    for (year, round_num), race_group in race_results_df.groupby(['Year', 'RoundNumber']):
        # Initialize year dictionaries if needed
        if year not in driver_points_map:
            driver_points_map[year] = {d: 0 for d in results_df['DriverNumber'].unique()}
            team_points_map[year] = {t: 0 for t in results_df['TeamName'].unique()}
        
        # First pass: record points BEFORE this race for all drivers
        for _, row in race_group.iterrows():
            driver = row['DriverNumber']
            team = row['TeamName']
            
            driver_points_before = driver_points_map[year].get(driver, 0)
            team_points_before = team_points_map[year].get(team, 0)
            
            driver_points_list.append(driver_points_before)
            team_points_list.append(team_points_before)
        
        # Second pass: update totals AFTER recording all "before" values
        for _, row in race_group.iterrows():
            driver = row['DriverNumber']
            team = row['TeamName']
            
            driver_points_map[year][driver] += row['Points']
            team_points_map[year][team] += row['Points']

    race_results_df['driver_points_before_race'] = driver_points_list
    race_results_df['team_points_before_race'] = team_points_list
    print("Finished calculating championship features.")
    
    # --- 4. Assemble Final Dataset ---
    final_df = base_df.copy()

    final_df = final_df.merge(practice_avg_df, on=['Year', 'RoundNumber', 'DriverNumber'], how='left')
    final_df = final_df.merge(
        race_results_df[['Year', 'RoundNumber', 'DriverNumber', 'driver_points_before_race', 'team_points_before_race']],
        on=['Year', 'RoundNumber', 'DriverNumber'],
        how='left'
    )

    # --- 5. Handle Final Missing Values ---
        # Use max grid position + 1 for missing/pit lane starts
    max_grid = final_df['GridPosition'].max()
    final_df['GridPosition'] = final_df['GridPosition'].replace(0.0, max_grid + 1).fillna(max_grid + 1)
    final_df = final_df.drop(columns=['AvgPracticePosition'])

    print(f"Assembled final dataset. Shape: {final_df.shape}")

    # --- 6. Save to CSV ---
    output_path = config['data']['model_features']
    final_df.to_csv(output_path, index=False)

    print(f"Successfully saved model features to {output_path}")
    print("Feature engineering complete.")
    
if __name__ == "__main__":
    create_model_features()