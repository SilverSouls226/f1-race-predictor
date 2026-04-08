import pandas as pd
import glob
import os
import numpy as np

def load_and_combine_data(data_dir="data/raw"):
    """Loads all season_*.csv files and combines them."""
    all_files = glob.glob(os.path.join(data_dir, "season_*.csv"))
    if not all_files:
        raise FileNotFoundError("No season data files found in 'data/raw' directory.")
    
    df_list = []
    for filename in all_files:
        print(f"Loading {filename}...")
        df = pd.read_csv(filename)
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def clean_data(df):
    """Cleans the raw dataframe."""
    print("Cleaning data...")
    
    # 1. Pit Stops: Convert distinct pit times to a binary 'pitted' flag for the lap
    # If pit_in_time or pit_out_time is present (not NaN), it's a pit lap
    df['pitted'] = (~df['pit_in_time'].isna()) | (~df['pit_out_time'].isna())
    df['pitted'] = df['pitted'].astype(int)
    
    # 2. Handle missing values
    # Fill missing qualifying times with a large number or max? 
    # Or just keep as NaN if using tree models that handle it.
    # For now, let's fill with 0 or a flag, or leave it. 
    # 'grid_position' should be numeric.
    
    # Convert 'lap_time' to numeric if it's not
    df['lap_time'] = pd.to_numeric(df['lap_time'], errors='coerce')
    
    # Sort
    # We need a way to order races. 'event_date' is best.
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values(by=['event_date', 'lap_number'])
    
    return df

def aggregate_to_race_level(df):
    """Aggregates lap-level data to driver-race level."""
    print("Aggregating to driver-race level...")
    
    # Group by Race and Driver
    # We assume 'race_name' and 'season' uniquely identify a race. 
    # Better to use 'event_date' to be safe or (season, race_name).
    
    # Aggregations
    agg_funcs = {
        'grid_position': 'first',
        'finish_position': 'last', # Position at end of race
        'points_scored': 'last',
        'team': 'last',
        'lap_time': ['mean', 'std', 'min'],
        'sector1_time': 'mean',
        'sector2_time': 'mean',
        'sector3_time': 'mean',
        'tyre_compound': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        'pitted': 'sum', # Total pit stops
        'status': 'last', # Finished, Collision, etc.
        'air_temperature': 'mean',
        'track_temperature': 'mean',
        'humidity': 'mean',
        'rainfall': 'mean' # Proportion of rainy laps? 'rainfall' is boolean?
    }
    
    # Check if rainfall is boolean or numeric
    if df['rainfall'].dtype == 'object':
        df['rainfall'] = df['rainfall'].astype(bool).astype(int)
    elif df['rainfall'].dtype == bool:
        df['rainfall'] = df['rainfall'].astype(int)
        
    # Perform aggregation
    race_df = df.groupby(['season', 'race_name', 'event_date', 'driver']).agg(agg_funcs).reset_index()
    
    # Flatten columns
    race_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in race_df.columns.values]
    
    # Rename for clarity
    race_df.rename(columns={
        'lap_time_mean': 'avg_lap_time',
        'lap_time_std': 'std_lap_time',
        'lap_time_min': 'best_lap_time',
        'sector1_time_mean': 'avg_sector1',
        'sector2_time_mean': 'avg_sector2',
        'sector3_time_mean': 'avg_sector3',
        'tyre_compound_<lambda>': 'main_compound',
        'pitted_sum': 'pit_stop_count',
        'grid_position_first': 'grid_position',
        'finish_position_last': 'finish_position',
        'points_scored_last': 'points',
        'team_last': 'team',
        'status_last': 'status',
        'air_temperature_mean': 'avg_air_temp',
        'track_temperature_mean': 'avg_track_temp',
        'humidity_mean': 'avg_humidity',
        'rainfall_mean': 'rain_probability'
    }, inplace=True)
    
    # 3. Reliability: Was the driver classified?
    # Simple heuristic: if 'status' starts with 'Finished' or '+', count as classified.
    def is_classified(status):
        if pd.isna(status): return 0
        status = str(status)
        if status.startswith('Finished') or status.startswith('+'):
            return 1
        return 0
    
    race_df['is_classified'] = race_df['status'].apply(is_classified)
    
    # Calculate position gain
    race_df['position_gain'] = race_df['grid_position'] - race_df['finish_position']
    
    return race_df

def feature_engineering(race_df):
    """Adds engineered features."""
    print("Feature engineering...")
    
    # 1. Driver consistency (std_lap_time is already there)
    
    # 2. Team Power (avg points per race for the team in that season so far? Or just one-hot team)
    # Simple encoded team for now.
    
    return race_df

if __name__ == "__main__":
    combined_df = load_and_combine_data()
    cleaned_df = clean_data(combined_df)
    
    # Save intermediate lap-level data
    cleaned_df.to_csv("data/processed/f1_cleaned_laps.csv", index=False)
    print("Saved data/processed/f1_cleaned_laps.csv")
    
    race_level_df = aggregate_to_race_level(cleaned_df)
    race_level_df = feature_engineering(race_level_df)
    
    # Save aggregated data
    race_level_df.to_csv("data/processed/f1_driver_race.csv", index=False)
    print("Saved data/processed/f1_driver_race.csv")
