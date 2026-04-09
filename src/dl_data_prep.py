import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
import os

class F1Dataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

def prepare_dl_data(df_path='data/processed/f1_driver_race.csv', output_dir='data/dl_processed'):
    df = pd.read_csv(df_path)
    
    # Sort by date
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Cumulative stats (shift by 1 to avoid leakage)
    df['past_avg_pos'] = df.groupby('driver')['finish_position'].transform(lambda x: x.shift().expanding().mean())
    df['past_avg_points'] = df.groupby('driver')['points'].transform(lambda x: x.shift().expanding().mean())
    df['team_avg_points'] = df.groupby('team')['points'].transform(lambda x: x.shift().expanding().mean())
    
    # Fill missing values
    df = df.fillna(0) # Simple 0 fill for early races
    
    # Fill any remaining NaNs in sector times with mean (though fillna(0) already caught them, 
    # but keeping explicit structure if we want to change strategy)
    sector_cols = ['avg_sector1', 'avg_sector2', 'avg_sector3']
    for col in sector_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Define features
    num_features = [
        'grid_position', 'past_avg_pos', 'past_avg_points', 
        'team_avg_points', 'avg_sector1', 'avg_sector2', 
        'avg_sector3', 'avg_air_temp', 'avg_track_temp', 
        'avg_humidity', 'rain_probability', 'is_classified'
    ]
    cat_features = ['driver', 'team', 'race_name', 'main_compound']
    target = 'finish_position'

    # Filter out drivers with very few races if necessary (optional here)
    
    # Encode Categorical
    cat_encoders = {}
    X_cat = []
    for col in cat_features:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col + '_idx'] = le.fit_transform(df[col])
        cat_encoders[col] = le
        X_cat.append(df[col + '_idx'].values)
    
    X_cat = np.stack(X_cat, axis=1)
    
    # Split by time (2018-2023 Train, 2024-2025 Test)
    train_df = df[df['season'] <= 2023].copy()
    test_df = df[df['season'] >= 2024].copy()
    
    # Scale Numerical
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(train_df[num_features])
    X_num_test = scaler.transform(test_df[num_features])
    
    X_cat_train = X_cat[df['season'] <= 2023]
    X_cat_test = X_cat[df['season'] >= 2024]
    
    y_train = train_df[target].values
    y_test = test_df[target].values
    
    # Save objects
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    joblib.dump(cat_encoders, os.path.join(output_dir, 'cat_encoders.joblib'))
    
    # Meta information for embeddings
    embedding_info = {col: len(le.classes_) for col, le in cat_encoders.items()}
    joblib.dump(embedding_info, os.path.join(output_dir, 'embedding_info.joblib'))
    
    return (X_num_train, X_cat_train, y_train), (X_num_test, X_cat_test, y_test), embedding_info

if __name__ == "__main__":
    (X_num_tr, X_cat_tr, y_tr), (X_num_ts, X_cat_ts, y_ts), info = prepare_dl_data()
    print(f"Data prepared. Train samples: {len(y_tr)}, Test samples: {len(y_ts)}")
    print(f"Embedding info: {info}")
