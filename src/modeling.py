import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

def load_data(filepath="data/processed/f1_driver_race.csv"):
    """Loads the aggregated driver-race data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
    df = pd.read_csv(filepath)
    return df

def prepare_modeling_data(df):
    """Prepares data for modeling (encoding, splitting)."""

    # Sort by date
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Cumulative stats (shift by 1 to avoid leakage)
    df['past_avg_pos'] = df.groupby('driver')['finish_position'].transform(lambda x: x.shift().expanding().mean())
    df['past_avg_points'] = df.groupby('driver')['points'].transform(lambda x: x.shift().expanding().mean())
    df['team_avg_points'] = df.groupby('team')['points'].transform(lambda x: x.shift().expanding().mean())
    
    # Fill NA for first races
    df = df.fillna(0) # Simplified
    
    # Encode categorical
    # One-hot encoding for Team
    df_encoded = pd.get_dummies(df, columns=['team'], drop_first=True)
    
    # Select features
    features = ['grid_position', 'past_avg_pos', 'past_avg_points', 'team_avg_points', 
                'rain_probability', 'avg_air_temp']
    # Add team columns
    team_cols = [c for c in df_encoded.columns if c.startswith('team_')]
    features += team_cols
    
    target = 'finish_position'
    
    # Filter columns
    model_df = df_encoded[features + [target, 'season', 'event_date']]
    
    return model_df

def train_and_evaluate(df):
    """Trains models and evaluates them."""
    
    # Train/Test Split (Time-series)
    # Train: 2018-2023, Test: 2024-2025
    train_df = df[df['season'] <= 2023]
    test_df = df[df['season'] >= 2024]
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    features = [c for c in df.columns if c not in ['finish_position', 'season', 'event_date']]
    target = 'finish_position'
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    results = {}
    
    # 1. Baseline: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    results['Linear Regression'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'MAE': mean_absolute_error(y_test, y_pred_lr),
        'R2': r2_score(y_test, y_pred_lr)
    }
    
    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    results['Random Forest'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'R2': r2_score(y_test, y_pred_rf)
    }
    
    # Print results
    print("\nModel Evaluation Results:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
    # Feature Importance (RF)
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(importances.head(10))
    
    # Save Feature Importance Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importances['feature'].head(10), importances['importance'].head(10))
    plt.xlabel('Importance')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig("plots/05_feature_importance.png")
    print("\nSaved plots/05_feature_importance.png")

if __name__ == "__main__":
    df = load_data()
    model_df = prepare_modeling_data(df)
    train_and_evaluate(model_df)
