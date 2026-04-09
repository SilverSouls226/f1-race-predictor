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
    # One-hot encoding for Team and Compound
    df_encoded = pd.get_dummies(df, columns=['team', 'main_compound'], drop_first=True)
    
    # Select features
    features = [
        'grid_position', 'past_avg_pos', 'past_avg_points', 'team_avg_points', 
        'rain_probability', 'avg_air_temp', 'avg_sector1', 'avg_sector2', 'avg_sector3',
        'is_classified'
    ]
    # Add team and compound columns
    added_cols = [c for c in df_encoded.columns if c.startswith('team_') or c.startswith('main_compound_')]
    features += added_cols
    
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
    
    # 1. Pipeline: Scaling + Linear Regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

    results = {}
    
    # 1. Baseline: Scaled Linear Regression
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    
    results['Linear Regression'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'MAE': mean_absolute_error(y_test, y_pred_lr),
        'R2': r2_score(y_test, y_pred_lr)
    }
    
    # 2. Random Forest with Tuning
    print("\nTuning Random Forest...")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    rf_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=10, 
        cv=tscv, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
    )
    
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"Best RF Params: {rf_search.best_params_}")
    
    y_pred_rf = best_rf.predict(X_test)
    
    results['Random Forest (Tuned)'] = {
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
            
    # Feature Importance (Best RF)
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(importances.head(10))
    
    # Save Feature Importance Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importances['feature'].head(10), importances['importance'].head(10))
    plt.xlabel('Importance')
    plt.title('Random Forest Feature Importance (Tuned)')
    plt.gca().invert_yaxis()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig("plots/05_feature_importance.png")
    print("\nSaved plots/05_feature_importance.png")

if __name__ == "__main__":
    df = load_data()
    model_df = prepare_modeling_data(df)
    train_and_evaluate(model_df)
