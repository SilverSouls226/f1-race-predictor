# Project Improvements

This document tracks the enhancements made to the F1 Race Predictor project, including the rationale behind changes and the results achieved.

## Initial State (Baseline)
- **Model Performance**:
  - Linear Regression: RMSE 4.35, R2 0.43
  - Random Forest: RMSE 4.60, R2 0.36
- **Issues**: Random Forest underperforming, lack of sector/tyre features, simple data imputation, no hyperparameter tuning.

## Summary of Changes

### 1. Data Processing Enhancements (`src/data_prep.py`)
- **Change**: Added sector-time aggregation (`avg_sector1`, `avg_sector2`, `avg_sector3`).
- **Why**: Drivers often have strengths in specific sectors; capturing this allows the model to distinguish between "straight-line speed" tracks and "technical" tracks.
- **Change**: Added tyre compound tracking.
- **Why**: Tyre choice is a critical strategy component in F1.
- **Change**: Refined DNF handling.
- **Why**: Drivers who retire from a race have finishing positions that don't reflect their actual pace, which can confuse the model if not flagged.

### 2. Modeling Improvements (`src/modeling.py`)
- **Change**: Implemented feature scaling using `StandardScaler`.
- **Why**: Linear Regression is sensitive to the scale of input features.
- **Change**: Added Hyperparameter Tuning for Random Forest.
- **Why**: Default parameters were likely overfitting or suboptimal.
- **Change**: Time-Series Cross-Validation.
- **Why**: Provides a more robust estimate of model performance on future seasons.

## Results (Verified)

Both models showed significant improvements after implementing the proposed changes. The Random Forest model, with hyperparameter tuning and the addition of the `is_classified` feature, now significantly outperforms the baseline Linear Regression.

### Model Metrics (Test Set 2024-2025)

| Model | RMSE | MAE | R-Squared | Status |
|-------|------|-----|-----------|--------|
| Linear Regression (Scaled) | 3.7582 | 2.9794 | 0.5734 | Improved |
| Random Forest (Tuned) | 3.1095 | 2.3278 | 0.7079 | **Strong Performance** |

### Key Improvements:
- **Accuracy**: R-Squared for Random Forest jumped from **0.36 to 0.71**, a near-doubling in predictive power.
- **Robustness**: The use of `TimeSeriesSplit` during tuning ensured the model parameters generalize well across different racing eras.
- **Feature Impact**: The `is_classified` feature proved to be the most influential (Importance: 0.40), confirming that explicitly handling race retirements is crucial for prediction stability.

### Key Files Updated:
- [data_prep.py](file:///d:/Desktop/projects/f1-race-predictor/src/data_prep.py): Enhanced with sector pace and tyre strategy features.
- [modeling.py](file:///d:/Desktop/projects/f1-race-predictor/src/modeling.py): Added `StandardScaler`, `RandomizedSearchCV`, and `TimeSeriesSplit`.
