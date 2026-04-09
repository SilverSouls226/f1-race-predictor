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

### 3. Model Metrics (Test Set 2024-2025)

| Model | RMSE | MAE | R-Squared | Status |
|-------|------|-----|-----------|--------|
| Linear Regression (Scaled) | 3.7582 | 2.9794 | 0.5734 | Solid Baseline |
| Gradient Boosting | 3.2016 | 2.4125 | 0.6904 | Scikit Tree Engine |
| Random Forest (Tuned) | 3.1095 | 2.3278 | 0.7079 | Phase 2 Peak System |
| Deep Learning MLP (Wide Set) | 2.9811 | 2.1244 | 0.7610 | **Highest Precision** |

### 4. Deep Learning Expansion (`src/dl_data_prep.py`, `src/dl_modeling.py`)
- **Change**: Added an entirely new Deep Learning infrastructure using PyTorch.
- **Why**: Traditional models max out at complex non-linear relationships. PyTorch allows the use of embedding layers for categorical variables (Drivers & Teams) to recognize dynamic similarities. 
- **Change**: Expanded the neural network architecture horizontally (`[256, 128]` Hidden Layers) across 350 epochs.
- **Why**: Overcoming the 75% accuracy drop-off inherent in F1 dataset sparsity required heavier feature mapping.
- **Result**: The deep learning wide model successfully broke the 75% goal, achieving an R-Squared of **0.761**.

### 5. Streamlit Interactive Hub (`app.py`)
- **Change**: Developed a comprehensive real-time dashboard spanning Historical Analytics (Positions gained vs grid spot), Model Evaluation alignment scatter matrices, and a Hypthotetical Race Simulator.
- **Why**: Dash applications possessed heavier latency; Streamlit natively allowed faster UI iteration while providing users access to interact with the PyTorch Neural network directly via dropdown widgets mimicking changing weather/grid conditions.
