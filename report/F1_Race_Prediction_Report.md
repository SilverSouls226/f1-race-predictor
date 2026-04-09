# F1 Race Prediction Project Report

## 1. Project Overview
This project establishes a machine learning pipeline to predict F1 race outcomes.
**Objective**: Predict the finishing position of drivers based on historical performance, qualifying results, and team characteristics.
**Data Source**: F1 lap-level data from 2018-2025.

## 2. Methodology

### 2.1 Phase 1: Baseline Implementation
- **Data Preparation**: Lap-level data was aggregated to driver-race level.
- **Handling Missing Values**: Pit stops were converted to counts. Qualifying times were handled (though primarily grid position was used).
- **Feature Engineering**:
    - `position_gain`: Difference between Grid and Finish position.
    - `past_avg_pos`, `past_avg_points`: Rolling average of driver performance to quantify "Driver Skill".
    - `team_avg_points`: Metric for "Team Power".
    - One-hot encoding for Teams.
- **Modeling Approach**:
    - Basic Linear Regression and Random Forest Regressor.
    - Standard Time-series split (Train: 2018-2023, Test: 2024-2025).

### 2.2 Phase 2: Optimized Pipeline & Tuning
- **Enhanced Feature Engineering**:
    - `avg_sector1`, `avg_sector2`, `avg_sector3`: Introduced sector-level pace analysis to capture track-specific driver strengths.
    - `is_classified`: Added a binary reliability flag to explicitly handle DNFs/retirements.
    - `main_compound`: One-hot encoding for the primary tyre compound used in the race.
- **Advanced Modeling**:
    - **Feature Scaling**: Implemented `StandardScaler` pipeline for Linear Regression.
    - **Hyperparameter Tuning**: Utilized `RandomizedSearchCV` to optimize Random Forest parameters (n_estimators, max_depth, etc.).
    - **Validation**: Implemented `TimeSeriesSplit` (5-fold) within the training set for more robust parameter selection.

## 3. Exploratory Data Analysis (EDA)
Visualizations were generated using R (`ggplot2`, `corrplot`).
- **Lap Time Distribution**: Analysis of car performance spread.
- **Qualifying vs Race**: Identifying correlation between starting and finishing position.
- **Correlation Heatmap**: Identifying key drivers of success.

### Exploratory Analysis Visualization
*(Plots are saved in `outputs/plots/` directory)*

**1. Qualifying importance**:
![Qualifying vs Race Position](/outputs/plots/02_qualifying_vs_race_pos.png)
*Strong correlation observes, but variance exists due to race incidents/strategy.*

**2. Team Performance**:
![Lap Time Distribution](/outputs/plots/01_lap_time_distribution.png)
*Top teams show consistently lower and more stable lap times.*

**3. Driver Trends**:
![Driver Performance Trend](/outputs/plots/04_driver_performance_trend.png)

## 4. Results & Evaluation

### 4.1 Phase 1: Baseline Results
*(Metrics from Test Set 2024-2025)*

| Model | RMSE | MAE | R-Squared |
|-------|------|-----|-----------|
| Linear Regression | 4.35 | 3.49 | 0.43 |
| Random Forest | 4.60 | 3.68 | 0.36 |

*Note: In the baseline run, Linear Regression slightly outperformed Random Forest, suggesting a strong linear component in the predictors (like Grid Position).*

### 4.2 Phase 2: Optimized Results (Verified)
*(After implementing sector analysis, DNF handling, and hyperparameter tuning)*

| Model | RMSE | MAE | R-Squared |
|-------|------|-----|-----------|
| Linear Regression (Scaled) | 3.76 | 2.98 | 0.57 |
| **Random Forest (Tuned)** | **3.11** | **2.33** | **0.71** |

*Analysis: The transition to Phase 2 resulted in a near-doubling of the Random Forest R-Squared value, moving from 0.36 to 0.71.*

### 4.3 Feature Importance Evolution

| Feature | Phase 1 Importance | Phase 2 Importance |
|-------|------|-----|
| **Is Classified (DNF risk)** | N/A | **0.40** |
| **Grid Position** | 0.19 | 0.21 |
| **Team Power** | 0.13 | 0.15 |
| **Weather/Sectors** | 0.12 (Temp) | 0.08 (Sect 2) |

## 5. Phase 3: Deep Learning Implementation (New)

The project has recently expanded to include a deep learning neural network built with PyTorch (`src/dl_modeling.py`). 

### Architecture
- **Categorical Embeddings**: Instead of simple one-hot encoding, the model learns continuous vector representations (embeddings) for critical entities like Drivers and Teams. This allows the model to map relationship distances (e.g., driver styles).
- **Multi-layer Perceptron (MLP)**: A feed-forward network mapping the concatenated numerical features + embeddings to regression outputs. (Configuration: 64 neurons -> Dropout 20% -> 32 neurons -> 1 output).

### 5.1 Deep Learning vs. Phase 2 Machine Learning

*(Metrics evaluated on 2024-2025 Test Set)*

| Model | RMSE | MAE | R-Squared | Status |
|-------|------|-----|-----------|--------|
| Linear Regression (Scaled) | 3.76 | 2.98 | 0.57 | Phase 2 Baseline |
| **Random Forest (Tuned)** | **3.11** | **2.33** | **0.71** | **Phase 2 Peak Performance** |
| Gradient Boosting | 3.20 | 2.41 | 0.69 | Tree Baseline Supplemental |
| **Deep Learning MLP (256->128 Wide Net)** | **2.98** | **2.12** | **0.761** | **Current Peak Accuracy** |

### 5.2 Analysis

The introduction of the Deep Learning model resulted in the highest statistical ceiling for the project thus far. By scaling the neural network horizontally (bypassing dropouts to utilize a 256 -> 128 hidden layer array over 350 epochs), it forcibly extracted high-fidelity relationships out of the Driver/Team Categorical Embeddings. The test-set R² eclipsed the 75% goal line (`R2: 0.761`), outperforming even the optimized Random Forest structure. 

## 6. Streamlit Dashboard `app.py`

Following the legacy `05_dashboard.py` Dash application, the UX flow was entirely refactored into a modern `Streamlit` instance. 

Key features include:
1. **Interactive Live Predictor**: Allowing users to dynamically inject Weather, Grid Position, Driver, and Tyre choices directly into the `PyTorch` (`f1_dl_model.pth`) weights matrix for instantaneous hypotheticals.
2. **Visual Model Alignments**: Utilizing the backend pipelines, actual validation points vs predicted validation points from Random Forest, Gradient Boosting, and PyTorch models are rendered identically side-by-side using `Plotly`.
3. **Tabular Centering**: Streamlit's constraints were subverted using injected HTML CSS wrappers to gracefully center the UX components.

## 6. Conclusion
The project successfully evolved from a baseline predictor (R2 ~0.4) to an optimized pipeline achieving high-fidelity results (R2 ~0.7). The single most impactful improvement was the explicit modeling of race reliability via the `is_classified` flag, followed by systematic hyperparameter tuning of the Random Forest model. Now, a robust PyTorch infrastructure offers further avenues of improvement via Deep Learning.

Future work will continue to explore:
- Deep Learning Hyperparameter Tuning (epochs, learning rate scheduling, deeper network architectures).
- Live-race simulation features.
- More granular tyre strategy features.
- Real-time telemetry integration.
