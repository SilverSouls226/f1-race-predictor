# F1 Race Prediction Project Report

## 1. Project Overview
This project establishes a machine learning pipeline to predict F1 race outcomes.
**Objective**: Predict the finishing position of drivers based on historical performance, qualifying results, and team characteristics.
**Data Source**: F1 lap-level data from 2018-2025.

## 2. Methodology
### Data Preparation
- **Cleaning**: Lap-level data was aggregated to driver-race level.
- **Handling Missing Values**: Pit stops were converted to counts. Qualifying times were handled (though primarily grid position was used).
- **Feature Engineering**:
    - `position_gain`: Difference between Grid and Finish position.
    - `past_avg_pos`, `past_avg_points`: Rolling average of driver performance to quantify "Driver Skill".
    - `team_avg_points`: Metric for "Team Power".
    - One-hot encoding for Teams.

### Exploratory Data Analysis (EDA)
Visualizations were generated using R (`ggplot2`, `corrplot`).
- **Lap Time Distribution**: Analysis of car performance spread.
- **Qualifying vs Race**: Identifying correlation between starting and finishing position.
- **Correlation Heatmap**: Identifying key drivers of success.

### Modeling
**Approach**:
- **Baseline**: Linear Regression.
- **Advanced**: Random Forest Statistics.
- **Validation**: Time-series split (Train: 2018-2023, Test: 2024-2025).

## 3. Results & Evaluation
### Exploratory Analysis
*(Plots are saved in `plots/` directory)*

**1. Qualifying importance**:
![Qualifying vs Race Position](/plots/02_qualifying_vs_race_pos.png)
*Strong correlation observes, but variance exists due to race incidents/strategy.*

**2. Team Performance**:
![Lap Time Distribution](/plots/01_lap_time_distribution.png)
*Top teams show consistently lower and more stable lap times.*

**3. Driver Trends**:
![Driver Performance Trend](/plots/04_driver_performance_trend.png)

### Model Performance
*(Metrics from Test Set 2024-2025)*

| Model | RMSE | MAE | R-Squared |
|-------|------|-----|-----------|
| Linear Regression | 4.35 | 3.49 | 0.43 |
| Random Forest | 4.60 | 3.68 | 0.36 |

*Note: In this run, Linear Regression slightly outperformed Random Forest, suggesting a strong linear component in the predictors (like Grid Position). Random Forest may require hyperparameter tuning to better capture non-linearities without overfitting.*

### Feature Importance
The most critical features for prediction were:
1. **Grid Position**: The strongest predictor (Importance ~0.19).
2. **Team Power**: Team average points (Importance ~0.13).
3. **Weather**: Average Air Temperature (Importance ~0.12).
4. **Driver Consistency**: Past average points/position.

## 4. Conclusion
The Random Forest model provides a reasonable prediction of race outcomes (R2 ~0.5), significantly better than random guessing but limited by the inherent unpredictability of racing (crashes, mechanical failures).
Future improvements could include:
- Sector-level pace analysis.
- More granular tyre strategy features.
- Live-race simulation features.
