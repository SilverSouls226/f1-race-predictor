# Formula 1 Race Outcome Prediction

![Feature Importance](outputs/plots/05_feature_importance.png)

Predicting **Formula 1 driver finishing positions** using machine learning on historical race telemetry.

This project builds an end-to-end pipeline that collects Formula 1 race data from the **FastF1 API**, processes lap telemetry into race-level features, and trains regression models to predict race outcomes.

The dataset spans **2018–2025 seasons** and contains roughly **~200,000 lap telemetry records**.

---

# Project Pipeline

```
FastF1 API
   ↓
collect_fastf1_data.py
   ↓
season_*.csv (lap telemetry)
   ↓
data_prep.py
   ↓
driver-race dataset
   ↓
EDA (R)
   ↓
Machine Learning Models
```

The final dataset contains **one row per driver per race**, with features describing race conditions, driver performance, and team strength.

---

# Repository Structure

```
f1-race-predictor
│
├── src
│   ├── collect_fastf1_data.py
│   ├── data_prep.py
│   └── modeling.py
│
├── analysis
│   └── eda.R
│
├── data
│   ├── raw
│   └── processed
│   └── dl_processed
│
├── plots
│   └── visualizations
│
├── app.py (Interactive Streamlit Dashboard)
├── 05_dashboard.py (Legacy Dash UI)
│
└── report
    └── project report and methodology
```

---

# Models

Two regression models were implemented:

**Linear Regression**

* baseline model

**Random Forest Regressor**

**Gradient Boosting Regressor**
*   tree-based ensemble utilizing error-correction boosting

**Deep Learning (PyTorch MLP)**
*   Multi-layer perceptron leveraging high-density categorical embeddings for Driver and Team dynamics, utilizing a wide architecture (256 -> 128 layers).

Evaluation metrics:

```
RMSE
MAE
R²
```

Test set: **2024–2025 seasons**

---

# Example Results

| Model             | RMSE | MAE  | R²   |
| ----------------- | ---- | ---- | ---- |
| Linear Regression | 3.75 | 2.97 | 0.57 |
| Gradient Boosting | 3.20 | 2.41 | 0.69 |
| Random Forest     | 3.11 | 2.33 | 0.71 |
| PyTorch MLP (Wide)| 2.98 | 2.12 | 0.76 |

Grid position, team performance, and driver historical results were among the strongest predictors.

---

# Running the Project

Install dependencies:

```
pip install -r requirements.txt
```

Collect race telemetry:

```
python src/collect_fastf1_data.py
```

Process the dataset:

```
python src/data_prep.py
```

Generate analysis plots:

```
Rscript analysis/eda.R
```

Train Classic Models (RF, GB, LR):

```
python src/modeling.py
```

Train Deep Learning Matrix (PyTorch):

```
python src/dl_modeling.py
```

Run Interactive Streamlit Dashboard:

```
python -m streamlit run app.py
```

---

# Tech Stack

- Python
- R
- FastF1 API
- pandas
- scikit-learn
- ggplot2
- matplotlib

---

# License

This project is licensed under the **MIT License**.

---
