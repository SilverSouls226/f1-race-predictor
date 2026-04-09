import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import joblib
import os
import sys

# Add src to path so we can import dataset/model classes if needed cleanly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Deep Learning definition inside the app to load weights easily
class F1Predictor(nn.Module):
    def __init__(self, num_features, embedding_info, embedding_dim=12, hidden1=256, hidden2=128, dp1=0.0, dp2=0.0):
        super(F1Predictor, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim)
            for col, num_embeddings in embedding_info.items()
        ])
        total_emb_dim = len(embedding_info) * embedding_dim
        input_dim = num_features + total_emb_dim
        
        layers = [
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dp1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        ]
        if dp2 > 0:
            layers.append(nn.Dropout(dp2))
        layers.append(nn.Linear(hidden2, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x_num, x_cat):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        embs = torch.cat(emb_outs, dim=1)
        x = torch.cat([x_num, embs], dim=1)
        return self.network(x)

# --- CONFIG AND CACHING ---
st.set_page_config(page_title="F1 Analytics & Predictor", page_icon="🏎️", layout="wide")

@st.cache_data
def load_historical_data():
    return pd.read_csv("data/processed/f1_driver_race.csv")

@st.cache_resource
def load_dl_artifacts():
    try:
        scaler = joblib.load('data/dl_processed/scaler.joblib')
        cat_encoders = joblib.load('data/dl_processed/cat_encoders.joblib')
        embedding_info = joblib.load('data/dl_processed/embedding_info.joblib')
        
        # Num features count (from dl_data_prep.py: 12 features generally + engineered ones if present)
        # Let's dynamically load the model state and infer if needed, or explicitly pass.
        # From the script, num_features was 12. Let's verify by just using the scaler shape.
        num_features = scaler.mean_.shape[0]
        
        dl_model = F1Predictor(num_features=num_features, embedding_info=embedding_info)
        dl_model.load_state_dict(torch.load('models/f1_dl_model.pth', weights_only=True))
        dl_model.eval()
        return dl_model, scaler, cat_encoders
    except Exception as e:
        return None, None, None

df = load_historical_data()
dl_model, scaler, cat_encoders = load_dl_artifacts()

# --- HEADER & STYLING ---
st.markdown("""
<style>
/* Center the main title */
h1 {
    text-align: center;
}
/* Center the Tabs List */
.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🏎️ F1 Race Predictor & Analytics Hub")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Historical Analytics", "🧠 Model Evaluation", "🔮 Live Predictor"])

# ==========================================
# TAB 3: LIVE PREDICTOR (Moved to the back)
# ==========================================
with tab3:
    st.header("Hypothetical Race Simulator")
    st.subheader("Race Conditions")
    if cat_encoders:
        driver_options = list(cat_encoders['driver'].classes_)
        team_options = list(cat_encoders['team'].classes_)
        compound_options = list(cat_encoders['main_compound'].classes_)
        
        sel_driver = st.selectbox("Driver", driver_options, index=driver_options.index("Max Verstappen") if "Max Verstappen" in driver_options else 0)
        sel_team = st.selectbox("Team", team_options, index=team_options.index("Red Bull") if "Red Bull" in team_options else 0)
        sel_grid = st.slider("Grid Position", 1, 20, 1)
        sel_rain = st.slider("Rain Probability", 0.0, 1.0, 0.0, step=0.1)
        sel_compound = st.selectbox("Main Tyre Compound", compound_options)
        
        sel_temp = st.slider("Track Temperature (°C)", 15, 60, 30)
        
        if st.button("Run Simulation", type="primary"):
            # Construct mock vector for the DL model
            with st.spinner("Processing through Deep Learning Neural Network..."):
                try:
                    # Fallback heuristic averages for the continuous vars
                    avg_past_pos = df[df['driver'] == sel_driver]['finish_position'].mean() if len(df[df['driver'] == sel_driver]) > 0 else 10.0
                    avg_past_points = df[df['driver'] == sel_driver]['points'].mean() if len(df[df['driver'] == sel_driver]) > 0 else 5.0
                    avg_team_points = df[df['team'] == sel_team]['points'].mean() if len(df[df['team'] == sel_team]) > 0 else 5.0
                    
                    # Numerical features based on typical order
                    # 'grid_position', 'past_avg_pos', 'past_avg_points', 'team_avg_points', 'avg_sector1', 'avg_sector2', 'avg_sector3', 'avg_air_temp', 'avg_track_temp', 'avg_humidity', 'rain_probability', 'is_classified'
                    num_vector = np.array([[
                        sel_grid, avg_past_pos, avg_past_points, avg_team_points, 
                        30.0, 40.0, 30.0, # Dummy sector times
                        sel_temp - 10, sel_temp, 50.0, # Weather
                        sel_rain, 1.0 # Classified
                    ]])
                    
                    # Categorical: 'driver', 'team', 'race_name', 'main_compound'
                    cat_vector = np.array([[
                        cat_encoders['driver'].transform([sel_driver])[0],
                        cat_encoders['team'].transform([sel_team])[0],
                        0, # Dummy race name
                        cat_encoders['main_compound'].transform([sel_compound])[0]
                    ]])
                    
                    # Scale
                    num_scaled = scaler.transform(num_vector)
                    
                    # Predict
                    x_num = torch.tensor(num_scaled, dtype=torch.float32)
                    x_cat = torch.tensor(cat_vector, dtype=torch.long)
                    
                    pred = dl_model(x_num, x_cat).detach().numpy()[0][0]
                    
                    st.success(f"Simulation Complete!")
                    st.metric(label="Predicted Finish Position", value=f"P{max(1, int(round(pred)))}")
                    
                except Exception as e:
                    st.error(f"Error running predictor: {e}")
    else:
        st.error("Model artifacts not found. Please run the Deep Learning pipeline first.")
            
# ==========================================
# TAB 1: HISTORICAL ANALYTICS
# ==========================================
with tab1:
    st.header("Performance Analytics Dashboard")
    
    st.subheader("Finish Position Variance by Driver (All Drivers)")
    # Driver performance over time split into two graphs
    all_drivers = df['driver'].value_counts().index
    mid_point = len(all_drivers) // 2
    
    drivers_part_1 = all_drivers[:mid_point]
    drivers_part_2 = all_drivers[mid_point:]
    
    df_part_1 = df[df['driver'].isin(drivers_part_1)]
    df_part_2 = df[df['driver'].isin(drivers_part_2)]
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig1a = px.box(df_part_1, x="driver", y="finish_position", color="driver", 
                       title="Finish Position Variance (Top Half by Races)", points="all")
        fig1a.update_yaxes(autorange="reversed")
        st.plotly_chart(fig1a, use_container_width=True)
        
    with col_d2:
        fig1b = px.box(df_part_2, x="driver", y="finish_position", color="driver", 
                       title="Finish Position Variance (Bottom Half by Races)", points="all")
        fig1b.update_yaxes(autorange="reversed")
        st.plotly_chart(fig1b, use_container_width=True)
        
    st.markdown("---")
    
    colA, colB = st.columns(2)
    with colA:
        # Grid vs Finish Scatter
        fig2 = px.scatter(df, x="grid_position", y="finish_position", color="season", 
                          trendline="ols", title="Grid vs. Finish Position Correlation",
                          hover_data=["driver", "race_name"])
        # Set origin to 0,0 (bottom-left)
        max_pos = max(df['grid_position'].max(), df['finish_position'].max()) + 2
        fig2.update_xaxes(range=[0, max_pos])
        fig2.update_yaxes(range=[0, max_pos])
        st.plotly_chart(fig2, use_container_width=True)
        
    with colB:
        st.markdown("### Top Teams Sector Pace")
        top_teams = df['team'].value_counts().head(5).index
        team_df = df[df['team'].isin(top_teams)].groupby('team')[['avg_sector1', 'avg_sector2', 'avg_sector3']].mean().reset_index()
        
        fig3 = go.Figure(data=[
            go.Bar(name='Sector 1', x=team_df['team'], y=team_df['avg_sector1']),
            go.Bar(name='Sector 2', x=team_df['team'], y=team_df['avg_sector2']),
            go.Bar(name='Sector 3', x=team_df['team'], y=team_df['avg_sector3'])
        ])
        fig3.update_layout(barmode='group', title="Average Sector Times (Lower is Better)")
        st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Overtaking Prowess (Average Positions Gained)")
    df['position_gain'] = df['grid_position'] - df['finish_position']
    # Filter to main drivers for cleaner graph
    gain_df = df[df['driver'].isin(all_drivers)].groupby('driver')['position_gain'].mean().reset_index().sort_values('position_gain', ascending=False)
    
    fig_gain = px.bar(gain_df, x='driver', y='position_gain', color='position_gain',
                      color_continuous_scale='RdYlGn',
                      title="Net Positions Gained from Grid to Finish")
    st.plotly_chart(fig_gain, use_container_width=True)

# ==========================================
# TAB 2: MODEL EVALUATION
# ==========================================
with tab2:
    st.header("Algorithm Performance Matrix")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Linear Regression Accuracy (R²)", "57.0%")
    m2.metric("Random Forest Accuracy (R²)", "71.0%")
    m3.metric("Deep Learning Accuracy (R²)", "76.1%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    e1, e2, e3 = st.columns(3)
    e1.metric("Linear Regression RMSE (Accuracy)", "3.76 positions")
    e2.metric("Random Forest RMSE (Accuracy)", "3.11 positions")
    e3.metric("Deep Learning RMSE (Accuracy)", "2.98 positions")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    a1, a2, a3 = st.columns(3)
    a1.metric("Linear Regression MAE", "2.98 positions")
    a2.metric("Random Forest MAE", "2.33 positions")
    a3.metric("Deep Learning MAE", "2.12 positions")
    
    st.markdown("---")
    
    st.subheader("Predictive Alignment (Actual vs Predicted Finish)")
    try:
        df_preds = pd.read_csv("data/model_predictions.csv")
        p1, p2, p3 = st.columns(3)
        max_val = max(df_preds["Actual"].max(), df_preds["RF_Pred"].max())
        
        def make_pred_plot(model_col, title):
            fig = px.scatter(df_preds, x="Actual", y=model_col, color="team", 
                             hover_data=["driver", "event_date"], title=title)
            fig.add_shape(type="line", x0=1, y0=1, x1=max_val, y1=max_val, line=dict(color="White", dash="dash"))
            return fig
            
        with p1:
            st.plotly_chart(make_pred_plot("RF_Pred", "Random Forest"), use_container_width=True)
        with p2:
            st.plotly_chart(make_pred_plot("GB_Pred", "Gradient Boosting"), use_container_width=True)
        with p3:
            if "DL_Pred" in df_preds.columns:
                st.plotly_chart(make_pred_plot("DL_Pred", "Deep Learning"), use_container_width=True)
            else:
                st.warning("DL_Pred missing from predictions.")
                
    except FileNotFoundError:
        st.error("Could not locate data/model_predictions.csv to plot visual alignments.")
        
    st.markdown("---")
    
    st.subheader("RF Feature Importance Breakdown")
    # Hardcoded from Phase 2 findings for static dashboard.
    # Reversed so that Plotly horizontal bar puts the most important at the top.
    features = ['Is Classified', 'Grid Position', 'Team Power', 'Weather/Sectors', 'Driver Past Form'][::-1]
    importance = [0.40, 0.21, 0.15, 0.12, 0.12][::-1]
    
    fig4 = px.bar(x=importance, y=features, orientation='h', 
                  title="Random Forest Feature Prominence", 
                  labels={'x': 'Relative Importance', 'y': 'Feature'})
    st.plotly_chart(fig4, use_container_width=True)
