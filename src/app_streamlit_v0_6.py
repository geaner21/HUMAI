import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor

# ===== Paths =====
THIS_DIR = os.path.dirname(__file__)
PROJ_DIR = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(PROJ_DIR, "data")
MODELS_DIR = os.path.join(PROJ_DIR, "models")
REPORTS_DIR = os.path.join(PROJ_DIR, "reports")

MODEL_PATH = os.path.join(MODELS_DIR, "humai_v0_3_rf.pkl")
CSV_PATH = os.path.join(DATA_DIR, "premier-league-player-23-24.csv")

# ===== App Config =====
st.set_page_config(page_title="HUMAI Dashboard v0.6", page_icon="🏟", layout="wide")
st.title("🏟 HUMAI Coach - Live Performance Overview")

# ===== Load data/model =====
@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def preprocess_player_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.fillna(0)
    # !!! corect: PrgR, nu "Prgr"
    df["Performance_Index"] = (
        (df["xG"] * 40 + df["xAG"] * 30 + df["PrgR"] * 20 + df["Min"] / 90 * 10)
    )
    return df

df_raw = load_data()
df = preprocess_player_data(df_raw) # <<< creează Performance_Index ÎNAINTE de a-l folosi
model = load_model()

team = st.selectbox("Select team:", sorted(df["Team"].unique()))
st.bar_chart(df[df["Team"] == team].groupby("Pos")["Performance_Index"].mean())

st.subheader("Top 10 Players by Performance Index")
st.dataframe(df.nlargest(10, "Performance_Index")[["Player", "Team", "Pos", "Performance_Index"]])

# ===== UI (după ce avem Performance_Index) =====
# mic fallback dacă Pos nu există
if "Pos" not in df.columns:
    df["Pos"] = "UNK"

st.metric("Average League Score", round(float(df["Performance_Index"].mean()), 2))
st.bar_chart(df.groupby("Pos")["Performance_Index"].mean())

def retrain_model(df, model_path):
    needed = ["xG", "xAG", "npxG", "PrgR", "PrgP", "Min", "Gls"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Missing columns for retraining: {missing}")
        return None
    
    X = df[["xG", "xAG", "npxG", "PrgR", "PrgP", "Min"]]
    y = df["Gls"]
    model = RandomForestRegressor(n_estimators=800, max_depth=6, random_state=42)
    model.fit(X, y)
    dump(model, model_path)
    st.success(f"✅ Model retrained and saved to {model_path}")
    return model

def categorize_player(score):
    if score > 180: return "🔥 Elite Performer"
    elif score > 130: return "💪 Consistent"
    elif score >80: return "⚙️ Developing"
    else: return "🧩 Prospect"

output_path = os.path.join(REPORTS_DIR, "humai_dashboard_report.csv")
df.to_csv(output_path, index=False)
st.success(f"✅ Report saved to {output_path}")