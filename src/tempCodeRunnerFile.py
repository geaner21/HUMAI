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
CSV_PATH = os.path.join(DATA_DIR, "premier-player-23-24.csv")

# ===== App Config =====
st.set_page_config(page_title="HUMAI Dashboard v0.6", page_icon="🏟", layout="wide")

st.title("🏟 HUMAI Coach - Live Performance Overview")

# ===== Load data/model =====
@st.cache_data
def load_data():
    df=pd.read_csv(CSV_PATH)
    return df

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

df = load_data()
model = load_model()

st.metric("Average Team Score", round(df["Performance_Index"].mean(), 2))
st.bar_chart(df.groupby("Pos")["Performance_Index"].mean())

def preprocess_player_data(df):
    df = df.copy()
    df.fillna(0, inplace=True)
    df["Performance_Index"] = (
        (df["xG"] * 0.4 + df["xAG"] * 0.3 + df["Prgr"] * 0.2 + df["Min"] / 1000 * 0.1)
    )
    return df

def retrain_model(df, model_path):
    X = df[["xG", "xAG", "npxG", "PrgR", "PrgP", "Min"]]
    y = df["Gls"]

    model = RandomForestRegressor(n_estimators=800, max_depth=6, random_state=42)
    model.fit(X, y)

    dump(model, model_path)
    print(f"✅ Model retrained and saved to {model_path}")
    return model

def categorize_player(score):
    if score > 180: return "🔥 Elite Performer"
    elif score > 130: return "💪 Consistent"
    elif score > 80: return "⚙️ Developing"
    else: return "🧩 Prospect"