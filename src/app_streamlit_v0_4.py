import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ===== Paths =====
THIS_DIR = os.path.dirname(__file__)
PROJ_DIR = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(PROJ_DIR, "data")
MODELS_DIR = os.path.join(PROJ_DIR, "models")
REPORTS_DIR = os.path.join(PROJ_DIR, "reports")

MODEL_PATH = os.path.join(MODELS_DIR, "humai_v0_3_rf.pkl")
CSV_PATH = os.path.join(DATA_DIR, "premier-player-23-24.csv")

# ===== App Config =====
st.set_page_config(page_title="HUMAI Dashboard v0.4", page_icon="🧠", layout="wide")

st.title("🧠 HUMAI Dashboard v0.4")
st.caption("AI Coach • Predictions & Insights for Player Performance")

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

# Features (exactly the order from Day 17)
FEATURES = [
    "xG", "xAG", "npxG", "PrgP", "PrgC", "PrgR", "Min", "Age", "90s",
    "xG_per90", "xAG_per90", "npxG_per90", "usage"
]

# Compute engineered features if missing
if not set(FEATURES).issubset(df.columns):
    # derive engineered columns
    df = df.copy()
    df["xG_per90"] = df["xG"] / df["90s"].replace(0, np.nan)
    df["xAG_per90"] = df["xAG"] / df["90s"].replace(0, np.nan)
    df["npxG_per90"] = df["npxG"] / df["90s"].replace(0, np.nan)
    df["usage"] = df["Min"] / (df["90s"] * 90).replace(0, np.nan)
    df = df.fillna(0)

# ===== Sidebar: pick mode =====
mode = st.sidebar.radio("Mode", ["Pick existing player", "Manual input"])

def player_card(row):
    st.markdown(f"### {row['Player']} - {row['Team']}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gls", f"{row.get('Gls', 0):.0f}")
    col2.metric("Ast", f"{row.get('Ast', 0):.0f}")
    col3.metric("xG", f"{row.get('xG', 0):.2f}")
    col4.metric("xAG", f"{row.get('xAG', 0):.2f}")

def predict_from_features(sample_df):
    # model is a Pipeline (with scaler + rf), safe to pass DataFrame with same FEATURES
    y_pred = model.predict(sample_df[FEATURES])[0]
    return float(y_pred)

st.sidebar.markdown("-")

# ===== Mode 1: Existing player =====
if mode == "Pick existing player":
    # Filter by team / player for quicker search
    teams = ["All"] + sorted(df["Team"].dropna().unique().tolist())
    team = st.sidebar.selectbox("Team", teams, index=0)

    pool = df if team == "All" else df[df["Team"] == team]
    player = st.sidebar.selectbox("Player", sorted(pool["Player"].unique().tolist()))

    row = pool[pool["Player"] == player].iloc[0].copy()

    # derive engineered features (for the selected row)
    row["xG_per90"] = 0 if row["90s"] == 0 else row["xG"] / row["90s"]
    row["xAG_per90"] = 0 if row["90s"] == 0 else row["xAG"] / row["90s"]
    row["npxG_per90"] = 0 if row["90s"] == 0 else row["npxG"] / row["90s"]
    row["usage"] = 0 if row["90s"] == 0 else row["Min"] / (row["90s"] * 90)

    st.subheader("🎯 Player Overview")
    player_card(row)

    # Show inputs (editable sliders) seeded with player values
    st.subheader("⚙️ Adjust inputs (what-if)")
    cols = st.columns(4)
    row["xG"] = cols[0].slider("xG", 0.0, float(max(1.0, df["xG"].max())), float(row["xG"]), 0.1)
    row["xAG"] = cols[1].slider("xAG", 0.0, float(max(1.0, df["xAG"].max())), float(row["xAG"]), 0.1) 
    row["npxG"] = cols[2].slider("npxG", 0.0, float(max(1.0, df["npxG"].max())), float(row["npxG"]), 0.1) 
    row["PrgP"] = cols[3].slider("PrgP", 0.0, float(max(1.0, df["PrgP"].max())), float(row["PrgP"]), 1.0)

    cols2 = st.columns(4)
    row["PrgC"] = cols2[0].slider("PrgC", 0.0, float(max(1.0, df["PrgC"].max())), float(row["PrgC"]), 1.0)
    row["PrgR"] = cols2[1].slider("PrgR", 0.0, float(max(1.0, df["PrgR"].max())), float(row["PrgR"]), 1.0)
    row["Min"] = cols2[2].slider("Min", 0.0, float(max(90.0, df["Min"].max())), float(row["Min"]), 10.0)
    row["Age"] = cols2[3].slider("Age", 16.0, 40.0, float(row["Age"]), 0.5)

    row["90s"] = st.slider("90s (full matches played)", 0.0, float(max(1.0, df["90s"].max())), float(row["90s"]), 0.1)

    # recompute engineered after edits
    row["xg_per90"] = 0 if row["90s"] == 0 else row ["xG"] / row ["90s"]
    row["xAG_per90"] = 0 if row["90s"] == 0 else row ["xAG"] / row["90s"]
    row["npxG_per90"] = 0 if row["90s"] == 0 else row ["npxG"] / row["90s"]
    row["usage"] = 0 if row["90s"] == 0 else row ["Min"] / (row["90s"] * 90)

    sample = pd.DataFrame([row])[FEATURES]
    pred = predict_from_features(sample)

    st.success(f"🔮 **Predicted Goals**: {pred:.2f}")

    # Simple parity vs actual (if exists)
    if "Gls" in df.columns:
        st.caption(f"Actual last season: **{float(row.get("Gls", np.nan)):.0f}** goals")

    # Mini chart: how inputs compare to dataset distribution
    st.subheader("📊 Context vs dataset")
    cols3 = st.columns(3)
    for i, feat in enumerate(["xG", "npxG", "PrgR"]):
        ax = cols3[i].container()
        fig, axp = plt.subplots(figsize=(3.5,2.2))
        axp.hist(df[feat].dropna(), bins=20)
        axp.axvline(row[feat], color="red")
        axp.set_title(f"{feat} distribution")
        ax.pyplot(fig)

# ===== Mode 2: Manual input =====
else:
    st.subheader("✍️ Manual Player Inputs")

    def slider_float(label, minv, maxv, val, step):
        return st.slider(label, float(minv), float(maxv), float(val), float(step))
    
    xG = slider_float("xG", 0, 35, 10, 0.1)
    xAG = slider_float("xAG", 0, 25, 6, 0.1)
    npxG = slider_float("npxG", 0, 35, 9, 0.1)
    PrgP = slider_float("PrgP", 0, 400, 150, 1)
    PrgC = slider_float("PrgC", 0, 350, 120, 1)
    PrgR = slider_float("PrgR", 0, 400, 160, 1)
    Min = slider_float("Min", 0, 3240, 2200, 10)
    Age = slider_float("Age", 16, 40, 25, 0.5)
    n90s = slider_float("90s", 0, 38, 24, 0.5)

    xG_per90 = 0 if n90s == 0 else xG / n90s
    xAG_per90 = 0 if n90s == 0 else xAG / n90s
    npxG_per90 = 0 if n90s == 0 else npxG / n90s
    usage = 0 if n90s == 0 else Min / (n90s * 90)

    sample = pd.DataFrame([{
        "xG": xG, "xAG": xAG, "npxG": npxG, "PrgP": PrgP, "PrgC": PrgC, "PrgR": PrgR,
        "Min": Min, "Age": Age, "90s": n90s,
        "xG_per90": xG_per90, "xAG_per90": xAG_per90, "npxG_per90": npxG_per90, "usage": usage
    }])

    pred = predict_from_features(sample)
    st.success(f"🔮 **Predicted Goals**: {pred:.2f}")

st.divider()
st.caption("© HUMAI - The Human Intelligence Company")


    