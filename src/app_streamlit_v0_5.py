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
st.set_page_config(page_title="HUMAI Dashboard v0.5", page_icon="🧠", layout="wide")

st.title("🧠 HUMAI Dashboard v0.5")
st.caption("AI Coach • Predictions & Insights for Player Performance")
tab1, tab2 = st.tabs(["🔁 Compare 2 Players", "🧪 Scenario Planning"])

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

def predict_df(model, df_features, FEATURES):
    return model.predict(df_features[FEATURES])

def coach_text(delta):
    if delta > 2:
        return f"🔥 Major uplift (+{delta:.2f}). Continue: raise xG through positioning and PrgR.)"
    if delta > 0.5:
        return f"✅ Moderate uplift (+{delta:.2f}). Maintain intensity and progressive passes."
    if delta > -0.5:
        return f"⚖️ Stable (Δ {delta:.2f}). Optimize minutes and recovery."
    return f"⚠️ Decrease (Δ {delta:.2f}). Review non-penalty shots and creativity (xAG)."

with tab1:
    st.subheader("🔁 Compare two players (side-by-side)")

    teams = ["All"] + sorted(df["Team"].dropna().unique().tolist())
    c1, c2 = st.columns(2)

    teamA = c1.selectbox("Team A", teams, index=0, key="teamA")
    poolA = df if teamA == "All" else df[df["Team"] == teamA]
    playerA = c1.selectbox("Player A", sorted(poolA["Player"].unique().tolist()), key="playerA")
    rowA = poolA[poolA["Player"] == playerA].iloc[0].copy()

    teamB = c2.selectbox("Team B", teams, index=0, key="teamB")
    poolB = df if teamB == "All" else df[df["Team"] == teamB]
    playerB = c2.selectbox("Player B", sorted(poolB["Player"].unique().tolist()), key="playerB")
    rowB = poolB[poolB["Player"] == playerB].iloc[0].copy()

    # engineered
    for r in (rowA, rowB):
        r["xG_per90"] = 0 if r["90s"] == 0 else r["xG"] / r["90s"]
        r["xAG_per90"] = 0 if r["90s"] == 0 else r["xAG"] / r["90s"]
        r["npxG_per90"] = 0 if r["90s"] == 0 else r["npxG"] / r["90s"]
        r["usage"] = 0 if r["90s"] == 0 else r["Min"] / (r["90s"] * 90)

    sampleA = pd.DataFrame([rowA])[FEATURES]
    sampleB = pd.DataFrame([rowB])[FEATURES]

    predA = float(predict_from_features(sampleA))
    predB = float(predict_from_features(sampleB))

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"### {rowA["Player"]} - {rowA["Team"]}")
        st.metric("Predicted Goals", f"{predA:.2f}", help="HUMAI Prediction")
        st.caption(f"Actual last season: {float(rowA.get("Gls", np.nan)):.0f}")
    with colB:
        st.markdown(f"### {rowB["Player"]} - {rowB["Team"]}")
        st.metric("Predicted Goals", f"{predB:.2f}")
        st.caption(f"Actual last season: {float(rowB.get("Gls", np.nan)):.0f}")

    # Export compare CSV
    if st.button("📤 Export Compare (CSV)"):
        out = pd.DataFrame([
            {"Player": rowA["Player"], "Team": rowA["Team"], "Predicted_Goals": predA},
            {"Player": rowB["Player"], "Team": rowB["Team"], "Predicted_Goals": predB}
        ])
        out_path = os.path.join(REPORTS_DIR, "compare_players_v0_5.csv")
        out.to_csv(out_path, index=False)
        st.success(f"Saved: {out_path}")

with tab2:
    st.subheader("🧪 Scenario Planning (what-if)")

    base_player = st.selectbox("Choose player", sorted(df["Player"].unique().tolist()))
    row = df[df["Player"] == base_player].iloc[0].copy()

    # engineered
    row["xG_per90"] = 0 if row["90s"] == 0 else row["xG"] / row["90s"]
    row["xAG_per90"] = 0 if row["90s"] == 0 else row["xAG"] / row["90s"]
    row["npxG_per90"] = 0 if row["90s"] == 0 else row["npxG"] / row["90s"]
    row["usage"] = 0 if row["90s"] == 0 else row["Min"] / (row["90s"] * 90)

    base_sample = pd.DataFrame([row])[FEATURES]
    base_pred = float(predict_from_features(base_sample))

    st.markdown("**Adjust inputs to simulate improvement/decline:**")
    c1, c2, c3 = st.columns(3)
    xG_delta = c1.slider("xG change (%)", -30, 50, 10, 1)
    PrgR_delta = c2.slider("PrgR change (%)", -30, 50, 10, 1)
    Min_delta = c3.slider("Minutes change(%)", -30, 30, 0, 1)

    sim = row.copy()
    sim["xG"] = row["xG"] * (1 + xG_delta/100)
    sim["PrgR"] = row["PrgR"] * (1 + PrgR_delta/100)
    sim["Min"] = row["Min"] * (1 + Min_delta/100)

    # recompute engineered for scenario
    sim["xG_per90"] = 0 if row["90s"] == 0 else sim["xG"] / row["90s"]
    sim["xAG_per90"] = 0 if row["90s"] == 0 else row["xAG"] / row["90s"] # we let xAG unchanged
    sim["npxG_per90"] = 0 if row["90s"] == 0 else row["npxG"] / row["90s"] # idem
    sim["usage"] = 0 if row["90s"] == 0 else sim["Min"] / (row["90s"] * 90)

    sim_df = pd.DataFrame([sim])[FEATURES]
    sim_pred = float(predict_from_features(sim_df))

    delta = sim_pred - base_pred
    st.metric("Δ Predicted Goals", f"{delta:+.2f}", help="Difference scenario vs base")
    st.success(coach_text(delta))

    # Save scenario
    if st.button("💾 Save Scenario"):
        out = pd.DataFrame([{
            "Player": base_player,
            "Predicted_base": base_pred,
            "Predicted_scenario": sim_pred,
            "Delta": delta,
            "xG_delta_pct": xG_delta,
            "PrgR_delta_pct": PrgR_delta,
            "Min_delta_pct": Min_delta
        }])
        out_path = os.path.join(REPORTS_DIR, "scenario_v0_5.csv")
        if os.path.exists(out_path):
            out.to_csv(out_path, mode="a", header=False, index=False)
        else:
            out.to_csv(out_path, index=False)
        st.success(f"Scenario appended to: {out_path}")

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

    







    