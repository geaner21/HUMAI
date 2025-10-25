import os, json, platform
import joblib
import pandas as pd
import streamlit as st
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timezone

from utils_scores import attach_perf_index # type: ignore

# === PATH SETUP ===
THIS_DIR = os.path.dirname(__file__)
PROJ_DIR = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(PROJ_DIR, "data")
MODELS_DIR = os.path.join(PROJ_DIR, "models")
REPORTS_DIR = os.path.join(PROJ_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "premier-league-player-23-24.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "humai_v0_3_rf.pkl")

# === STREAMLIT CONFIG ===
st.set_page_config(page_title="HUMAI Coach v0.7", page_icon="🏟", layout="wide")
st.title("🏟 HUMAI Coach - v0.7")

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    # fallback pentru lipsă de coloane
    for col in ["Team", "Pos", "xG", "xAG", "npxG", "PrgP", "PrgC", "PrgR", "Min", "Gls", "90s", "Age"]:
        if col not in df.columns:
            df[col] = 0
    return df

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ Missing .pkl model. 'Explain' tab will have limited functionality")
        return None
    return joblib.load(MODEL_PATH)

def safe_div(num, den):
    if hasattr(den, "replace"):
        den = den.replace(0, np.nan)
    elif den == 0:
        den = np.nan
    if hasattr(num, "fillna"):
        return (num / den).fillna(0)
    return 0 if pd.isna(den) else num / den

# === Define features manually (exact same as used in v0.3) ===
FEATURES = [
    "xG", "xAG", "npxG",
    "xG_per90", "xAG_per90", "npxG_per90",
    "PrgP", "PrgC", "PrgR",
    "Min", "90s", "Age",
    "usage"
]

df_raw = load_data()
df = attach_perf_index(df_raw)
model = load_model()

# ========= TABS =========
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧠 Explain", "📈 Drift & Logs"])

# ======================= TAB 1: OVERVIEW ======================
with tab1:
    st.subheader("Performance Index (0-100) + filters")

    teams = ["All"] + sorted(df["Team"].dropna().unique().tolist())
    positions = ["All"] + sorted(df["Pos"].dropna().unique().tolist())

    c1, c2, c3 = st.columns([1,1,2])
    team = c1.selectbox("Team", teams, index=0)
    pos = c2.selectbox("Position", positions, index=0)
    topk = c3.slider("Top K", 5, 30, 10)

    view = df.copy()
    if team != "All": view = view[view["Team"] == team]
    if pos != "All": view = view[view["Pos"] == pos]

    st.metric("Average Score (filtered)", float(view["Performance_Index"].mean()))
    if "Pos" in view.columns and not view.empty:
        st.bar_chart(view.groupby("Pos")["Performance_Index"].mean().sort_values())

    st.subheader("Top performers")
    cols_keep = ["Player", "Team", "Pos", "Gls", "Ast", "xG", "xAG", "PrgR", "Min", "Performance_Index"]
    st.dataframe(
        view.nlargest(topk, "Performance_Index")[cols_keep],
        width="stretch") # în loc de use_container_width=True

    # Export
    out_csv = os.path.join(REPORTS_DIR, "humai_report_v0_7.csv")
    if st.button("📤 Export CSV"):
        view.sort_values("Performance_Index", ascending=False)[cols_keep].to_csv(out_csv, index=False)
        st.success(f"Saved: {out_csv}")

# ================ TAB 2: EXPLAIN (SHAP) ================
with tab2:
    st.subheader("Explainable AI - Why the model predicts that")

    if model is None:
        st.warning("The .pkl model was not found. Train/save the model v0.3 as 'models/humai_v0_3_rf.pkl'.")
    else:
        # Construiește features cerute de pipeline
        work = df_raw.copy()
        for base in ["xG", "xAG", "npxG", "PrgP", "PrgC", "PrgR", "Min", "Gls", "90s", "Age"]:
            if base not in work.columns:
                work[base] = 0

        if "xG_per90" not in work.columns:
            work["xG_per90"] = safe_div(work["xG"], work["90s"])
        if "xAG_per90" not in work.columns:
            # BUG în codul vechi: am împărțit xG/90s; trebuie xAG/90s
            work["xAG_per90"] = safe_div(work["xAG"], work["90s"])
        if "npxG_per90" not in work.columns:
            work["npxG_per90"] = safe_div(work["npxG"], work["90s"])
        if "usage" not in work.columns:
            work["usage"] = safe_div(work["Min"], work["90s"] * 90)

        # cleaning
        needed = [c for c in FEATURES if c in work.columns]
        data_clean = work.dropna(subset=needed + ["Gls"]).copy()
        X_all = data_clean[needed].copy()
        y_all = data_clean["Gls"].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.25, random_state=42
        )

    # === SHAP Explainability (pipeline-aware) ===
    st.markdown("#### SHAP (global feature influence)")

    if not isinstance(model, Pipeline):
        st.error("The loaded model is not a sklearn Pipeline (prep + rf).")
    else:
        rf = model.named_steps.get("rf", None)
        prep = model.named_steps.get("prep", None)
        if rf is None or prep is None:
            st.error("Pipeline does not contain the steps 'prep' and 'rf'. Verify the saved model.")
        else:
            # explainability sampled
            sample_X = X_test.sample(min(200, len(X_test)), random_state=42)
            # transform data through prep (StandardScaler etc.)
            sample_X_t = prep.transform(sample_X) # -> ndarray

            # prepare feature names (same as in FEATURES, the order correspons to the numerical columns)
            feature_names = [c for c in FEATURES if c in sample_X.columns]

            #build TreeExplainer on final estimator
            explainer = shap.TreeExplainer(rf)
            try:
                shap_values = explainer(sample_X_t) # API new SHAP
                values = shap_values.values
            except Exception:
                # compat SHAP older
                values = explainer.shap_values(sample_X_t)

            # bar summary
            fig = plt.figure(figsize=(7, 5))
            shap.summary_plot(values, feature_names=feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            plot_path = os.path.join(REPORTS_DIR, "humai_v0_7_explainability.png")
            plt.savefig(plot_path, bbox_inches="tight")
            st.pyplot(fig)
            st.caption(f"Saved: {plot_path}")

            # mean aboslute SHAP per feature
            mean_shap = np.abs(values).mean(axis=0)
            explain_df = pd.DataFrame({"Feature": feature_names, "Mean_Impact": mean_shap}) \
                            .sort_values("Mean_Impact", ascending=False)
            
            st.markdown("#### Top factors (mean SHAP impact)")
            st.dataframe(explain_df, width="stretch")

            csv_path = os.path.join(REPORTS_DIR, "humai_v0_7_explain_report.csv")
            explain_df.to_csv(csv_path, index=False)
            st.success(f"Explainability report saved: {csv_path}")

            # ✅ Insight text (global)
            def generate_global_insight(df_imp):
                if df_imp is None or df_imp.empty:
                    return "No feature impact data available."
                top = df_imp.head(3)["Feature"].tolist()
                text_map = {
                    "xG": "quality of goal-scoring chances",
                    "xAG": "expected assists and creative output",
                    "npxG": "non-penalty goal contribution",
                    "PrgR": "progressive runs and attacking involvement",
                    "PrgP": "progressive passing influence",
                    "PrgC": "ball carrying and control",
                    "Min": "match consistency and availability",
                    "Age": "player experience and maturity",
                    "usage": "playing time ratio and match fitness",
                    "90s": "time normalization (per 90)"
                }
                desc = ", ".join([text_map.get(f, f) for f in top])
                return f"The model is primarily influenced by {desc}."

        if "explain_df" in locals() and explain_df is not None:
            insight_text = generate_global_insight(explain_df)
            st.info(insight_text)  

    st.markdown("#### Individual player analysis")
    player_list = sorted(df["Player"].dropna().unique().tolist())
    selected_player = st.selectbox("Select player", player_list)

    if selected_player:
        p = df[df["Player"] == selected_player].iloc[0]
        st.write(f"**Team:** {p["Team"]} | **Position:** {p["Pos"]} | **Score:** {p["Performance_Index"]:.1f}")

        msg = []
        if p["Performance_Index"] > 80:
            msg.append("🔥 High-impact performer. Maintain consistency and recovery routines.")
        elif p["Performance_Index"] > 60:
            msg.append("💪 Solid performer with room to grow through creative involvement.")
        else:
            msg.append("⚙️ Developing player. Focus on decision-making and positional awareness.")
        
        # Compare numerically correct (NO ["xG"])
        if float(p["xG"]) < float(df["xG"].mean()):
            msg.append("xG below league average - focus on shot quality and positioning.")
        if float(p["xAG"]) < float(df["xAG"].mean()):
            msg.append("Creative output below league average - work on key passes and final ball.")

        # Relative example at league average (more useful than fixed thresholds)
        league_prgr_mean = float(df["PrgR"].mean())
        if float(p["PrgR"]) < league_prgr_mean:
            msg.append("Increase progressive runs to support attacks.")

        st.success(" ".join(msg))

    st.markdown("#### 🔧 Prescriptive coaching tips")
    league_means = {
        "xG": float(df["xG"].mean()),
        "xAG": float(df["xAG"].mean()),
        "npxG": float(df["npxG"].mean()),
        "PrgR": float(df["PrgR"].mean()),
        "PrgP": float(df["PrgP"].mean()),
        "PrgC": float(df["PrgC"].mean()),
    }
    tips = []
    if float(p["xG"]) < league_means["xG"]:
        tips.append("Improve shot quality: more central shots, reduce low-xG attempts.")
    if float(p["xAG"]) < league_means["xAG"]:
        tips.append("Increase creative output: target through-balls/cutbacks in final third.")
    if "PrgR" in p and float(p["PrgR"]) < league_means["PrgR"]:
        tips.append("Add progressive carries: attack space between lines 3-5 times per half.")
    if "PrgP" in p and float(p["PrgP"]) < league_means["PrgP"]:
        tips.append("Use progressive passes: switch play to overload weak side at least 4 times/match.")
    if float(p["Min"]) < 1200:
        tips.append("Build match fitness: structured minutes ramp-up + recovery protocol.")

    if tips:
        st.write("• " + "\n• ".join(tips))
    else:
        st.write("Player already exceeds league benchmarks on core metrics.")

    retrain_path = os.path.join(MODELS_DIR, "humai_v0_9_rf.pkl")

    if st.button("♻️ Retrain model on current data"):
        # simple RF with tuned params you had
        rf_new = RandomForestRegressor(
            n_estimators=800,
            max_depth=6,
            max_features="sqrt",
            min_samples_split=5,
            random_state=42
        )
        rf_new.fit(X_all, y_all)
        # rebuild a Pipeline-like replacement if prep exists in old model
        if isinstance(model, Pipeline):
            prep = model.named_steps.get("prep", None)
            if prep is not None:
                from sklearn.pipeline import Pipeline as SkPipeline
                model_new = SkPipeline(steps=[("prep", prep), ("rf", rf_new)])
            else:
                model_new = rf_new
        else:
            model_new = rf_new

        joblib.dump(model_new, retrain_path)
        st.success(f" Model retrained and saved to {retrain_path}")
    
    st.markdown("#### 🎛️ Scenario simulator (what-if)")
    if model is not None and not X_all.empty:
        # pick player
        pl = st.selectbox("Simulate for player", player_list, key="sim_player")
        row = data_clean[data_clean["Player"] == pl].iloc[0] if "Player" in data_clean.columns and pl in data_clean["Player"].values else data_clean.iloc[0]

        # sliders (bounded around current values)
        sxg = st.slider("xG (simulated)", 0.0, max(1.0, float(row["xG"]*1.5)), float(row["xG"]), 0.05)
        sxag = st.slider("xAG (simulated)", 0.0, max(1.0, float(row["xAG"]*1.5)), float(row["xAG"]), 0.05)
        spr = st.slider("PrgR (simulated)", 0.0, max(5.0, float(row["PrgR"]*1.5)), float(row["PrgR"]), 0.5)

        sim = row.copy()
        sim["xG"] = sxg
        sim["xAG"] = sxag
        sim["PrgR"] = spr
        # recompute per90 & usage if 90s > 0
        if sim["90s"] > 0:
            sim["xG_per90"] = sim["xG"] / sim["90s"]
            sim["xAG_per90"] = sim["xAG"] / sim["90s"]
            sim["npxG_per90"] = sim["npxG"] / sim["90s"] if sim["90s"] else 0
            sim["usage"] = sim["Min"] / (sim["90s"] * 90)
        sim_X = pd.DataFrame([sim[needed]])

        # pipeline-safe prediction
        try:
            y_hat = model.predict(sim_X)[0]
            st.info(f"Predicted goals (what-if): **{y_hat:.2f}**")
        except Exception as e:
            st.warning(f"Simulation failed: {e}")

    # export insight JSON
    if st.button("💾 Export Insights"):
        insights_path = os.path.join(REPORTS_DIR, "humai_v0_8_insights.json")
        insights = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": insight_text,
            "top_features": explain_df.head(5).to_dict(orient="records")
        }
        with open(insights_path, "w", encoding="utf-8") as f:
            json.dump(insights, f, indent=2)
        st.success(f"Saved interpretability insights: {insights_path}")

# ================= TAB 3: DRIFT & LOGS =================
with tab3:
    st.subheader("Data drift & run logs")

    feats_for_drift = ["xG", "xAG", "npxG", "PrgR", "Min", "Performance_Index"]
    cur = {f: {"mean": float(df[f].mean()), "std": float(df[f].std())}
           for f in feats_for_drift if f in df.columns}
    DRIFT_PATH = os.path.join(REPORTS_DIR, "drift_baseline.json")

    if os.path.exists(DRIFT_PATH):
        base = json.load(open(DRIFT_PATH, "r"))
        zs = []
        for k in base:
            if k in cur and base[k]["std"] > 1e-6:
                zs.append(abs(cur[k]["mean"] - base[k]["mean"]) / base[k]["std"])
        z = float(np.mean(zs)) if zs else 0.0
        st.info(f"Drift score (mean z): {z:.2f}")
        if z > 0.8:
            st.warning("⚠️ Data drift detected - consider retraining.")
    else: 
        json.dump(cur, open(DRIFT_PATH, "w"))
        st.success("Drift baseline saved.")

    # run log
    run_log = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "rows": int(df.shape[0]),
        "features": sorted(list(set(df.columns))),
        "python": platform.python_version()
    }
    json.dump(run_log, open(os.path.join(REPORTS_DIR, "run_log_v0_7.json"), "w"))
    st.caption("Run log written to reports/run_log_v0_7.json")