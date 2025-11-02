# src/app_streamlit_v0_9.py
import os, base64, time, json
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

print("ENV:", os.getenv("HUMAI_ENV"))
print("API:", os.getenv("HUMAI_API_URL"))
print("REPORTS:", os.getenv("HUMAI_REPORTS_DIR"))

# ========== CONFIG FIRST LINE ==========
st.set_page_config(page_title="HUMAI Dashboard v0.9", page_icon="🏟", layout="wide")

# ========== CONSTANTS / PATHS ==========
PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "humai_logo.png")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "premier-league-player-23-24.csv")
REPORTS_DIR = os.getenv("HUMAI_REPORTS_DIR", os.path.join(PROJ_DIR, "reports"))
LOG_PATH = os.path.join(REPORTS_DIR, "api_requests_log.json")
COMPARE_REPORT = os.path.join(REPORTS_DIR, "compare_metrics.json")
os.makedirs(REPORTS_DIR, exist_ok=True)

APP_VERSION = "v0.9" 
API_URL = os.getenv("HUMAI_API_URL", "http://127.0.0.1:8000")

# ========== HUMAI CLIENT ==========
def _url(path: str) -> str:
    base = os.getenv("HUMAI_API_URL", API_URL)
    return f"{base.rstrip('/')}{path}"

def api_get(path: str, timeout=5):
        r = requests.get(_url(path), timeout=timeout)
        r.raise_for_status()
        return r.json()
    
def api_post(path: str, payload=None, timeout=10):
        r = requests.post(_url(path), json=payload or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()

def health():
    return api_get("/health")

def health_extended():
    return api_get("/health_extended")

def expected_features():
    return api_get("/model/expected_features")

def analytics():
    return api_get("/analytics")

def csv_head():
    return api_get("/debug/csv_head")

def compare(save_report=True):
    return api_get(f"/compare?save_report={'true' if save_report else 'false'}")

def evaluate_cv(k=5):
    return api_get(f"/evaluate_cv?k={int(k)}")

def retrain():
    return api_post("/retrain")

def promote():
    return api_post("/promote")

def predict(payload: dict):
    return api_post("/predict", payload)

def predict_batch(items: list[dict]):
    return api_post("/predict_batch", {"items": items})

# ========== UI: HEADER ==========
def render_logo(path: str, height_px: int = 44):
    if not os.path.exists(path):
        st.markdown("### **HUMAI**")
        return
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"<img src='data:image/png;base64,{b64}' style='height:{height_px}px;'>",
        unsafe_allow_html=True,
    )

top_left, top_right = st.columns([1, 6])
with top_left:
    render_logo(LOGO_PATH, 44)
with top_right:
    st.markdown(f"<h1 style='margin:8px 0 0 0;'>HUMAI Dashboard {APP_VERSION}</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container { padding-top: 0.75rem; }
</style>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.subheader("API Settings")
    api_default = os.getenv("HUMAI_API_URL", API_URL)
    user_api_url = st.text_input("HUMAI_API_URL", api_default, help="Set ENV var HUMAI_API_URL for persistence.")
    if user_api_url != os.getenv("HUMAI_API_URL"):
        # we do not change the environment; we only update the internal base for session
        API_URL = user_api_url

    st.write("---")
    if st.button("🔎 Health check"):
        try:
            h = health()
            st.success(h)
            he = health_extended()
            st.info(he)
        except Exception as e:
            st.error(f"Health failed: {e}")

    auto_refresh = st.checkbox("Auto-refresh footer (30s)", value=False)

    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=30_000, limit=100000, key="footer_refresh")
        except Exception:
            pass # without optional dependance, you don't do nothing

# ========== LOADING DATA ==========
@st.cache_data()
def load_data(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df
    
df_raw = pd.DataFrame()
try:
    df_raw = load_data(DATA_PATH)
    st.success(f"✅ {len(df_raw)} rows loaded from {DATA_PATH}")
except Exception as e:
    st.warning(f"The dataset could not be loaded: {e}")

# Utility: performance index (0-100)
def compute_performance_index(df_in: pd.DataFrame) -> pd.DataFrame:
    dfw = df_in.copy()
    for col in ["xG", "xAG", "PrgR", "Min"]:
        if col not in dfw.columns:
            dfw[col] = 0.0
    prgr_max = float(dfw["PrgR"].replace(0, np.nan).max() or 1.0)
    raw = (
        dfw["xG"] * 0.40 +
        dfw["xAG"] * 0.30 +
        (dfw["PrgR"] / prgr_max) * 0.20 * 10 +
        (dfw["Min"] / 3420.0) * 0.10 *10
    )
    lo, hi = float(raw.min() or 0.0), float(raw.max() or 1.0)
    if hi == lo:
        dfw["Performance_Index"] = 50.0
    else:
        dfw["Performance_Index"] = ((raw - lo) / (hi - lo) * 100).round(2)
    return dfw

# ========== TABS ==========
tab1, tab2, tab3 = st.tabs(["📈 Overview", "🎯 Predict", "🛰️ API Analytics"])

# ========== TAB 1: OVERVIEW ==========
with tab1:
    st.header("📈 Performance Index (0-100) + Filters")


    
    df_view = df_raw.copy()
    if not df_view.empty and "Performance_Index" not in df_view.columns:
        df_view = compute_performance_index(df_view)

    c1, c2, c3 = st.columns([1, 1, 2])
    teams = ["All"] + (sorted(df_view["Team"].dropna().astype(str).unique().tolist()) if "Team" in df_view.columns else [])
    positions = ["All"] + (sorted(df_view["Pos"].dropna().astype(str).unique().tolist()) if "Pos" in df_view.columns else [])

    team_sel = c1.selectbox("Team", teams, index=0)
    pos_sel = c2.selectbox("Position", positions, index=0)
    topk = c3.slider("Top K", 5, 50, 10)

    filt = df_view.copy()
    if team_sel != "All" and "Team" in filt.columns:
        filt = filt[filt["Team"] == team_sel]
    if pos_sel != "All" and "Pos" in filt.columns:
        filt = filt[filt["Pos"] == pos_sel]

    avg_score = float(filt["Performance_Index"].mean()) if not filt.empty else 0.0
    st.metric("Average Score (filtered)", f"{avg_score:.2f}")

    if "Pos" in filt.columns and not filt.empty:
        st.bar_chart(filt.groupby("Pos")["Performance_Index"].mean().sort_values())

    st.subheader("🏅 Top Performers")
    cols_keep = [c for c in ["Player", "Team", "Pos", "Gls", "Ast", "xG", "xAG", "PrgR", "Min", "Performance_Index"] if c in filt.columns]
    if cols_keep:
        top_tbl = filt.nlargest(topk, "Performance_Index")[cols_keep]
        st.dataframe(top_tbl, width="stretch")
        st.download_button(
            "⬇️ Download Top CSV",
            top_tbl.to_csv(index=False).encode("utf-8"),
            file_name=f"humai_overview_top_{topk}.csv",
            mime="text/csv"
        )
    else:
        st.info("I did not find the standard columns for the table with top players (Player/Team/Pos/...).") 

    st.divider()
    with st.expander("🔎 Quick insights"):
        if not filt.empty:
            st.write(
                f"- Filtered players: **{len(filt)}**\n"
                f"- Min/Max score: **{float(filt['Performance_Index'].min()):.1f}** / **{float(filt['Performance_Index'].max()):.1f}**"     
            )
        else:
            st.write("There is no recording after filters.")

# ========== TAB 2: PREDICT ==========
with tab2:
    st.header("🎯 Predict Player Goals")

    with st.form(key="predict_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            xG = st.number_input("xG (Expected Goals)", value=5.0, min_value=0.0, key="xg")
            xAG = st.number_input("xAG (Expected Assisted Goals)", value=3.0, min_value=0.0, key="xag")
            npxG = st.number_input("npxG (Non-penalty xG)", value=4.5, min_value=0.0, key="npxg")
        with col2:
            PrgP = st.number_input("Progressive Passes (PrgP)", value=15.0, min_value=0.0, key="prgp")
            PrgC = st.number_input("Progressive Carries (PrgC)", value=12.0, min_value=0.0, key="prgc")
            PrgR = st.number_input("Progressive Receives (PrgR)", value=20.0, min_value=0.0, key="prgr")
        with col3:
            Min = st.number_input("Minutes Played", value=1800.0, min_value=0.0, key="min")
            Age = st.number_input("Age", value=25.0, min_value=0.0, key="age")
            ninety_s = st.number_input("90s Played", value=20.0, min_value=0.1, key="n90s")
        
        submitted = st.form_submit_button("🚀 Predict")
    
    if submitted:
        payload = {
            "xG": xG, "xAG": xAG, "npxG": npxG,
            "PrgP": PrgP, "PrgC": PrgC, "PrgR": PrgR,
            "Min": Min, "Age": Age, "ninety_s": ninety_s
        }
        try:
            res = predict(payload)
            st.session_state["last_pred"] = res.get("predicted_goals")
            st.success(f"Predicted Goals: **{st.session_state['last_pred']}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        
    # maintain the last result at rerun (if exists)
    if "last_pred" in st.session_state and not submitted:
        st.info(f"Last predicted: **{st.session_state['last_pred']}**")

    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None

    if st.button("🚀 Predict"):
        try:
            result = predict(payload)
            st.session_state["last_prediction"] = result
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if st.session_state["last_prediction"]:
        st.success(f"Predicted Goals: **{st.session_state['last_prediction']['predicted_goals']}**")

    st.divider()
    st.subheader("Batch Predict (CSV)")

    up = st.file_uploader("Choose a CSV with the columns: xG, xAG, npxG, PrgP, PrgC, PrgR, Min, Age, ninety_s", type=["csv"])
    if up:
        try:
            dfb = pd.read_csv(up)
            st.dataframe(dfb.head(), use_container_width=True)
            if st.button("📊 Run Batch Predictions"):
                items = dfb.to_dict(orient="records")
                res = predict_batch(items)
                # API will return {"count": N, "results": [ {predicted_goals:..} ..]}
                if isinstance(res, dict) and "results" in res:
                    preds = []
                    for r in res["results"]:
                        preds.append(r.get("predicted_goals", None))
                    dfb["Predicted_Goals"] = preds
                elif isinstance(res, list):
                    # old fallback (simple list)
                    dfb["Predicted_Goals"] = [r.get("predicted_goals", None) for r in res]
                else:
                    raise ValueError(f"Unexpected answer format: {type(res)}")
                
                st.dataframe(dfb, use_container_width=True)
                st.download_button("⬇️ Download results", dfb.to_csv(index=False).encode("utf-8"), "predictions.csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
            
# ========== TAB 3: API ANALYTICS ==========
with tab3:
    st.header("🛰️ API Analytics")

    # Compare metrics section (if exists)
    st.subheader("🏋️ Model Metrics Comparison")
    if os.path.exists(COMPARE_REPORT):
        try:
            data = json.load(open(COMPARE_REPORT, "r"))
            v09 = data.get("v0.9", {}) or data.get("v0_9", {})
            v10 = data.get("v1.0", {}) or data.get("v1_0", {})
            timestamp = data.get("timestamp", "N/A")
            winner = data.get("winner", "N/A")
            features = data.get("features_used", [])   

            st.caption(f"Last comparison: **{timestamp}**")
            a, b = st.columns(2)
            a.metric("🏷️ Best model", winner)
            b.metric("🔢 Total features", len(features))   

            dfm = pd.DataFrame([
                {"Model": "v0.9", "MSE": v09.get("MSE", None), "R²": v09.get("R2", None)},
                {"Model": "v1.0", "MSE": v10.get("MSE", None), "R²": v10.get("R2", None)},
            ])
            st.dataframe(dfm, width="stretch") 

            
            c1, c2 = st.columns(2)
            with c1:
                st.caption("R² Score")
                st.bar_chart(dfm.set_index("Model")[["R²"]])
            with c2:
                st.caption("MSE (Mean Squared Error)")
                st.bar_chart(dfm.set_index("Model")[["MSE"]])

            with st.expander("🔍 Features used during training"):
                st.code("\n".join(features), language="text")
        except Exception as e:
            st.warning(f"I could not read compare_metrics.json: {e}")
    else: 
        st.info("I did not find the comparison report. Run `compare?save_report=true` in API.")
    
    st.divider()
    st.subheader("📜 Requests log")

    if not os.path.exists(LOG_PATH):
        st.info("No logs yet.")
    else:
        try:
            logs = json.load(open(LOG_PATH, "r", encoding="utf-8"))
            if not isinstance(logs, list):
                logs = [logs]

            # robust normalize, we use '.' as separator    
            df = pd.json_normalize(logs, sep=".")

            # timestamp
            df["ts"] = pd.to_datetime(df.get("timestamp"), errors="coerce")

            df["pred"] = pd.to_numeric(df.get("result.predicted_goals", np.nan), erorrs="coerce")
            df["pred"] = df["pred"].fillna(0.0)
            if "result.latency_ms" in df.columns:
                df["latency_ms"] = pd.to_numeric(df["result.latency_ms"], erorrs="coerce")
            else:
                df["latency_ms"] = np.nan

            # predicted goals - trying first the plate column,
            # then extract from 'result' dict
            if "result.predicted_goals" in df.columns:
                df["pred"] = pd.to_numeric(df["result.predicted_goals"], errors="coerce")
            elif "result" in df.columns:
                df["pred"] = df["result"].apply(
                    lambda x: x.get("predicted_goals") if isinstance(x, dict) else np.nan
                ).astype(float)
            else:
                df["pred"] = np.nan

            # latency (if exists with different names)
            lat_cols = [c for c in df.columns if c.endswith("latency_ms")]
            if lat_cols:
                df["latency_ms"] = pd.to_numeric(df[lat_cols[0]], errors="coerce")
            else:
                df["latency_ms"] = np.nan

            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Requests", len(df))
            c2.metric("Avg goals", f"{df['pred'].mean():.2f}")
            c3.metric("Max goals", f"{df['pred'].max():.2f}")
            c4.metric("Avg latency",
                       f"{df['latency_ms'].mean():.1f} ms" if df["latency_ms"].notna().any() else "-"
                       )
            # Charts
            st.subheader("Predictions over time")
            ts = df.set_index("ts")[["pred"]].sort_index().dropna()
            if not ts.empty:
                st.line_chart(ts, height=220)
            else:
                st.caption("No time-series data yet.")

            st.subheader("Prediction distribution")
            if df["pred"].notna().any():
                st.bar_chart(df["pred"].round(1).value_counts().sort_index(), height=220)
            else:
                st.caption("No predictions to plot yet.")
            
            # Recent logs table (columns can be missing -> we use .get)
            st.subheader("Recent API requests")
            show_cols = []
            for cand in ["ts", "endpoint", "payload.xG", "payload.xAG", "payload.Min", "pred", "latency_ms"]:
                if cand in df.columns:
                    show_cols.append(cand)
            st.dataframe(
                df.sort_values("ts", ascending=False).head(30)[show_cols], 
                width="stretch"
            )

        except Exception as e:
            st.error(f"Log parsing failed (handled): {e}")
    
# ========== FOOTER (version + API status) ==========
def ping_api() -> tuple[bool, int, str]:
    try:
        t0 = time.time()
        r = requests.get(_url("/health_extended"), timeout=2.5)
        dt_ms = int((time.time() - t0) * 1000)
        if r.ok:
            j = r.json()
            last_trained = j.get("last_trained", "-")
            size = j.get("model_size_mb", "-")
            return True, dt_ms, f"{size} MB · {last_trained}"
        return False, dt_ms, "-"
    except Exception:
        return False, 0, "-"
    
st.markdown("""
<style>
.humai-footer{
    position: fixed; right: 14px; bottom: 10px;
    background: rgba(20,20,20,0.65); color: #e5e7eb;
    padding: 6px 10px; border-radius: 8px; font-size: 12px;
    backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
    z-index: 9999;
}
.status-dot{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; background:#9ca3af; }
.status-ok { background:#22c55e; }
.status-bad{ background:#ef4444; }
</style>
""", unsafe_allow_html=True)

ok, ms, extra = ping_api()
dot = "status-ok" if ok else "status-bad"
st.markdown(
    f"""
    <div class="humai-footer">
        <span class="status-dot {dot}"></span>
        <b>API</b> {'OK' if ok else 'DOWN'} · {ms} ms &nbsp;|&nbsp; <b>{APP_VERSION}</b><br/>
        <span style="opacity:.8">{extra}</span>
    </div>
    """,
    unsafe_allow_html=True
)