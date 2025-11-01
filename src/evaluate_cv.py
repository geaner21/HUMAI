# src/evaluate_cv.py
import os, json, datetime as dt
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "premier-league-player-23-24.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "humai_v0_9_rf.pkl")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
OUT_PATH = os.path.join(REPORTS_DIR, "model_eval_history.json")

def _safe_per90(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # if 90s is missing, we create it with 0 as to not break; then we treat the division
    if "90s" not in df.columns:
        df["90s"] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        df["xG_per90"] = (df["xG"] / df["90s"])
        df["xAG_per90"] = (df["xAG"] / df["90s"])
        df["npxG_per90"] = (df["npxG"] / df["90s"])
        df["usage"] = (df["Min"] / (df["90s"] * 90.0))

    # we replace inf/-inf and NaN
    for c in ["xG_per90","xAG_per90","npxG_per90","usage"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)

    return df

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)

# (optional but recommmended) we bring out the rows where 90s = 0 (per90 does not make sense)
df = df.copy()
if "90s" in df.columns:
    df = df[df["90s"] > 0].copy()

# assuring the base columns
for col in ["xG","xAG","npxG","PrgP","PrgC","PrgR","Min","90s","Age","Gls"]:
    if col not in df.columns:
        df[col] = 0.0

# Feature engineering safe
df = _safe_per90(df)

features = ["xG","xAG","npxG","xG_per90","xAG_per90","npxG_per90","PrgP","PrgC","PrgR","Min","90s","Age","usage"]
target = "Gls"

# drop NaN on necessary columns (after we cleaned per90)
df = df.dropna(subset=features + [target]).copy()

# cast at float64 (we dodge "value too large for float32")
X = df[features].astype("float64")
y = df[target].astype("float64")

# === LOAD MODEL ===
model = joblib.load(MODEL_PATH)

# === EVALUATE ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2", error_score="raise")
mse_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error", error_score="raise")

result = {
    "timestamp": dt.datetime.now().isoformat(),
    "model_path": MODEL_PATH,
    "mean_R2": float(r2_scores.mean()),
    "std_R2": float(r2_scores.std()),
    "mean_MSE": float(mse_scores.mean()),
    "std_MSE": float(mse_scores.std()),
    "n_samples": int(len(df)),
    "features_used": features
}

# === SAVE HISTORY ===
history = []
if os.path.exists(OUT_PATH):
    try:
        history = json.load(open(OUT_PATH, "r"))
    except Exception:
        history = []

history.append(result)
json.dump(history[-20:], open(OUT_PATH, "w"), indent=2)

print(f"[✅] Evaluation done for {MODEL_PATH}")
print(f"R² mean: {result['mean_R2']:.4f} ± {result['std_R2']:.4f}")
print(f"MSE mean: {result['mean_MSE']:.4f} ± {result['std_MSE']:.4f}")
print(f"Saved to: {OUT_PATH}")