# HUMAI API v1.0
# Serviciu pentru predicții, evaluare și managementul modelelor

import os
import json
import joblib
import datetime as dt
import pandas as pd
import numpy as np
import shutil
import time, uuid
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_score
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

print("ENV:", os.getenv("HUMAI_ENV"))
print("API:", os.getenv("HUMAI_API_URL"))
print("REPORTS:", os.getenv("HUMAI_REPORTS_DIR"))

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assure xG_per90 / xAG_per90 / npxG_per90 / usage exactly as in training."""
    df = df.copy()

    # necessary base columns
    for base in ["xG", "xAG", "npxG", "Min", "90s"]:
        if base not in df.columns:
            # if they are totally missing from CSV, we create with 0 as to not break
            df[base] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        if "xG_per90" not in df.columns:
            df["xG_per90"] = (df["xG"] / df["90s"]).replace([np.inf, -np.inf], 0).fillna(0)
        if "xAG_per90" not in df.columns:
            df["xAG_per90"] = (df["xAG"] / df["90s"]).replace([np.inf, -np.inf], 0).fillna(0)
        if "npxG_per90" not in df.columns:
            df["npxG_per90"] = (df["npxG"] / df["90s"]).replace([np.inf, -np.inf], 0).fillna(0)
        if "usage" not in df.columns:
            df["usage"] = (df["Min"] / (df["90s"] * 90)).replace([np.inf, -np.inf], 0).fillna(0)

    # safe types
    for c in ["xG_per90","xAG_per90","npxG_per90","usage"]:
        df[c] = df[c].astype(float)  

    return df  

def _sanitize_payload(d: dict) -> dict:
    """Correct wrong names and differences of capitalization from payload"""
    alias = {
        "xaG": "xAG", "xag": "xAG",
        "xG": "xG", "npxg": "npxG",
        "prgp": "PrgP", "prgc": "PrgC", "prgr": "PrgR",
        "min": "Min", "age": "Age",
        "ninety_s": "ninety_s",
    }
    out = {}
    for k, v in d.items():
        if k in alias:
            out[alias[k]] = v
        elif k.lower() in alias:
            out[alias[k.lower()]] = v
        else:
            out[k] = v
    return out


# now you can access them:
env = os.getenv("HUMAI_ENV")
api_url = os.getenv("HUMAI_API_URL")
secret = os.getenv("HUMAI_SECRET_KEY")

print(f"Running in {env} mode. API={api_url}")
      
HUMAI_ENV = os.getenv("HUMAI_ENV", "dev")
SECRET_KEY = os.getenv("HUMAI_SECRET_KEY", "default_key")
API_URL = os.getenv("HUMAI_API_URL", "http://127.0.0.1:8000")
print(f"✅ Environment: {HUMAI_ENV}")

# === PATH SETUP ===
THIS_DIR = os.path.dirname(__file__)
PROJ_DIR = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(PROJ_DIR, "data")
MODELS_DIR = os.path.join(PROJ_DIR, "models")
REPORTS_DIR = os.getenv("HUMAI_REPORTS_DIR", os.path.join(PROJ_DIR, "reports"))
os.makedirs(REPORTS_DIR, exist_ok=True) 

MODEL_CURRENT = os.path.join(MODELS_DIR, "humai_v0_9_rf.pkl")
MODEL_NEW = os.path.join(MODELS_DIR, "humai_v1_0_rf.pkl")
CSV_PATH = os.path.join(DATA_DIR, "premier-league-player-23-24.csv")

REQUIRED_BASE = ["xG", "xAG", "npxG", "PrgP", "PrgC", "PrgR", "Min", "90s", "Age"]
TRAIN_FEATURES = [
    "xG", "xAG", "npxG",
    "xG_per90", "xAG_per90", "npxG_per90",
    "PrgP", "PrgC", "PrgR",
    "Min", "90s", "Age", "usage"
]
TARGET = "Gls"

def _load_csv(CSV_PATH: str) -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    return df

def _build_training_frame(CSV_PATH: str) -> tuple[pd.DataFrame, pd.Series]:
    df = _load_csv(CSV_PATH).copy()

    # verify the minimum base
    missing = [c for c in REQUIRED_BASE + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    
    # feature engineering (identical with /predict)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["xG_per90"] = (df["xG"] / df["90s"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["xAG_per90"] = (df["xAG"] / df["90s"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["npxG_per90"] = (df["npxG"] / df["90s"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["usage"] = (df["Min"] / (df["90s"] * 90)).replace([np.inf, -np.inf], 0).fillna(0)

    # keep only valid rows
    df2 = df.dropna(subset=TRAIN_FEATURES + [TARGET]).copy()

    X = df2[TRAIN_FEATURES].astype(float)
    y = df2[TARGET].astype(float)
    if X.empty:
        raise ValueError("After preprocessing, X is empty. Check CSV values and required columns.")
    return X, y

# === Accepted features superset (compat with old/new models) ===
FEATURES_SUPERSET = [
    "xG", "xAG", "npxG",
    "xG_per90", "xAG_per90", "npxG_per90",
    "PrgP", "PrgC", "PrgR",
    "Min", "90s", "Age", "usage"
]

def _columns_from_pipeline(m):
        DEFAULT = [
            "xG", "xAG", "npxG",
            "xG_per90", "xAG_per90", "npxG_per90",
            "PrgP", "PrgC", "PrgR",
            "Min", "90s", "Age", "usage"
        ]
        try:
            if isinstance(m, SkPipeline):
                prep = m.named_steps.get("prep")
                if isinstance(prep, ColumnTransformer):
                    trfs = getattr(prep, "transformers_", None) or getattr(prep, "transformers", None) or []
                    cols = []
                    for _, _, colsel in trfs:
                        if colsel is None:
                            continue
                        if isinstance(colsel, (list, tuple)):
                            cols += [str(c) for c in colsel]
                        else:
                            cols.append(str(colsel))
                    cols = [c for c in cols if c and c != "remainder"]
                    cols = sorted(set(cols))
                    if cols:
                        return cols
        except Exception:
            pass
        return DEFAULT

def _predict_dict(payload: dict) -> float:
    """Pure function used by both /predict and /predict_batch."""
    m = get_model()
    if m is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    # normalize keys (exactly for "xAG" etc.)
    payload = _sanitize_payload(payload)

    # build df from payload
    df = pd.DataFrame([payload])

    # require ninety_s
    if "ninety_s" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing field 'ninety_s'")
    
    # rename ninety_s => 90s
    df["90s"] = float(df.pop("ninety_s").iloc[0])

    # assure that all necessary bases exist (fallback 0.0)
    for c in ["xG", "xAG", "npxG", "PrgP", "PrgC", "PrgR", "Min", "Age", "90s"]:
        if c not in df.columns:
            df[c] = 0.0

    # ---- CLAMP (edge-case guardrails) ----
    df["xG"] = df["xG"].clip(0, 60)
    df["xAG"] = df["xAG"].clip(0, 60)
    df["npxG"] = df["npxG"].clip(0, 60)
    for c in ["PrgP","PrgC","PrgR"]:
        df[c] = df[c].clip(0, 80)
    df["Min"] = df["Min"].clip(0, 3420)
    df["Age"] = df["Age"].clip(15, 40)
    df["90s"] = df["90s"].clip(0.1, 38)

    # engineered features
    with np.errstate(divide="ignore", invalid="ignore"):
        df["xG_per90"] = (df["xG"] / df["90s"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["xAG_per90"] = (df["xAG"] / df["90s"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["npxG_per90"] = (df["npxG"] / df["90s"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["usage"] = (df["Min"] / (df["90s"] * 90)).replace([np.inf, -np.inf], 0).fillna(0)

    # columns expected by model
    expected_cols = _columns_from_pipeline(m)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[expected_cols]
    pred = float(m.predict(X)[0])
    return round(max(0.0, pred), 2)

# === FastAPI init ===
app = FastAPI(title="HUMAI Coach API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # pune o listă fixă în producție
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load default model ===
MODEL_CURRENT = os.path.join(MODELS_DIR, "humai_v0_9_rf.pkl")

@lru_cache()
def get_model():
    try:
        return joblib.load(MODEL_CURRENT)
    except Exception:
        return None

@app.get("/health")
def health():
    m = get_model()
    return {
        "status": "ok",
        "model_loaded": m is not None,
        "model_type": str(type(m)) if m else None
    }

@app.get("/health_extended")
def health_extended():
    m = get_model() 
    info = {
        "status": "ok",
        "model_loaded": m is not None,
        "model_path": MODEL_CURRENT,
        "model_type": str(type(m)) if m else None,
        "model_size_mb": None,
        "last_trained": None
    }
    try:
        if os.path.exists(MODEL_CURRENT):
            size_bytes = os.path.getsize(MODEL_CURRENT)
            info["model_size_mb"] = round(size_bytes / (1024 * 1024), 2)
            mod_time = os.path.getmtime(MODEL_CURRENT)
            info["last_trained"] = dt.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
        else:
            info["status"] = "missing_model_file"
    except Exception as e:
        info["status"] = "error"
        info["error"] = str(e)
        info["model_version"] = "v1.0" if os.path.basename(MODEL_CURRENT).startswith("humai_v1_0") else "v0.9"
    return info

@app.get("/model/expected_features")
def model_expected_features():
    m = get_model()
    if m is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    # "official" fallback - EXACTLY the columns waited for preprocessing + model
    FEATURES = [
        "xG", "xAG", "npxG",
        "xG_per90", "xAG_per90", "npxG_per90",
        "PrgP", "PrgC", "PrgR",
        "Min", "90s", "Age", "usage"
    ]

    try:
        if isinstance(m, SkPipeline):
            prep = m.named_steps.get("prep")
            if isinstance(prep, ColumnTransformer):
                cols = []

                # If it is fitted, has .transformers_; otherwise, .transformers
                trfs = getattr(prep, "transformers_", None) or getattr(prep, "transformers", None) or []
                for name, trans, colsel in trfs:
                    if colsel is None:
                        continue
                    if isinstance(colsel, (list, tuple)):
                        cols += [str(c) for c in colsel]
                    else:
                        cols.append(str(colsel))

                # Cleaning & unique
                cols = [c for c in cols if c and c != "remainder"]
                cols = sorted(set(cols))
                if cols:
                    return {"expected_features": cols, "source": "column_transformer"}
    except Exception as e:
        return {"expected_features": FEATURES, "note": f"fallback due to: {e}"}
    
    return {"expected_features": FEATURES, "note": "fallback (no pipeline/prep detected)"}

# === Define request schema ===
class PlayerInput(BaseModel):
    xG: float
    xAG: float
    npxG: float
    PrgP: float
    PrgC: float
    PrgR: float
    Min: float
    Age: float
    ninety_s: float

class PredictResponse(BaseModel):
    predicted_goals: float

class PlayerBatch(BaseModel):
    items: List[PlayerInput]

@app.post("/predict", response_model=PredictResponse)
def predict_player(data: PlayerInput, request: Request):
    try:
        req_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        raw = _sanitize_payload(data.dict())
        pred = _predict_dict(raw)

        lat_ms = round((time.perf_counter() - t0) * 1000, 1)
        log_api_event(
            endpoint="/predict",
            payload=raw,
            result={
                "predicted_goals": float(pred),
                "model_version": "v1.0" if os.path.basename(MODEL_CURRENT).startswith("humai_v1_0") else "v0.9",
                "latency_ms": float(lat_ms),
            },
            status="ok",
        )

        if request.query_params.get("debug") == "1":
            m = get_model()
            used = _columns_from_pipeline(m)
            dbg = pd.DataFrame([raw])
            dbg["90s"] = dbg.pop("ninety_s")
            with np.errstate(divide="ignore", invalid="ignore"):
                dbg["xG_per90"] = (dbg["xG"] / dbg["90s"]).replace([np.inf, -np.inf], 0).fillna(0)
                dbg["xAG_per90"] = (dbg["xAG"] / dbg["90s"]).replace([np.inf, -np.inf], 0).fillna(0)
                dbg["npxG_per90"] = (dbg["npxG"] / dbg["90s"]).replace([np.inf, -np.inf], 0).fillna(0)
                dbg["usage"] = (dbg["Min"] / (dbg["90s"] * 90)).replace([np.inf, -np.inf], 0).fillna(0)
            for c in used:
                if c not in dbg.columns: 
                    dbg[c] = 0.0
            return {
                "request_id": req_id,
                "predicted_goals": pred,
                "used_features": used,
                "latency_ms": lat_ms,
                "debug_X": dbg[used].iloc[0].to_dict()
            }
        
        return {"request_id": req_id, "predicted_goals": pred, "latency_ms": lat_ms}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
@app.post("/predict_batch")
def predict_batch(payload: PlayerBatch, request: Request):
    results =[]
    for idx, item in enumerate(payload.items):
        t0 = time.perf_counter()
        req_id = str(uuid.uuid4())
        try:
            raw = _sanitize_payload(item.dict())
            pred = _predict_dict(raw)
            lat_ms = round((time.perf_counter() - t0) * 1000, 1)

            # log for each item from batch
            log_api_event(
            endpoint="/predict",
            payload=raw,
            result={
                "predicted_goals": float(pred),
                "model_version": "v0.9",
                "latency_ms": float(lat_ms),
            },
            status="ok",
        )

            results.append({
                "idx": idx,
                "request_id": req_id,
                "predicted_goals": pred,
                "latency_ms": lat_ms
            })
        except HTTPException as e:
            results.append({"idx": idx, "error": e.detail})
        except Exception as e:
            results.append({"idx": idx, "error": str(e)})

    return {"count": len(results), "results": results}
    
# === Evaluate models ===
@app.get("/compare")
def compare_models(save_report: bool = True):
    m = get_model()
    if m is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=500, detail=f"Missing CSV: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    df = _add_engineered_features(df)

    target = "Gls"
    features = [
        "xG", "xAG", "npxG", "xG_per90", "xAG_per90", "npxG_per90",
        "PrgP", "PrgC", "PrgR", "Min", "90s", "Age", "usage"
    ]
    features_present = [c for c in features if c in df.columns]
    if target not in df.columns:
        raise HTTPException(status_code=500, detail="CSV missing target column 'Gls'")

    df = df.dropna(subset=features_present + [target]).copy()
    X, y = df[features_present], df[target]

    model_old = joblib.load(MODEL_CURRENT)
    preds_old = model_old.predict(X)

    model_new = joblib.load(MODEL_NEW) if os.path.exists(MODEL_NEW) else None
    preds_new = model_new.predict(X) if model_new else None

    mse_old = mean_squared_error(y, preds_old)
    r2_old = r2_score(y, preds_old)

    result = {
        "v0.9": {"MSE": float(mse_old), "R2": float(r2_old)},
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "features_used": features_present
    }
    if preds_new is not None:
        mse_new = mean_squared_error(y, preds_new)
        r2_new = r2_score(y, preds_new)
        result["v1_0"] = {"MSE": float(mse_new), "R2": float(r2_new)}
        result["winner"] = "v1.0" if r2_new > r2_old else "v0.9"
    else:
        result["note"] = "Model v1.0 not found."

    if save_report:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        out_path = os.path.join(REPORTS_DIR, "compare_metrics.json")
        try:
            json.dump(result, open(out_path, "w"), indent=2)
        except Exception as e:
            print(f"[WARN] Could not save compare_metrics.json: {e}")

    return result

# === Retrain endpoint ===
@app.post("/retrain")
def retrain_model():
    m = get_model()
    if m is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=500, detail=f"Missing CSV: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    df = _add_engineered_features(df)

    target = "Gls"
    features = [
        "xG", "xAG", "npxG",
        "xG_per90", "xAG_per90", "npxG_per90",
        "PrgP", "PrgC", "PrgR",
        "Min", "90s", "Age", "usage"
    ]
    features_present = [c for c in features if c in df.columns]
    if target not in df.columns:
        raise HTTPException(status_code=500, detail="CSV missing target column 'Gls'")
    
    df = df.dropna(subset=features_present + [target]).copy()
    X, y = df[features_present], df[target]

    model_new = RandomForestRegressor(
        n_estimators=1000, max_depth=8, max_features="sqrt",
        min_samples_split=3, random_state=42
    )
    model_new.fit(X, y)
    joblib.dump(model_new, MODEL_NEW)

    return {
        "status": "retrained",
        "saved_as": MODEL_NEW,
        "train_rows": int(df.shape[0]),
        "features_used": features_present
    }

LOG_PATH = os.path.join(REPORTS_DIR, "api_requests_log.json")

def log_api_event(endpoint: str, payload: dict, result: dict, status: str = "ok"):
    event = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "endpoint": endpoint,
        "payload": payload,
        "result": result,
        "status": status
    }
    try:
        if os.path.exists(LOG_PATH):
            existing = json.load(open(LOG_PATH))
        else:
            existing = []
        existing.append(event)
        json.dump(existing[-100:], open(LOG_PATH, "w"), indent=2)
    except Exception as e:
        print(f"[WARN] Logging failed: {e}")

@app.get("/analytics")
def get_analytics():
    if not os.path.exists(LOG_PATH):
        return {"count": 0, "message": "No logs yet."}
    
    logs = json.load(open(LOG_PATH))
    count = len(logs)
    mean_pred = np.mean([log["result"].get("predicted_goals", 0) for log in logs if "result" in log])
    latest = logs[-5:]

    return {
        "requests_count": count,
        "avg_predicted_goals": round(float(mean_pred), 2),
        "recent_requests": latest
    }

@app.get("/debug/csv_head")
def csv_head():
    try:
        df = _load_csv(CSV_PATH)
        return {
            "path": CSV_PATH,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "head": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/debug/csv_head failed: {e}")
    
@app.get("/evaluate_cv")
def evaluate_cv(k: int = 5):
    try:
        X, y = _build_training_frame(CSV_PATH)
        m = get_model()
        if m is None:
            raise HTTPException(status_code=500, detail="Model not loaded.")
        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # scores
        r2_scores = cross_val_score(m, X, y, cv=kf, scoring="r2")
        mse_scores = -cross_val_score(m, X, y, cv=kf, scoring="neg_mean_squared_error")

        return {
            "kfold": k,
            "R2_mean": float(r2_scores.mean()),
            "R2_std": float(r2_scores.std()),
            "MSE_mean": float(mse_scores.mean()),
            "MSE_std": float(mse_scores.std())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/evaluate_cv failed: {e}")
    
@app.post("/promote")
def promote_model():
    try:
        if not os.path.exists(MODEL_NEW):
            raise HTTPException(status_code=404, detail=f"Model v1.0 not found at {MODEL_NEW}")
        
        backup = MODEL_CURRENT + ".bak"
        if os.path.exists(MODEL_CURRENT):
            shutil.copy2(MODEL_CURRENT, backup)

        shutil.copy2(MODEL_NEW, MODEL_CURRENT)

        # reload the model in cache
        get_model.cache_clear()
        _ = get_model()

        return {
            "status": "promoted",
            "current_model": MODEL_CURRENT,
            "backup": backup
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/promote failed: {e}")