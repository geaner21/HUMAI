# src/utils_scores.py
import numpy as np
import pandas as pd

WEIGHTS = {"xG": 0.40, "xAG": 0.30, "PrgR": 0.20, "Min": 0.10}

def compute_raw_index(df: pd.DataFrame) -> pd.Series:
    xg = df.get("xG", pd.Series(0, index=df.index)).fillna(0)
    xag = df.get("xAG", pd.Series(0, index=df.index)).fillna(0)
    prgr = df.get("PrgR", pd.Series(0, index=df.index)).fillna(0)
    mins = df.get("Min", pd.Series(0, index=df.index)).fillna(0)

    # normalizări simple pentru comparabilitate între jucători
    prgr_scaled = (prgr / max(prgr.max(), 1)).fillna(0) * 10
    mins_scaled = (mins / 3420).clip(0, 1) * 10 #38*90=3420

    return (
        xg * WEIGHTS["xG"] +
        xag * WEIGHTS["xAG"] +
        prgr_scaled * WEIGHTS["PrgR"] +
        mins_scaled * WEIGHTS["Min"]
    )

def minmax_norm(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo: #fallback dacă toti au același scor
        return pd.Series(50.0, index=s.index)
    return (s - lo) / (hi - lo) * 100.0

def attach_perf_index(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    raw = compute_raw_index(df2)
    df2["Performance_Index"] = minmax_norm(raw).round(2)
    return df2
