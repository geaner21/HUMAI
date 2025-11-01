# 🧠 HUMAI - Football Analytics API & Dashboard (Beta v0.9)

**HUMAI** is an analytics and prediction platform for football players performance, built on FastAPI + Streamlit.
Includes Random Forest models for estimating the number of goals (`Predicted Goals`) and comparative metrics between versions.

---

## ⚙️ Run locally

### 1️⃣ Start the FastAPI backend
```bash
uvicorn src.app_api_v1:app --reload

### 2️⃣ Start the Streamlit dashboard
streamlit run src/app_streamlit_v0_9.py


### 🌐 Endpoints
- GET /health, /health_extended, /model/expected_features
- POST /predict, /predict_batch, /retrain
- GET /compare?save_report=true
- POST /promote
- GET /analytics, /evaluate_cv

### Project structure
HUMAI/
HUMAI/
├── data/
│   └── premier-league-player-23-24.csv      # main dataset
│
├── models/
│   ├── humai_v0_9_rf.pkl                    # current model (prod)
│   └── humai_v1_0_rf.pkl                    # mnew model (after retrain)
│
├── reports/
│   ├── compare_metrics.json                 # model comparison report
│   └── api_requests_log.json                # logs / predict
│
├── src/
│   ├── app_api_v1.py                        # FastAPI backend
│   ├── app_streamlit_v0_9.py                # frontend dashboard
│   ├── humai_client.py                      # common API client
│   └── test_env.py                          # variables test .env
│
├── assets/
│   └── humai_logo.png                       # logo for UI
│
├── .env.example                             # local config example
├── .gitignore
└── README.md

### 🧩 Environment configuration
Copy .env.example in .env:
```bash
cp .env.example .env
Complete:
HUMAI_ENV=dev
HUMAI_SECRET_KEY=your_secret_here
HUMAI_API_URL=http://127.0.0.1:8000
HUMAI_REPORTS_DIR=C:\Users\caner\HUMAI\reports

### 🧠 Dataset
We use:
data/premier-league-player-23-24.csv
with the following base columns:
xG, xAG, npxG, PrgP, PrgC, PrgR, Min, Age, 90s, Gls

### 📊 Streamlit Tabs Overview
📈 Overview - Filter players, Performance Index (0-100), top performers
🎯 Predict - Individual prediction + batch upload
🛰️ API Analytics - Logs, latency, predictions distribution
🏋️ Model Metrics - Compare v0.9 vs v1.0, MSE and R² visually

### 🛠️ Dependencies
Install the dependencies:
```bash
pip install -r requirements.txt

### 🚀 Version
HUMAI Beta v0.9

### 👤 Author
Geaner M.
Manager & Football Data Developer ⚽
📧 Contact: canermustafa219@icloud.com