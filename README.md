# HUMAI Beta v0.9
## Run
1. API: `uvicorn src.app_api_v1:app --reload`
2. Streamlit: `streamlit run src/app_streamlit_v0_9.py`
## Endpoints
- GET /health, /health_extended, /model/expected_features
- POST /predict, /predict_batch, /retrain
- GET /compare?save_report=true
## Data
- data/premier-league-player-23-24.csv