"""
HUMAI Coach v0.3 (Day 17)
- Feature engineering
- Cross-validation
- GridSearchCV
- Explicability
- Saving model & report
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# === Robust paths ===
THIS_DIR = os.path.dirname(__file__)
PROJ_DIR = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(PROJ_DIR, "data")
REPORTS_DIR = os.path.join(PROJ_DIR, "reports")
MODELS_DIR = os.path.join(PROJ_DIR, "models")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# === Load ===
csv_path = os.path.join(DATA_DIR, "premier-player-23-24.csv")
df = pd.read_csv(csv_path)

# === Select + Feature Engineering ===
base_feats = ["xG", "xAG", "npxG", "PrgP", "PrgC", "PrgR", "Min", "Age", "90s"]
df = df.dropna(subset=base_feats + ["Gls"]).copy()

# Simple Derivatives
df["xG_per90"] = df["xG"] / df["90s"].replace(0, np.nan)
df["xAG_per90"] = df["xAG"] / df["90s"].replace(0, np.nan)
df["npxG_per90"] = df["npxG"] / df["90s"].replace(0, np.nan)
df["usage"] = df["Min"] / (df["90s"] * 90).replace(0, np.nan) # cât din 90' joacă efectiv
df["finishing_delta"] = df["Gls"] - df["xG"] # over/under performance

# Clean NaN Results from divisions
df = df.dropna(subset=["xG_per90", "xAG_per90", "npxG_per90", "usage"]).copy()

# Target & features
target = "Gls"
features = [
    "xG", "xAG", "npxG", "PrgP", "PrgC", "PrgR", "Min", "Age", "90s",
    "xG_per90", "xAG_per90", "npxG_per90", "usage"
]

X = df[features].copy()
y = df[target].copy()

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# === Preprocess (Numeric Scaling) ===
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

preprocess = ColumnTransformer(
    transformers=[("num", numeric_transformer, features)],
    remainder="drop"
)

# === Model pipeline ===
rf = RandomForestRegressor(random_state=42)
pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("rf", rf)
])

# === Basic cross-validation (without tuning) ===
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(pipe, X, y, scoring="r2", cv=cv)
cv_mse = cross_val_score(pipe, X, y, scoring="neg_mean_squared_error", cv=cv)

print(f"[CV] R^2 mean ={cv_r2.mean():.3f} (±{cv_r2.std():.3f})")
print(f"[CV] MSE mean={cv_mse.mean():.3f}")

# === Hyperparameter search ===
param_grid = {
    "rf__n_estimators": [200, 400, 800],
    "rf__max_depth": [6, 10, None],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2", None],
}
gcv = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="r2",
    cv=cv,
    n_jobs = -1,
    verbose=1
)
gcv.fit(X_train, y_train)

print("\nBest params:", gcv.best_params_)
print(f"Best CV R^2: {gcv.best_score_:.3f}")

# === Test evaluate ===
best_model = gcv.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n[TEST] MSE={mse:.3f} R^2={r2:.3f} (~RMSE={np.sqrt(mse):.2f} goals)")

# === Importances (from pipeline model) ===
# In order to obtain importances, we take the estimator from "rf" step
rf_fitted = best_model.named_steps["rf"]
importances = rf_fitted.feature_importances_
imp_df = pd.DataFrame({"Feature": features, "Importance": importances}) \
        .sort_values("Importance", ascending=False)

print("\nTop features:\n", imp_df)

plt.figure(figsize=(8,6))
sns.barplot(data=imp_df, x="Importance", y="Feature")
plt.title("HUMAI v0.3 - Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "humai_v0_3_feature_importance.png"))
plt.show()

# === Parity plot (y_test vs y_pred) ===
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred)
minv, maxv = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([minv, maxv], [minv, maxv], "r--", linewidth=2, label="Ideal")
plt.xlabel("Real Goals")
plt.ylabel("Predicted Goals")
plt.title("HUMAI v0.3 - Parity plot")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "humai_v0_3_parity.png"))
plt.show()

# === Save the model + columns ===
model_path = os.path.join(MODELS_DIR, "humai_v_03_rf.pkl")
joblib.dump(best_model, model_path)
pd.Series(features).to_csv(os.path.join(MODELS_DIR, "humai_v0_3_features.txt"), index=False)

print(f"\n Model saved to: {model_path}")
print("Reports saved to:", REPORTS_DIR)