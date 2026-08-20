"""
AI-Based Crash Severity Prediction for Autonomous Vehicles
Divyank Seervi (24BCE0793)

Dataset: AV_accident_data__1_.xlsx (n=891)
Models:
    1. Ordinal Logistic Regression (custom, scipy-based)
    2. Random Forest Classifier (sklearn, class-weighted)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.special import expit
import scipy.optimize as opt
from scipy.stats import chi2_contingency

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, brier_score_loss,
    roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import json
import os
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
SEVERITY_LABELS = ["POD", "Minor", "Moderate", "Serious"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F9FA",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  AI-BASED CRASH SEVERITY PREDICTION — AUTONOMOUS VEHICLES")
print("=" * 65)

df = pd.read_excel("AV_accident_data__1_.xlsx")
print(f"\n[DATA]  Shape: {df.shape}")
print("\n[DATA]  Severity distribution:")
print(df["Severity"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

# Target encoding (ordinal: POD=0, Minor=1, Moderate=2, Serious=3)
severity_map = {"POD": 0, "Minor": 1, "Moderate": 2, "Serious": 3}
df["Severity_Enc"] = df["Severity"].map(severity_map)

# Fix "Unknown" strings in numeric columns — replace with median
for col in ["Posted Speed Limit (MPH)", "SV Precrash Speed (MPH)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# Engineer binary features from categorical columns
df["Is_Night"]       = (df["Incident_Time"] == "Night").astype(int)
df["Is_Wet"]         = (df["Roadway_Surface"] == "Wet").astype(int)
df["Is_Highway"]     = (df["Roadway_Type"] == "Highway").astype(int)
df["Is_Dark"]        = df["Lighting"].apply(lambda x: 1 if "Dark" in str(x) else 0)
df["Is_BadWeather"]  = df["Weather"].apply(lambda x: 0 if x in ["Clear", "Unknown"] else 1)
df["AirBag_Deployed"]= (df["Air_Bag"] == "Yes").astype(int)

# NEW: Advanced engineered features
# Speed ratio: how fast relative to limit (>1 = speeding)
df["Speed_Ratio"] = df["SV Precrash Speed (MPH)"] / df["Posted Speed Limit (MPH)"].replace(0, np.nan)
df["Speed_Ratio"] = df["Speed_Ratio"].fillna(0)

# Temporal features from Incident Date
df["Incident_Year"]  = pd.to_datetime(df["Incident Date"], errors="coerce").dt.year.fillna(2023).astype(int)
df["Incident_Month"] = pd.to_datetime(df["Incident Date"], errors="coerce").dt.month.fillna(6).astype(int)

# Vehicle age indicator
df["Model Year"] = pd.to_numeric(df["Model Year"], errors="coerce")
df["Is_OldVehicle"] = (df["Model Year"] < df["Model Year"].median()).astype(int)
df["Is_OldVehicle"] = df["Is_OldVehicle"].fillna(0).astype(int)

# Interaction: Speed × Night driving
df["Speed_Night"] = df["SV Precrash Speed (MPH)"] * df["Is_Night"]

# One-hot encode remaining categorical columns
cat_features = ["Roadway_Type", "Weather", "Lighting", "Crash_With"]
df_ohe = pd.get_dummies(df[cat_features], drop_first=True).astype(int)

# Assemble feature matrix
num_features = ["Posted Speed Limit (MPH)", "Mileage", "SV Precrash Speed (MPH)", "Speed_Ratio", "Speed_Night"]
eng_features = ["Is_Night", "Is_Wet", "Is_Highway", "Is_Dark", "Is_BadWeather", "AirBag_Deployed",
                "Incident_Year", "Incident_Month", "Is_OldVehicle"]

X = pd.concat([df[num_features + eng_features].astype(float), df_ohe], axis=1)
y = df["Severity_Enc"].values
feat_names = X.columns.tolist()

# Standardize numeric features (mean=0, std=1)
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

print(f"\n[FEAT]  Feature matrix shape: {X.shape}")

# Train / test split (80/20, stratified by severity)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[SPLIT] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL 1 — ORDINAL LOGISTIC REGRESSION (from scratch)
#
# Uses the Proportional-Odds / Cumulative Link model.
# Instead of predicting one class directly, it models cumulative
# probabilities: P(Y <= k) for each threshold k.
# Parameters are found by minimising negative log-likelihood via L-BFGS-B.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  MODEL 1: ORDINAL LOGISTIC REGRESSION")
print("─" * 65)

class OrdinalLogisticRegression:
    def __init__(self, n_classes=4):
        self.n = n_classes

    def _sigmoid(self, x):
        return expit(x)   # 1 / (1 + exp(-x))

    def _negative_log_likelihood(self, params, X, y):
        n_feat = X.shape[1]
        beta       = params[:n_feat]           # feature coefficients
        raw_thresh = params[n_feat:]           # unconstrained threshold params
        # Enforce strictly ordered thresholds via cumulative exp trick
        thresholds = np.cumsum(np.exp(raw_thresh))

        Xb = X @ beta
        log_likelihood = 0.0
        for i in range(len(y)):
            k = y[i]
            if k == 0:
                # P(Y=0) = P(Y<=0) = sigmoid(thresh[0] - Xb)
                prob = self._sigmoid(thresholds[0] - Xb[i])
            elif k == self.n - 1:
                # P(Y=K) = 1 - P(Y<=K-1)
                prob = 1 - self._sigmoid(thresholds[-1] - Xb[i])
            else:
                # P(Y=k) = P(Y<=k) - P(Y<=k-1)
                prob = (self._sigmoid(thresholds[k]   - Xb[i]) -
                        self._sigmoid(thresholds[k-1] - Xb[i]))
            log_likelihood += np.log(max(prob, 1e-9))
        return -log_likelihood   # we minimise, so negate

    def fit(self, X, y):
        n_feat = X.shape[1]
        # Initial params: zeros for coefficients, zeros for thresholds
        params0 = np.zeros(n_feat + (self.n - 1))
        result = opt.minimize(
            self._negative_log_likelihood,
            params0,
            args=(X, y),
            method="L-BFGS-B",
            options={"maxiter": 800, "ftol": 1e-8}
        )
        self.coef_       = result.x[:n_feat]
        self.thresholds_ = np.cumsum(np.exp(result.x[n_feat:]))
        return self

    def predict_proba(self, X):
        Xb   = X @ self.coef_
        probs = np.zeros((len(X), self.n))
        for i in range(len(X)):
            cumulative = [0.0] + [self._sigmoid(t - Xb[i]) for t in self.thresholds_] + [1.0]
            for k in range(self.n):
                probs[i, k] = max(cumulative[k+1] - cumulative[k], 0)
        # Normalise rows to sum to 1
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


olr = OrdinalLogisticRegression(n_classes=4)
olr.fit(X_train.values, y_train)

y_pred_olr  = olr.predict(X_test.values)
y_proba_olr = olr.predict_proba(X_test.values)

acc_olr = accuracy_score(y_test, y_pred_olr)
f1_olr  = f1_score(y_test, y_pred_olr, average="weighted", zero_division=0)
print(f"\n[OLR]  Accuracy: {acc_olr:.3f}  |  Weighted F1: {f1_olr:.3f}")
print(classification_report(y_test, y_pred_olr, target_names=SEVERITY_LABELS, zero_division=0))

# Ranked coefficients (absolute value = influence on severity)
coef_df = pd.DataFrame({"Feature": feat_names, "Coefficient": olr.coef_})
coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=False).index)
print("\n[OLR]  Top 10 Features by |Coefficient|:")
print(coef_df.head(10).to_string(index=False))

# (file continues with same logic as original script — unchanged)
