"""
AI-Based Crash Severity Prediction for Autonomous Vehicles
Rohan Saxena (24BCE0801), Ankush Bhadouria (24BDS0428), Divyank Seervi (24BCE0793)

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

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL 2 — RANDOM FOREST CLASSIFIER (without SMOTE)
#
# Ensemble of 500 decision trees. Each tree is trained on a random
# bootstrap sample of the data, using a random subset of features at
# each split. Final prediction = majority vote across all trees.
# class_weight="balanced" up-weights minority classes (Minor/Moderate/Serious).
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  MODEL 2: RANDOM FOREST CLASSIFIER (Original)")
print("─" * 65)

rf = RandomForestClassifier(
    n_estimators=500,       # 500 trees
    max_depth=10,           # max depth per tree
    class_weight="balanced",# compensate for class imbalance
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
rf.fit(X_train, y_train)

y_pred_rf  = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf  = f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)
print(f"\n[RF]   Accuracy: {acc_rf:.3f}  |  Weighted F1: {f1_rf:.3f}")
print(classification_report(y_test, y_pred_rf, target_names=SEVERITY_LABELS, zero_division=0))

fi_df = pd.DataFrame({
    "Feature":    feat_names,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).head(12)
print("\n[RF]   Top 12 Feature Importances:")
print(fi_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 4b. SMOTE OVERSAMPLING + RETRAINED MODELS
#
# The dataset is heavily imbalanced (~91% POD). SMOTE generates synthetic
# minority-class samples so the models can learn to distinguish all classes.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  SMOTE OVERSAMPLING — Addressing Class Imbalance")
print("─" * 65)

print(f"\n[SMOTE] Before: {dict(zip(*np.unique(y_train, return_counts=True)))}")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"[SMOTE] After:  {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

# Retrain Random Forest on SMOTE-balanced data
rf_sm = RandomForestClassifier(
    n_estimators=500, max_depth=10, random_state=42, n_jobs=-1
)
rf_sm.fit(X_train_sm, y_train_sm)

y_pred_rf_sm  = rf_sm.predict(X_test)
y_proba_rf_sm = rf_sm.predict_proba(X_test)

acc_rf_sm = accuracy_score(y_test, y_pred_rf_sm)
f1_rf_sm  = f1_score(y_test, y_pred_rf_sm, average="weighted", zero_division=0)
print(f"\n[RF+SMOTE] Accuracy: {acc_rf_sm:.3f}  |  Weighted F1: {f1_rf_sm:.3f}")
print(classification_report(y_test, y_pred_rf_sm, target_names=SEVERITY_LABELS, zero_division=0))

# Retrain OLR on SMOTE-balanced data
olr_sm = OrdinalLogisticRegression(n_classes=4)
olr_sm.fit(X_train_sm.values, y_train_sm)
y_pred_olr_sm  = olr_sm.predict(X_test.values)
y_proba_olr_sm = olr_sm.predict_proba(X_test.values)

acc_olr_sm = accuracy_score(y_test, y_pred_olr_sm)
f1_olr_sm  = f1_score(y_test, y_pred_olr_sm, average="weighted", zero_division=0)
print(f"\n[OLR+SMOTE] Accuracy: {acc_olr_sm:.3f}  |  Weighted F1: {f1_olr_sm:.3f}")
print(classification_report(y_test, y_pred_olr_sm, target_names=SEVERITY_LABELS, zero_division=0))

fi_sm_df = pd.DataFrame({
    "Feature":    feat_names,
    "Importance": rf_sm.feature_importances_
}).sort_values("Importance", ascending=False).head(12)

# ─────────────────────────────────────────────────────────────────────────────
# 5. CROSS-VALIDATION & BRIER SCORES
# ─────────────────────────────────────────────────────────────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rf = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
cv_lr = cross_val_score(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    X, y, cv=cv, scoring="accuracy"
)
print(f"\n[CV]   RF  5-fold: {cv_rf.mean():.3f} ± {cv_rf.std():.3f}")
print(f"[CV]   LR  5-fold: {cv_lr.mean():.3f} ± {cv_lr.std():.3f}")

# Brier score: mean squared error between predicted probabilities and true labels
# Lower = better calibrated model
def full_proba(proba, classes, n_classes=4):
    out = np.zeros((proba.shape[0], n_classes))
    for i, c in enumerate(classes):
        out[:, c] = proba[:, i]
    return out

y_bin     = label_binarize(y_test, classes=[0, 1, 2, 3])
rf_pf     = full_proba(y_proba_rf, rf.classes_)
brier_rf  = np.mean([brier_score_loss(y_bin[:, k], rf_pf[:, k])  for k in range(4)])
brier_olr = np.mean([brier_score_loss(y_bin[:, k], y_proba_olr[:, k]) for k in range(4)])
print(f"[BRIER] RF: {brier_rf:.4f}  |  OLR: {brier_olr:.4f}")

rf_sm_pf     = full_proba(y_proba_rf_sm, rf_sm.classes_)
brier_rf_sm  = np.mean([brier_score_loss(y_bin[:, k], rf_sm_pf[:, k]) for k in range(4)])
brier_olr_sm = np.mean([brier_score_loss(y_bin[:, k], y_proba_olr_sm[:, k]) for k in range(4)])
print(f"[BRIER] RF+SMOTE: {brier_rf_sm:.4f}  |  OLR+SMOTE: {brier_olr_sm:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PROBABILITY & STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  PROBABILITY & STATISTICAL ANALYSIS")
print("─" * 65)

print("\n[PROB] Marginal Severity Probabilities:")
for s in SEVERITY_LABELS:
    print(f"  P({s}) = {(df['Severity'] == s).mean():.4f}")

print("\n[PROB] P(Severity | Weather):")
print(pd.crosstab(df["Weather"], df["Severity"], normalize="index").round(3).to_string())

# Wilson 95% Confidence Intervals for each severity proportion
print("\n[CI]   95% Wilson Confidence Intervals:")
n_all = len(df)
for s in SEVERITY_LABELS:
    k    = (df["Severity"] == s).sum()
    p    = k / n_all
    z    = 1.96
    denom  = 1 + z**2 / n_all
    centre = (p + z**2 / (2 * n_all)) / denom
    half   = z * np.sqrt(p*(1-p)/n_all + z**2/(4*n_all**2)) / denom
    print(f"  {s:8s}: p={p:.4f}  CI=[{max(0, centre-half):.4f}, {min(1, centre+half):.4f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 6b. CHI-SQUARE TESTS OF INDEPENDENCE
#
# H0: Severity is independent of the categorical variable
# H1: There is a significant association
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  CHI-SQUARE TESTS OF INDEPENDENCE")
print("─" * 65)

chi2_results = []
for col in ["Weather", "Lighting", "Roadway_Type", "Crash_With", "Roadway_Surface",
            "Incident_Time", "Air_Bag"]:
    ct = pd.crosstab(df[col], df["Severity"])
    chi2_val, p_val, dof, expected = chi2_contingency(ct)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    chi2_results.append({"Variable": col, "Chi2": chi2_val, "df": dof,
                         "p-value": p_val, "Significance": sig})
    print(f"  {col:20s}  χ²={chi2_val:8.2f}  df={dof:2d}  p={p_val:.4e}  {sig}")

print("\n  Legend: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
chi2_df = pd.DataFrame(chi2_results)

# ─────────────────────────────────────────────────────────────────────────────
# 6c. BAYES' THEOREM — CONDITIONAL PROBABILITY ANALYSIS
#
# Calculate P(Severity | Condition) using Bayes' Theorem:
# P(Severity=s | Condition=c) = P(Condition=c | Severity=s) * P(Severity=s)
#                                ─────────────────────────────────────────────
#                                              P(Condition=c)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BAYES' THEOREM — CONDITIONAL PROBABILITY ANALYSIS")
print("─" * 65)

def bayes_probability(df, condition_col, condition_val, severity_col="Severity"):
    """Compute P(Severity=s | Condition=c) for all severity levels using Bayes' Theorem."""
    results = {}
    p_condition = (df[condition_col] == condition_val).mean()
    if p_condition == 0:
        return results
    for s in SEVERITY_LABELS:
        p_severity = (df[severity_col] == s).mean()
        # P(Condition | Severity=s)
        sev_mask = df[severity_col] == s
        if sev_mask.sum() == 0:
            p_cond_given_sev = 0
        else:
            p_cond_given_sev = (df.loc[sev_mask, condition_col] == condition_val).mean()
        # Bayes' Theorem
        p_sev_given_cond = (p_cond_given_sev * p_severity) / p_condition
        results[s] = {
            "P(Cond|Sev)": p_cond_given_sev,
            "P(Severity)": p_severity,
            "P(Condition)": p_condition,
            "P(Sev|Cond)": p_sev_given_cond
        }
    return results

# Analyze key risk conditions
bayes_scenarios = [
    ("Incident_Time", "Night",  "Night-time driving"),
    ("Roadway_Surface", "Wet", "Wet road surface"),
    ("Roadway_Type", "Highway", "Highway driving"),
    ("Weather", "Snow/rain",    "Snow/Rain weather"),
    ("Lighting", "Dark/Not Lighted", "Dark/Not Lighted"),
]

for col, val, desc in bayes_scenarios:
    print(f"\n  Condition: {desc}  ({col} = '{val}')")
    bayes_res = bayes_probability(df, col, val)
    if not bayes_res:
        print(f"    No data found for {col}='{val}'")
        continue
    print(f"  {'Severity':10s} {'P(C|S)':>8s} {'P(S)':>8s} {'P(C)':>8s} {'P(S|C)':>8s}")
    print(f"  {'─'*46}")
    for s, v in bayes_res.items():
        print(f"  {s:10s} {v['P(Cond|Sev)']:8.4f} {v['P(Severity)']:8.4f} "
              f"{v['P(Condition)']:8.4f} {v['P(Sev|Cond)']:8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6d. ODDS RATIOS FROM OLR COEFFICIENTS
#
# Odds Ratio = exp(coefficient)
# OR > 1 → feature increases odds of higher severity
# OR < 1 → feature decreases odds of higher severity
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  ODDS RATIOS — OLR COEFFICIENT INTERPRETATION")
print("─" * 65)

odds_df = pd.DataFrame({
    "Feature":     feat_names,
    "Coefficient": olr.coef_,
    "Odds_Ratio":  np.exp(olr.coef_)
})
odds_df["Interpretation"] = odds_df["Odds_Ratio"].apply(
    lambda x: f"↑ {x:.2f}x higher severity" if x > 1
              else f"↓ {1/x:.2f}x lower severity" if x < 1 and x > 0
              else "No effect"
)
odds_df = odds_df.reindex(odds_df["Odds_Ratio"].apply(lambda x: abs(np.log(x))).sort_values(ascending=False).index)
print("\n  Top 15 Features by Odds Ratio Impact:")
print(f"  {'Feature':30s} {'Coeff':>8s} {'OR':>8s} {'Interpretation'}")
print(f"  {'─'*75}")
for _, row in odds_df.head(15).iterrows():
    print(f"  {row['Feature']:30s} {row['Coefficient']:8.4f} {row['Odds_Ratio']:8.4f} {row['Interpretation']}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATIONS (FIGURE 1 — Original 12-panel dashboard)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[VIZ]  Generating figures ...")

def safe_crosstab(col1, col2):
    ct = pd.crosstab(col1, col2, normalize="index")
    for c in SEVERITY_LABELS:
        if c not in ct.columns:
            ct[c] = 0
    return ct[SEVERITY_LABELS]

fig = plt.figure(figsize=(24, 28))
fig.suptitle(
    "AI-Based Crash Severity Prediction — Autonomous Vehicles\n"
    "Rohan Saxena · Ankush Bhadouria · Divyank Seervi  |  n=891",
    fontsize=16, fontweight="bold", y=0.995
)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38)

# Panel 1: Severity Distribution
ax1 = fig.add_subplot(gs[0, 0])
counts = df["Severity"].value_counts().reindex(SEVERITY_LABELS)
bars   = ax1.bar(SEVERITY_LABELS, counts.values, color=PALETTE, edgecolor="white", linewidth=1.2)
for bar, v in zip(bars, counts.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_title("Crash Severity Distribution (n=891)", fontweight="bold")
ax1.set_xlabel("Severity"); ax1.set_ylabel("Count")

# Panel 2: Severity by Weather
ax2 = fig.add_subplot(gs[0, 1])
safe_crosstab(df["Weather"], df["Severity"]).plot(kind="bar", ax=ax2, color=PALETTE, edgecolor="white")
ax2.set_title("Severity by Weather", fontweight="bold"); ax2.set_ylabel("Proportion")
ax2.legend(title="Severity", fontsize=8, title_fontsize=8); ax2.tick_params(axis="x", rotation=30)

# Panel 3: Severity by Lighting
ax3 = fig.add_subplot(gs[0, 2])
safe_crosstab(df["Lighting"], df["Severity"]).plot(kind="bar", ax=ax3, color=PALETTE, edgecolor="white")
ax3.set_title("Severity by Lighting", fontweight="bold"); ax3.set_ylabel("Proportion")
ax3.legend(title="Severity", fontsize=8, title_fontsize=8); ax3.tick_params(axis="x", rotation=30)

# Panel 4: Speed Limit Distribution
ax4 = fig.add_subplot(gs[1, 0])
for i, sev in enumerate(SEVERITY_LABELS):
    d = df.loc[df["Severity"] == sev, "Posted Speed Limit (MPH)"]
    if len(d) > 0:
        ax4.hist(d, bins=12, alpha=0.65, label=sev, color=PALETTE[i], edgecolor="white")
ax4.set_title("Speed Limit by Severity", fontweight="bold")
ax4.set_xlabel("Speed (MPH)"); ax4.set_ylabel("Freq"); ax4.legend(fontsize=8)

# Panel 5: Pre-crash Speed Boxplot
ax5 = fig.add_subplot(gs[1, 1])
db = [df.loc[df["Severity"] == s, "SV Precrash Speed (MPH)"].values for s in SEVERITY_LABELS]
bp = ax5.boxplot(db, patch_artist=True, medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax5.set_xticklabels(SEVERITY_LABELS)
ax5.set_title("Pre-Crash Speed by Severity", fontweight="bold"); ax5.set_ylabel("Speed (MPH)")

# Panel 6: RF Confusion Matrix (Original)
ax6 = fig.add_subplot(gs[1, 2])
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues",
            xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS,
            ax=ax6, linewidths=0.5, cbar=False)
ax6.set_title("Confusion Matrix — RF (Original)", fontweight="bold")
ax6.set_xlabel("Predicted"); ax6.set_ylabel("Actual")

# Panel 7: OLR Confusion Matrix (Original)
ax7 = fig.add_subplot(gs[2, 0])
sns.heatmap(confusion_matrix(y_test, y_pred_olr), annot=True, fmt="d", cmap="Oranges",
            xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS,
            ax=ax7, linewidths=0.5, cbar=False)
ax7.set_title("Confusion Matrix — OLR (Original)", fontweight="bold")
ax7.set_xlabel("Predicted"); ax7.set_ylabel("Actual")

# Panel 8: RF Feature Importances
ax8 = fig.add_subplot(gs[2, 1])
fi_plot = fi_df.sort_values("Importance")
ax8.barh(fi_plot["Feature"], fi_plot["Importance"],
         color=plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_plot))), edgecolor="white")
ax8.set_title("RF Feature Importances (Top 12)", fontweight="bold"); ax8.set_xlabel("Importance")

# Panel 9: OLR Coefficients
ax9 = fig.add_subplot(gs[2, 2])
top_c = coef_df.head(12).sort_values("Coefficient")
ax9.barh(top_c["Feature"], top_c["Coefficient"],
         color=["#F44336" if v > 0 else "#2196F3" for v in top_c["Coefficient"]],
         edgecolor="white")
ax9.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax9.set_title("OLR Coefficients (Top 12)", fontweight="bold"); ax9.set_xlabel("Coefficient")

# Panel 10: Model Comparison (Original vs SMOTE)
ax10 = fig.add_subplot(gs[3, 0])
metrics4  = {"Accuracy": [acc_olr, acc_rf, acc_olr_sm, acc_rf_sm],
             "Weighted F1": [f1_olr, f1_rf, f1_olr_sm, f1_rf_sm],
             "1-Brier":  [1-brier_olr, 1-brier_rf, 1-brier_olr_sm, 1-brier_rf_sm]}
x_pos, w = np.arange(3), 0.18
ax10.bar(x_pos-1.5*w, [v[0] for v in metrics4.values()], width=w, color="#FF9800", label="OLR",       edgecolor="white")
ax10.bar(x_pos-0.5*w, [v[1] for v in metrics4.values()], width=w, color="#2196F3", label="RF",        edgecolor="white")
ax10.bar(x_pos+0.5*w, [v[2] for v in metrics4.values()], width=w, color="#FFC107", label="OLR+SMOTE", edgecolor="white")
ax10.bar(x_pos+1.5*w, [v[3] for v in metrics4.values()], width=w, color="#00BCD4", label="RF+SMOTE",  edgecolor="white")
ax10.set_xticks(x_pos); ax10.set_xticklabels(list(metrics4.keys())); ax10.set_ylim(0, 1.1)
ax10.set_title("Model Comparison (All 4)", fontweight="bold"); ax10.set_ylabel("Score")
ax10.legend(fontsize=7, loc="lower right")

# Panel 11: Severity by Roadway Type
ax11 = fig.add_subplot(gs[3, 1])
safe_crosstab(df["Roadway_Type"], df["Severity"]).plot(kind="bar", ax=ax11, color=PALETTE, edgecolor="white")
ax11.set_title("Severity by Roadway Type", fontweight="bold"); ax11.set_ylabel("Proportion")
ax11.legend(title="Severity", fontsize=8, title_fontsize=8); ax11.tick_params(axis="x", rotation=15)

# Panel 12: 5-Fold CV Boxplot
ax12 = fig.add_subplot(gs[3, 2])
bpl = ax12.boxplot([cv_rf, cv_lr], patch_artist=True, medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bpl["boxes"], ["#2196F3", "#FF9800"]):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax12.set_xticklabels(["Random Forest", "Logistic Reg."])
ax12.set_title("5-Fold CV Accuracy", fontweight="bold"); ax12.set_ylabel("Accuracy"); ax12.set_ylim(0, 1.1)

plt.savefig("crash_severity_analysis_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print("[VIZ]  Saved → crash_severity_analysis_v2.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7b. FIGURE 2 — NEW STATISTICAL ANALYSIS DASHBOARD
#     SMOTE confusion matrices, ROC curves, Chi-Square, Bayes, Odds Ratios
# ─────────────────────────────────────────────────────────────────────────────
print("[VIZ]  Generating enhanced analysis figure ...")

fig2 = plt.figure(figsize=(24, 21))
fig2.suptitle(
    "Enhanced Statistical Analysis — SMOTE, ROC, Chi-Square, Bayes & Odds Ratios\n"
    "Rohan Saxena · Ankush Bhadouria · Divyank Seervi  |  n=891",
    fontsize=16, fontweight="bold", y=0.995
)
gs2 = gridspec.GridSpec(3, 3, figure=fig2, hspace=0.50, wspace=0.38)

# Panel A1: RF+SMOTE Confusion Matrix
ax_a1 = fig2.add_subplot(gs2[0, 0])
sns.heatmap(confusion_matrix(y_test, y_pred_rf_sm), annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS,
            ax=ax_a1, linewidths=0.5, cbar=False)
ax_a1.set_title("Confusion Matrix — RF + SMOTE", fontweight="bold")
ax_a1.set_xlabel("Predicted"); ax_a1.set_ylabel("Actual")

# Panel A2: OLR+SMOTE Confusion Matrix
ax_a2 = fig2.add_subplot(gs2[0, 1])
sns.heatmap(confusion_matrix(y_test, y_pred_olr_sm), annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS,
            ax=ax_a2, linewidths=0.5, cbar=False)
ax_a2.set_title("Confusion Matrix — OLR + SMOTE", fontweight="bold")
ax_a2.set_xlabel("Predicted"); ax_a2.set_ylabel("Actual")

# Panel A3: SMOTE Class Distribution (Before vs After)
ax_a3 = fig2.add_subplot(gs2[0, 2])
before_counts = dict(zip(*np.unique(y_train, return_counts=True)))
after_counts  = dict(zip(*np.unique(y_train_sm, return_counts=True)))
x_sm = np.arange(4)
w_sm = 0.35
ax_a3.bar(x_sm - w_sm/2, [before_counts.get(i, 0) for i in range(4)],
          width=w_sm, color="#F44336", label="Before SMOTE", edgecolor="white", alpha=0.8)
ax_a3.bar(x_sm + w_sm/2, [after_counts.get(i, 0) for i in range(4)],
          width=w_sm, color="#4CAF50", label="After SMOTE", edgecolor="white", alpha=0.8)
ax_a3.set_xticks(x_sm); ax_a3.set_xticklabels(SEVERITY_LABELS)
ax_a3.set_title("SMOTE: Class Distribution Before vs After", fontweight="bold")
ax_a3.set_ylabel("Count"); ax_a3.legend(fontsize=9)

# Panel B1: ROC Curves — Random Forest (One-vs-Rest)
ax_b1 = fig2.add_subplot(gs2[1, 0])
roc_colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
for k in range(4):
    fpr, tpr, _ = roc_curve(y_bin[:, k], rf_pf[:, k])
    roc_auc = auc(fpr, tpr)
    ax_b1.plot(fpr, tpr, color=roc_colors[k], linewidth=2,
               label=f"{SEVERITY_LABELS[k]} (AUC={roc_auc:.3f})")
ax_b1.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
ax_b1.set_title("ROC Curves — Random Forest (OvR)", fontweight="bold")
ax_b1.set_xlabel("False Positive Rate"); ax_b1.set_ylabel("True Positive Rate")
ax_b1.legend(fontsize=8); ax_b1.set_xlim([-0.02, 1.02]); ax_b1.set_ylim([-0.02, 1.02])

# Panel B2: ROC Curves — RF+SMOTE (One-vs-Rest)
ax_b2 = fig2.add_subplot(gs2[1, 1])
for k in range(4):
    fpr, tpr, _ = roc_curve(y_bin[:, k], rf_sm_pf[:, k])
    roc_auc = auc(fpr, tpr)
    ax_b2.plot(fpr, tpr, color=roc_colors[k], linewidth=2,
               label=f"{SEVERITY_LABELS[k]} (AUC={roc_auc:.3f})")
ax_b2.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
ax_b2.set_title("ROC Curves — RF + SMOTE (OvR)", fontweight="bold")
ax_b2.set_xlabel("False Positive Rate"); ax_b2.set_ylabel("True Positive Rate")
ax_b2.legend(fontsize=8); ax_b2.set_xlim([-0.02, 1.02]); ax_b2.set_ylim([-0.02, 1.02])

# Panel B3: ROC Curves — OLR (One-vs-Rest)
ax_b3 = fig2.add_subplot(gs2[1, 2])
for k in range(4):
    fpr, tpr, _ = roc_curve(y_bin[:, k], y_proba_olr[:, k])
    roc_auc = auc(fpr, tpr)
    ax_b3.plot(fpr, tpr, color=roc_colors[k], linewidth=2,
               label=f"{SEVERITY_LABELS[k]} (AUC={roc_auc:.3f})")
ax_b3.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
ax_b3.set_title("ROC Curves — OLR (OvR)", fontweight="bold")
ax_b3.set_xlabel("False Positive Rate"); ax_b3.set_ylabel("True Positive Rate")
ax_b3.legend(fontsize=8); ax_b3.set_xlim([-0.02, 1.02]); ax_b3.set_ylim([-0.02, 1.02])

# Panel C1: Chi-Square Test Results
ax_c1 = fig2.add_subplot(gs2[2, 0])
chi2_plot = chi2_df.sort_values("Chi2", ascending=True)
bar_colors = ["#F44336" if p < 0.05 else "#9E9E9E" for p in chi2_plot["p-value"]]
ax_c1.barh(chi2_plot["Variable"], chi2_plot["Chi2"], color=bar_colors, edgecolor="white")
ax_c1.set_title("Chi-Square Tests of Independence", fontweight="bold")
ax_c1.set_xlabel("χ² Statistic")
# Add p-value annotations
for i, (_, row) in enumerate(chi2_plot.iterrows()):
    sig_text = row["Significance"]
    ax_c1.text(row["Chi2"] + 0.5, i, f"p={row['p-value']:.3e} {sig_text}",
               va="center", fontsize=8, fontweight="bold" if sig_text != "ns" else "normal")

# Panel C2: Bayes' Theorem — P(Severity | Condition) Heatmap
ax_c2 = fig2.add_subplot(gs2[2, 1])
bayes_matrix = []
bayes_labels_y = []
for col, val, desc in bayes_scenarios:
    b_res = bayes_probability(df, col, val)
    if b_res:
        bayes_labels_y.append(desc)
        bayes_matrix.append([b_res[s]["P(Sev|Cond)"] for s in SEVERITY_LABELS])
if bayes_matrix:
    sns.heatmap(np.array(bayes_matrix), annot=True, fmt=".3f", cmap="RdYlGn_r",
                xticklabels=SEVERITY_LABELS, yticklabels=bayes_labels_y,
                ax=ax_c2, linewidths=0.5, cbar_kws={"label": "P(Severity|Condition)"})
ax_c2.set_title("Bayes' Theorem — P(Severity | Condition)", fontweight="bold")

# Panel C3: Odds Ratios (Top Features)
ax_c3 = fig2.add_subplot(gs2[2, 2])
odds_plot = odds_df.head(12).sort_values("Odds_Ratio")
bar_colors_or = ["#F44336" if v > 1 else "#2196F3" for v in odds_plot["Odds_Ratio"]]
ax_c3.barh(odds_plot["Feature"], odds_plot["Odds_Ratio"], color=bar_colors_or, edgecolor="white")
ax_c3.axvline(1, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
ax_c3.set_title("Odds Ratios — OLR (Top 12)", fontweight="bold")
ax_c3.set_xlabel("Odds Ratio (exp(β))")
ax_c3.annotate("OR>1 = ↑ severity", xy=(0.98, 0.98), xycoords="axes fraction",
               ha="right", va="top", fontsize=8, color="#F44336", fontweight="bold")
ax_c3.annotate("OR<1 = ↓ severity", xy=(0.98, 0.92), xycoords="axes fraction",
               ha="right", va="top", fontsize=8, color="#2196F3", fontweight="bold")

plt.savefig("crash_severity_enhanced_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("[VIZ]  Saved → crash_severity_enhanced_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SCENARIO RISK SCORING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  SCENARIO RISK SCORING (Random Forest)")
print("─" * 65)

def risk_score(speed, precrash, mileage, is_night, is_wet, is_highway, is_dark, is_bad_weather, airbag):
    row = {col: 0.0 for col in X.columns}
    speed_ratio = precrash / speed if speed > 0 else 0
    speed_night = precrash * is_night
    scaled = scaler.transform([[speed, mileage, precrash, speed_ratio, speed_night]])[0]
    row["Posted Speed Limit (MPH)"]  = scaled[0]
    row["Mileage"]                   = scaled[1]
    row["SV Precrash Speed (MPH)"]   = scaled[2]
    row["Speed_Ratio"]               = scaled[3]
    row["Speed_Night"]               = scaled[4]
    row["Is_Night"]        = is_night
    row["Is_Wet"]          = is_wet
    row["Is_Highway"]      = is_highway
    row["Is_Dark"]         = is_dark
    row["Is_BadWeather"]   = is_bad_weather
    row["AirBag_Deployed"] = airbag
    row["Incident_Year"]   = 2024
    row["Incident_Month"]  = 6
    row["Is_OldVehicle"]   = 0
    X_row = pd.DataFrame([row])[X.columns].astype(float)
    proba_raw = rf.predict_proba(X_row)[0]
    proba_full = np.zeros(4)
    for i, c in enumerate(rf.classes_):
        proba_full[c] = proba_raw[i]
    risk_index = sum(p * i for i, p in enumerate(proba_full))
    return proba_full, risk_index

print(f"\n{'Scenario':<42} {'POD':>6} {'Minor':>6} {'Moderate':>8} {'Serious':>7} {'RiskIdx':>8}")
print("─" * 82)
scenarios = [
    ("Low-risk:  Day / Dry  / 25 mph",   25,  5, 1000, 0, 0, 0, 0, 0, 0),
    ("Moderate:  Night/ Dry / 35 mph",   35, 10,  500, 1, 0, 0, 1, 0, 0),
    ("High-risk: Night/ Wet / 65 mph",   65, 50,  200, 1, 1, 1, 1, 1, 0),
    ("Worst-case:Wet / Hwy  / 65 mph",   65, 55,  100, 1, 1, 1, 1, 1, 1),
]
for name, spd, pre, mil, night, wet, hwy, dark, bw, ab in scenarios:
    pf, ri = risk_score(spd, pre, mil, night, wet, hwy, dark, bw, ab)
    print(f"{name:<42} {pf[0]:>6.3f} {pf[1]:>6.3f} {pf[2]:>8.3f} {pf[3]:>7.3f} {ri:>8.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. NEW MODELS — GRADIENT BOOSTING, MLP, SVM
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  MODEL 3: GRADIENT BOOSTING CLASSIFIER")
print("─" * 65)

gb = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    random_state=42, subsample=0.8
)
gb.fit(X_train_sm, y_train_sm)
y_pred_gb  = gb.predict(X_test)
y_proba_gb = gb.predict_proba(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)
f1_gb  = f1_score(y_test, y_pred_gb, average="weighted", zero_division=0)
print(f"\n[GBM]  Accuracy: {acc_gb:.3f}  |  Weighted F1: {f1_gb:.3f}")
print(classification_report(y_test, y_pred_gb, target_names=SEVERITY_LABELS, zero_division=0))

gb_pf = full_proba(y_proba_gb, gb.classes_)
brier_gb = np.mean([brier_score_loss(y_bin[:, k], gb_pf[:, k]) for k in range(4)])

print("\n" + "─" * 65)
print("  MODEL 4: MULTI-LAYER PERCEPTRON (Neural Network)")
print("─" * 65)

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), activation="relu",
    max_iter=500, random_state=42, early_stopping=True,
    validation_fraction=0.15, alpha=0.001
)
mlp.fit(X_train_sm, y_train_sm)
y_pred_mlp  = mlp.predict(X_test)
y_proba_mlp = mlp.predict_proba(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
f1_mlp  = f1_score(y_test, y_pred_mlp, average="weighted", zero_division=0)
print(f"\n[MLP]  Accuracy: {acc_mlp:.3f}  |  Weighted F1: {f1_mlp:.3f}")
print(classification_report(y_test, y_pred_mlp, target_names=SEVERITY_LABELS, zero_division=0))

mlp_pf = full_proba(y_proba_mlp, mlp.classes_)
brier_mlp = np.mean([brier_score_loss(y_bin[:, k], mlp_pf[:, k]) for k in range(4)])

print("\n" + "─" * 65)
print("  MODEL 5: SUPPORT VECTOR MACHINE")
print("─" * 65)

svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42, C=1.0)
svm.fit(X_train_sm, y_train_sm)
y_pred_svm  = svm.predict(X_test)
y_proba_svm = svm.predict_proba(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
f1_svm  = f1_score(y_test, y_pred_svm, average="weighted", zero_division=0)
print(f"\n[SVM]  Accuracy: {acc_svm:.3f}  |  Weighted F1: {f1_svm:.3f}")
print(classification_report(y_test, y_pred_svm, target_names=SEVERITY_LABELS, zero_division=0))

svm_pf = full_proba(y_proba_svm, svm.classes_)
brier_svm = np.mean([brier_score_loss(y_bin[:, k], svm_pf[:, k]) for k in range(4)])

# ─────────────────────────────────────────────────────────────────────────────
# 10. PERMUTATION IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  PERMUTATION IMPORTANCE — Random Forest")
print("─" * 65)

perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({
    "Feature": feat_names,
    "Importance_Mean": perm_imp.importances_mean,
    "Importance_Std":  perm_imp.importances_std
}).sort_values("Importance_Mean", ascending=False)
print("\n[PERM] Top 12 by Permutation Importance:")
print(perm_df.head(12).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 11. SAVE MODELS & EXPORT JSON FOR FRONTEND
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  SAVING MODELS & EXPORTING JSON")
print("─" * 65)

os.makedirs("static", exist_ok=True)
joblib.dump(rf, "static/rf_model.joblib")
joblib.dump(scaler, "static/scaler.joblib")
print("[SAVE] Models saved to static/")

# Build comprehensive JSON for frontend
roc_data = {}
for model_name, proba_mat in [("RF", rf_pf), ("RF_SMOTE", rf_sm_pf),
                               ("OLR", y_proba_olr), ("GBM", gb_pf),
                               ("MLP", mlp_pf), ("SVM", svm_pf)]:
    roc_data[model_name] = {}
    for k in range(4):
        fpr_arr, tpr_arr, _ = roc_curve(y_bin[:, k], proba_mat[:, k])
        roc_auc_val = auc(fpr_arr, tpr_arr)
        roc_data[model_name][SEVERITY_LABELS[k]] = {
            "fpr": fpr_arr[::max(1, len(fpr_arr)//50)].tolist(),
            "tpr": tpr_arr[::max(1, len(tpr_arr)//50)].tolist(),
            "auc": round(roc_auc_val, 4)
        }

# Confusion matrices
def cm_to_list(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).tolist()

# Bayes data for frontend
bayes_data = {}
for col, val, desc in bayes_scenarios:
    b_res = bayes_probability(df, col, val)
    if b_res:
        bayes_data[desc] = {s: round(v["P(Sev|Cond)"], 4) for s, v in b_res.items()}

# Scenario predictions for frontend
scenario_data = []
for name, spd, pre, mil, night, wet, hwy, dark, bw, ab in scenarios:
    pf, ri = risk_score(spd, pre, mil, night, wet, hwy, dark, bw, ab)
    scenario_data.append({
        "name": name, "probabilities": {SEVERITY_LABELS[i]: round(float(pf[i]), 4) for i in range(4)},
        "risk_index": round(float(ri), 4)
    })

# Dataset summary for frontend
sev_dist = df["Severity"].value_counts().to_dict()
weather_severity = pd.crosstab(df["Weather"], df["Severity"], normalize="index").round(4).to_dict()

model_results = {
    "dataset": {
        "total_samples": int(len(df)),
        "features_count": int(X.shape[1]),
        "feature_names": feat_names,
        "severity_distribution": {k: int(v) for k, v in sev_dist.items()},
        "severity_labels": SEVERITY_LABELS,
    },
    "models": {
        "OLR": {"accuracy": round(acc_olr, 4), "f1": round(f1_olr, 4), "brier": round(brier_olr, 4),
                "confusion_matrix": cm_to_list(y_test, y_pred_olr)},
        "RF":  {"accuracy": round(acc_rf, 4), "f1": round(f1_rf, 4), "brier": round(brier_rf, 4),
                "confusion_matrix": cm_to_list(y_test, y_pred_rf),
                "feature_importances": {feat_names[i]: round(float(rf.feature_importances_[i]), 4)
                                        for i in range(len(feat_names))}},
        "OLR_SMOTE": {"accuracy": round(acc_olr_sm, 4), "f1": round(f1_olr_sm, 4), "brier": round(brier_olr_sm, 4),
                      "confusion_matrix": cm_to_list(y_test, y_pred_olr_sm)},
        "RF_SMOTE":  {"accuracy": round(acc_rf_sm, 4), "f1": round(f1_rf_sm, 4), "brier": round(brier_rf_sm, 4),
                      "confusion_matrix": cm_to_list(y_test, y_pred_rf_sm),
                      "feature_importances": {feat_names[i]: round(float(rf_sm.feature_importances_[i]), 4)
                                              for i in range(len(feat_names))}},
        "GBM": {"accuracy": round(acc_gb, 4), "f1": round(f1_gb, 4), "brier": round(brier_gb, 4),
                "confusion_matrix": cm_to_list(y_test, y_pred_gb),
                "feature_importances": {feat_names[i]: round(float(gb.feature_importances_[i]), 4)
                                        for i in range(len(feat_names))}},
        "MLP": {"accuracy": round(acc_mlp, 4), "f1": round(f1_mlp, 4), "brier": round(brier_mlp, 4),
                "confusion_matrix": cm_to_list(y_test, y_pred_mlp)},
        "SVM": {"accuracy": round(acc_svm, 4), "f1": round(f1_svm, 4), "brier": round(brier_svm, 4),
                "confusion_matrix": cm_to_list(y_test, y_pred_svm)},
    },
    "cross_validation": {
        "RF_5fold_mean": round(float(cv_rf.mean()), 4),
        "RF_5fold_std":  round(float(cv_rf.std()), 4),
        "LR_5fold_mean": round(float(cv_lr.mean()), 4),
        "LR_5fold_std":  round(float(cv_lr.std()), 4),
    },
    "roc_curves": roc_data,
    "chi_square": chi2_df.to_dict(orient="records"),
    "bayes_analysis": bayes_data,
    "odds_ratios": odds_df.head(15)[["Feature","Coefficient","Odds_Ratio","Interpretation"]].to_dict(orient="records"),
    "permutation_importance": perm_df.head(15).to_dict(orient="records"),
    "scenarios": scenario_data,
    "smote": {
        "before": {SEVERITY_LABELS[i]: int(dict(zip(*np.unique(y_train, return_counts=True))).get(i, 0)) for i in range(4)},
        "after":  {SEVERITY_LABELS[i]: int(dict(zip(*np.unique(y_train_sm, return_counts=True))).get(i, 0)) for i in range(4)},
    },
    "olr_coefficients": {feat_names[i]: round(float(olr.coef_[i]), 4) for i in range(len(feat_names))},
    "scaler_params": {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": num_features,
    },
}

with open("model_results.json", "w") as f:
    json.dump(model_results, f, indent=2)
print("[JSON] Saved → model_results.json")

# ─────────────────────────────────────────────────────────────────────────────
# 12. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
=================================================================
  FINAL SUMMARY  (n=891, {X.shape[1]} features)
=================================================================
  ORIGINAL MODELS:
    OLR — Acc:{acc_olr:.3f}  F1:{f1_olr:.3f}  Brier:{brier_olr:.4f}
    RF  — Acc:{acc_rf:.3f}  F1:{f1_rf:.3f}  Brier:{brier_rf:.4f}

  SMOTE-ENHANCED MODELS:
    OLR+SMOTE — Acc:{acc_olr_sm:.3f}  F1:{f1_olr_sm:.3f}  Brier:{brier_olr_sm:.4f}
    RF+SMOTE  — Acc:{acc_rf_sm:.3f}  F1:{f1_rf_sm:.3f}  Brier:{brier_rf_sm:.4f}

  NEW MODELS (SMOTE-trained):
    GBM — Acc:{acc_gb:.3f}  F1:{f1_gb:.3f}  Brier:{brier_gb:.4f}
    MLP — Acc:{acc_mlp:.3f}  F1:{f1_mlp:.3f}  Brier:{brier_mlp:.4f}
    SVM — Acc:{acc_svm:.3f}  F1:{f1_svm:.3f}  Brier:{brier_svm:.4f}

  CROSS-VALIDATION:
    CV RF 5-fold: {cv_rf.mean():.3f} +/- {cv_rf.std():.3f}

  STATISTICAL ANALYSIS:
    Chi-Square: {sum(1 for r in chi2_results if r['p-value'] < 0.05)}/{len(chi2_results)} significant
    Bayes: 5 risk conditions analyzed
    Odds Ratios: {sum(1 for _, r in odds_df.iterrows() if r['Odds_Ratio'] > 1)} features increase severity

  OUTPUT FILES:
    -> crash_severity_analysis_v2.png
    -> crash_severity_enhanced_analysis.png
    -> model_results.json
    -> static/rf_model.joblib, static/scaler.joblib
=================================================================
""")
