import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import scipy.optimize as opt
from scipy.special import expit
from scipy.stats import chi2_contingency

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, brier_score_loss,
    roc_curve, auc, roc_auc_score, cohen_kappa_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import shap

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  CRASHPREDICT AI — FULL DATASET EXPERIMENTS & METRICS GENERATOR")
print("=" * 65)

excel_path = r"AV accident data.xlsx"
if not os.path.exists(excel_path):
    excel_path = os.path.join(r"crash-prediction", r"backend", r"AV_accident_data__1_.xlsx")

print(f"Loading data from: {excel_path}")
df = pd.read_excel(excel_path)
print(f"Dataset shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING (Exact matching repo pipeline)
# ─────────────────────────────────────────────────────────────────────────────
severity_map = {"POD": 0, "Minor": 1, "Moderate": 2, "Serious": 3}
df["Severity_Enc"] = df["Severity"].map(severity_map)

for col in ["Posted Speed Limit (MPH)", "SV Precrash Speed (MPH)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

df["Is_Night"]       = (df["Incident_Time"] == "Night").astype(int)
df["Is_Wet"]         = (df["Roadway_Surface"] == "Wet").astype(int)
df["Is_Highway"]     = (df["Roadway_Type"] == "Highway").astype(int)
df["Is_Dark"]        = df["Lighting"].apply(lambda x: 1 if "Dark" in str(x) else 0)
df["Is_BadWeather"]  = df["Weather"].apply(lambda x: 0 if x in ["Clear", "Unknown"] else 1)
df["AirBag_Deployed"]= (df["Air_Bag"] == "Yes").astype(int)

df["Speed_Ratio"] = df["SV Precrash Speed (MPH)"] / df["Posted Speed Limit (MPH)"].replace(0, np.nan)
df["Speed_Ratio"] = df["Speed_Ratio"].fillna(0)

df["Incident_Year"]  = pd.to_datetime(df["Incident Date"], errors="coerce").dt.year.fillna(2023).astype(int)
df["Incident_Month"] = pd.to_datetime(df["Incident Date"], errors="coerce").dt.month.fillna(6).astype(int)

df["Model Year"] = pd.to_numeric(df["Model Year"], errors="coerce")
df["Is_OldVehicle"] = (df["Model Year"] < df["Model Year"].median()).astype(int)
df["Is_OldVehicle"] = df["Is_OldVehicle"].fillna(0).astype(int)

df["Speed_Night"] = df["SV Precrash Speed (MPH)"] * df["Is_Night"]

cat_features = ["Roadway_Type", "Weather", "Lighting", "Crash_With"]
df_ohe = pd.get_dummies(df[cat_features], drop_first=True).astype(int)

num_features = ["Posted Speed Limit (MPH)", "Mileage", "SV Precrash Speed (MPH)", "Speed_Ratio", "Speed_Night"]
eng_features = ["Is_Night", "Is_Wet", "Is_Highway", "Is_Dark", "Is_BadWeather", "AirBag_Deployed",
                "Incident_Year", "Incident_Month", "Is_OldVehicle"]

X = pd.concat([df[num_features + eng_features].astype(float), df_ohe], axis=1)
y = df["Severity_Enc"].values
feat_names = X.columns.tolist()

scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
print("Class counts in train:", pd.Series(y_train).value_counts().to_dict())
print("Class counts in test: ", pd.Series(y_test).value_counts().to_dict())

# ─────────────────────────────────────────────────────────────────────────────
# 3. ORDINAL LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
class OrdinalLogisticRegression:
    def __init__(self, n_classes=4):
        self.n = n_classes

    def _sigmoid(self, x):
        return expit(x)

    def _negative_log_likelihood(self, params, X, y):
        n_feat = X.shape[1]
        beta       = params[:n_feat]
        raw_thresh = params[n_feat:]
        thresholds = np.cumsum(np.exp(raw_thresh))

        Xb = X @ beta
        log_likelihood = 0.0
        for i in range(len(y)):
            k = y[i]
            if k == 0:
                prob = self._sigmoid(thresholds[0] - Xb[i])
            elif k == self.n - 1:
                prob = 1 - self._sigmoid(thresholds[-1] - Xb[i])
            else:
                prob = (self._sigmoid(thresholds[k]   - Xb[i]) -
                        self._sigmoid(thresholds[k-1] - Xb[i]))
            log_likelihood += np.log(max(prob, 1e-9))
        return -log_likelihood

    def fit(self, X, y):
        n_feat = X.shape[1]
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
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

olr = OrdinalLogisticRegression(n_classes=4).fit(X_train.values, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

models = {
    "Random Forest (Baseline)": RandomForestClassifier(n_estimators=500, random_state=42),
    "Random Forest + SMOTE": RandomForestClassifier(n_estimators=500, random_state=42),
    "Random Forest (Cost-Sens.)": RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, eval_metric='mlogloss'),
    "LightGBM": lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1),
    "Ordinal Logistic Regression": olr,
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Multi-Layer Perceptron": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
}

results_summary = []
per_class_summary = {}
prob_dict = {}

y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

for name, model in models.items():
    if name == "Random Forest + SMOTE":
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
    elif name == "Ordinal Logistic Regression":
        y_pred = model.predict(X_test.values)
        y_proba = model.predict_proba(X_test.values)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

    prob_dict[name] = y_proba
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    brier = np.mean(np.sum((y_test_bin - y_proba)**2, axis=1))
    auc_score = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    
    results_summary.append({
        "Model": name,
        "Accuracy": acc,
        "Macro F1": f1_macro,
        "Weighted F1": f1_weighted,
        "Cohen's Kappa": kappa,
        "Brier Score": brier,
        "Macro AUC": auc_score
    })

    # Detailed per-class breakdown
    report = classification_report(y_test, y_pred, target_names=["POD", "Minor", "Moderate", "Serious"], output_dict=True, zero_division=0)
    per_class_summary[name] = report

df_res = pd.DataFrame(results_summary)
print("\n" + "=" * 65)
print("  TABLE IV: MODEL PERFORMANCE COMPARISON")
print("=" * 65)
print(df_res.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 5. PER-CLASS METRICS FOR TABLE VII
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TABLE VII: PER-CLASS METRICS BREAKDOWN")
print("=" * 65)

target_models = ["Random Forest (Baseline)", "Random Forest + SMOTE", "Ordinal Logistic Regression", "XGBoost", "LightGBM"]
for m_name in target_models:
    rep = per_class_summary[m_name]
    print(f"\n--- {m_name} ---")
    for cls in ["POD", "Minor", "Moderate", "Serious"]:
        p = rep[cls]["precision"]
        r = rep[cls]["recall"]
        f = rep[cls]["f1-score"]
        print(f"  {cls:10s} | Precision: {p:.3f} | Recall: {r:.3f} | F1: {f:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SHAP ANALYSIS (on Random Forest + SMOTE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SHAP ANALYSIS")
print("=" * 65)

rf_smote = models["Random Forest + SMOTE"]
explainer = shap.TreeExplainer(rf_smote)
shap_values = explainer.shap_values(X_test)

# Handle list of matrices (one per class) or 3D array
if isinstance(shap_values, list):
    mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
else:
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.ndim == 3 else np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "Feature": feat_names,
    "SHAP_Importance": mean_abs_shap
}).sort_values(by="SHAP_Importance", ascending=False)

print("\n--- Top 10 Features by Mean |SHAP| ---")
print(shap_df.head(10).to_string(index=False))

# Permutation importances
perm_imp = permutation_importance(rf_smote, X_test, y_test, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    "Feature": feat_names,
    "Perm_Importance": perm_imp.importances_mean
}).sort_values(by="Perm_Importance", ascending=False)

print("\n--- Top 10 Features by Permutation Importance ---")
print(perm_df.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 7. ROC CURVE DATA FOR FIGURE 4
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ROC CURVE VALUES FOR TIKZ PLOT")
print("=" * 65)

for m_name in ["Random Forest + SMOTE", "Ordinal Logistic Regression", "LightGBM", "XGBoost"]:
    y_proba = prob_dict[m_name]
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc = auc(fpr, tpr)
    print(f"\n{m_name} (Micro AUC = {roc_auc:.3f}):")
    indices = np.linspace(0, len(fpr) - 1, 15, dtype=int)
    coords = " ".join([f"({fpr[i]:.3f},{tpr[i]:.3f})" for i in indices])
    print(f"  Coordinates: {coords}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPORT RESULTS TO JSON FOR TEX UPDATE
# ─────────────────────────────────────────────────────────────────────────────
output_data = {
    "summary": results_summary,
    "per_class": per_class_summary,
    "shap_top": shap_df.head(10).to_dict(orient="records")
}

import json
with open("experiment_results.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("\n" + "=" * 65)
print("  COMPLETED ALL EXPERIMENTS & EXPORTED RESULTS!")
print("=" * 65)
