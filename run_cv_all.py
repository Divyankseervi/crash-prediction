import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import scipy.optimize as opt
from scipy.special import expit
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(42)

df = pd.read_excel('backend/AV_accident_data__1_.xlsx')
severity_map = {"POD": 0, "Minor": 1, "Moderate": 2, "Serious": 3}
df["Severity_Enc"] = df["Severity"].map(severity_map)
df["Is_Night"] = (df["Incident_Time"] == "Night").astype(int)
df["Is_Wet"] = (df["Roadway_Surface"] == "Wet").astype(int)
df["Is_Highway"] = (df["Roadway_Type"] == "Highway").astype(int)
df["Is_Dark"] = df["Lighting"].apply(lambda x: 1 if "Dark" in str(x) else 0)
df["Is_BadWeather"] = df["Weather"].apply(lambda x: 0 if x in ["Clear", "Unknown"] else 1)
df["AirBag_Deployed"] = (df["Air_Bag"] == "Yes").astype(int)
df["Incident_Year"] = pd.to_datetime(df["Incident Date"], errors="coerce").dt.year.fillna(2023).astype(int)
df["Incident_Month"] = pd.to_datetime(df["Incident Date"], errors="coerce").dt.month.fillna(6).astype(int)

for col in ["Posted Speed Limit (MPH)", "SV Precrash Speed (MPH)", "Model Year"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

m_yr_full = df["Model Year"].median()
df["Is_OldVehicle"] = (df["Model Year"] < m_yr_full).fillna(0).astype(int)
cat_cols = ["Roadway_Type", "Weather", "Lighting", "Crash_With"]
df_ohe = pd.get_dummies(df[cat_cols], drop_first=True).astype(int)
base_num = ["Posted Speed Limit (MPH)", "Mileage", "SV Precrash Speed (MPH)"]
eng_features = ["Is_Night", "Is_Wet", "Is_Highway", "Is_Dark", "Is_BadWeather", "AirBag_Deployed", "Incident_Year", "Incident_Month"]
X_raw = pd.concat([df[base_num + ["Model Year"] + eng_features].astype(float), df_ohe], axis=1)
y = df["Severity_Enc"].values

num_features = ["Posted Speed Limit (MPH)", "Mileage", "SV Precrash Speed (MPH)", "Speed_Ratio", "Speed_Night"]
feat_names = num_features + eng_features + ["Is_OldVehicle"] + list(df_ohe.columns)

class CustomOLR:
    def __init__(self, n=4): self.n = n
    def _sigmoid(self, x): return expit(x)
    def _nll(self, p, X, y):
        b = p[:X.shape[1]]; t = np.cumsum(np.exp(p[X.shape[1]:])); Xb = X @ b; ll = 0.0
        for i, k in enumerate(y):
            if k == 0: prob = self._sigmoid(t[0] - Xb[i])
            elif k == self.n-1: prob = 1 - self._sigmoid(t[-1] - Xb[i])
            else: prob = self._sigmoid(t[k] - Xb[i]) - self._sigmoid(t[k-1] - Xb[i])
            ll += np.log(max(prob, 1e-9))
        return -ll
    def fit(self, X, y):
        p0 = np.zeros(X.shape[1] + self.n - 1)
        r = opt.minimize(self._nll, p0, args=(X, y), method='L-BFGS-B')
        self.coef_ = r.x[:X.shape[1]]; self.thresh_ = np.cumsum(np.exp(r.x[X.shape[1]:]))
        return self
    def predict_proba(self, X):
        Xb = X @ self.coef_; p = np.zeros((len(X), self.n))
        for i in range(len(X)):
            c = [0.0] + [self._sigmoid(t - Xb[i]) for t in self.thresh_] + [1.0]
            for k in range(self.n): p[i,k] = max(c[k+1] - c[k], 0)
        return p / p.sum(axis=1, keepdims=True)
    def predict(self, X): return np.argmax(self.predict_proba(X), axis=1)

def build_dnn(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

models_dict = {
    "Random Forest Baseline": (RandomForestClassifier(n_estimators=500, criterion="gini", class_weight="balanced", random_state=42), False),
    "Random Forest + SMOTE": (RandomForestClassifier(n_estimators=500, criterion="gini", random_state=42), True),
    "Gradient Boosting": (GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42), False),
    "DNN / MLP": ("DNN", False),
    "SVM": (SVC(kernel="rbf", C=1.0, probability=True, random_state=42), False),
    "KNN": (KNeighborsClassifier(n_neighbors=7, weights="distance"), False),
    "Naive Bayes": (GaussianNB(), False),
    "Custom OLR": (CustomOLR(), False)
}

def process(X_df, mp, ms, my):
    X = X_df.copy()
    X["Posted Speed Limit (MPH)"] = X["Posted Speed Limit (MPH)"].fillna(mp)
    X["SV Precrash Speed (MPH)"] = X["SV Precrash Speed (MPH)"].fillna(ms)
    X["Speed_Ratio"] = (X["SV Precrash Speed (MPH)"] / X["Posted Speed Limit (MPH)"].replace(0, np.nan)).fillna(0)
    X["Is_OldVehicle"] = (X["Model Year"] < my).fillna(0).astype(int)
    X["Speed_Night"] = X["SV Precrash Speed (MPH)"] * X["Is_Night"]
    X.drop(columns=["Model Year"], inplace=True)
    return X[feat_names]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {k: [] for k in models_dict.keys()}

for tr_idx, val_idx in skf.split(X_raw, y):
    X_f_tr, X_f_val = X_raw.iloc[tr_idx], X_raw.iloc[val_idx]
    y_f_tr, y_f_val = y[tr_idx], y[val_idx]
    
    mp = X_f_tr["Posted Speed Limit (MPH)"].median()
    ms = X_f_tr["SV Precrash Speed (MPH)"].median()
    my = X_f_tr["Model Year"].median()
    
    X_ti = process(X_f_tr, mp, ms, my)
    X_vi = process(X_f_val, mp, ms, my)
    
    sc = StandardScaler()
    X_ti[num_features] = sc.fit_transform(X_ti[num_features])
    X_vi[num_features] = sc.transform(X_vi[num_features])
    
    X_ts, y_ts = SMOTE(random_state=42, k_neighbors=min(4, len(y_f_tr[y_f_tr==3])-1)).fit_resample(X_ti, y_f_tr)
    
    for name, (model, use_smote) in models_dict.items():
        Xr = X_ts if use_smote else X_ti
        yr = y_ts if use_smote else y_f_tr
        
        if name == "DNN / MLP":
            dnn = build_dnn(Xr.shape[1])
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            dnn.fit(Xr, yr, epochs=200, validation_split=0.2, callbacks=[es], verbose=0)
            pred = np.argmax(dnn.predict(X_vi, verbose=0), axis=1)
        elif name == "Custom OLR":
            model.fit(Xr.values, yr)
            pred = model.predict(X_vi.values)
        else:
            model.fit(Xr, yr)
            pred = model.predict(X_vi)
        
        cv_results[name].append(accuracy_score(y_f_val, pred))

for name, res in cv_results.items():
    print(f"{name}: Folds={res}, Mean={np.mean(res):.4f}, Std={np.std(res):.4f}")
