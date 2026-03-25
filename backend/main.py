"""
CrashPredict AI — FastAPI Backend
Serves the frontend, provides prediction API, and manages model lifecycle.

Usage:
    cd "c:\\Users\\divya\\OneDrive\\Desktop\\prob project"
    .venv\\Scripts\\python.exe -m uvicorn backend.main:app --reload --port 8000

Then open http://localhost:8000 in your browser.
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
MODEL_RESULTS_FILE = BACKEND_DIR / "model_results.json"
EXCEL_FILE = BACKEND_DIR / "AV_accident_data__1_.xlsx"
SAVED_MODEL_PATH = BACKEND_DIR / "static" / "rf_model.joblib"
SAVED_SCALER_PATH = BACKEND_DIR / "static" / "scaler.joblib"
TRAINING_SCRIPT = BACKEND_DIR / "preprocessrohan.py"

# ─────────────────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CrashPredict AI",
    description="AI-Based Crash Severity Prediction for Autonomous Vehicles",
    version="2.0",
)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Cached model data (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
_cached_results = None
_cached_rf_model = None
_cached_scaler = None


def load_model_results():
    """Load model results JSON (cached)."""
    global _cached_results
    if _cached_results is None:
        if not MODEL_RESULTS_FILE.exists():
            run_training_script()
        with open(MODEL_RESULTS_FILE, "r") as f:
            _cached_results = json.load(f)
    return _cached_results


def load_rf_model():
    """Load saved Random Forest model and scaler (cached)."""
    global _cached_rf_model, _cached_scaler
    if _cached_rf_model is None:
        import joblib
        if not SAVED_MODEL_PATH.exists():
            run_training_script()
        _cached_rf_model = joblib.load(str(SAVED_MODEL_PATH))
        _cached_scaler = joblib.load(str(SAVED_SCALER_PATH))
    return _cached_rf_model, _cached_scaler


def run_training_script():
    """Execute preprocessrohan.py to train models and generate results."""
    global _cached_results, _cached_rf_model, _cached_scaler
    python_exe = sys.executable
    print(f"[BACKEND] Running training script: {TRAINING_SCRIPT}")
    result = subprocess.run(
        [python_exe, str(TRAINING_SCRIPT)],
        cwd=str(BACKEND_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        err = result.stderr[-2000:] if result.stderr else "Unknown error"
        print(f"[BACKEND] Training FAILED:\n{err}")
        raise RuntimeError(f"Training script failed: {err}")
    print("[BACKEND] Training completed successfully!")
    # Clear cache so it loads fresh
    _cached_results = None
    _cached_rf_model = None
    _cached_scaler = None


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models (request/response schemas)
# ─────────────────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    speedLimit: float = 35
    precrashSpeed: float = 10
    mileage: float = 500
    isNight: int = 0
    isWet: int = 0
    isHighway: int = 0
    isDark: int = 0
    isBadWeather: int = 0
    airbag: int = 0


class PredictionResponse(BaseModel):
    probabilities: dict
    risk_index: float
    predicted_class: str
    model: str
    source: str


# ─────────────────────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/results")
def get_results():
    """Return all model results as JSON."""
    try:
        data = load_model_results()
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def apply_domain_corrections(proba: np.ndarray, req: PredictionRequest) -> np.ndarray:
    """
    Apply physics & domain-knowledge corrections to model predictions.
    
    Tree-based models CANNOT extrapolate beyond training data range (~0-80 mph).
    At extreme speeds (e.g. 1000 mph), the RF model assigns the same leaf as
    ~80 mph, which is clearly wrong. This function applies corrections based on:
    
    1. Speed ratio (precrash / limit) — speeding severity
    2. Absolute pre-crash speed — kinetic energy ∝ v²  
    3. Combined risk conditions — multiplicative hazards
    4. Airbag deployment — strong injury indicator
    """
    proba = proba.copy()  # Don't modify original
    speed = req.precrashSpeed
    limit = req.speedLimit if req.speedLimit > 0 else 35
    speed_ratio = speed / limit

    # ── Factor 1: Speed Ratio (speeding) ──
    # Training data max speed ratio is ~2-3x. Beyond that, severity increases.
    if speed_ratio > 1.0:
        # Progressively shift probability from POD → more severe
        # At 2x speed limit: moderate shift; at 5x+: extreme shift
        overspeed_factor = min((speed_ratio - 1.0) / 3.0, 1.0)  # 0→1 over ratio 1→4
        shift = proba[0] * overspeed_factor * 0.7  # Take up to 70% from POD
        proba[0] -= shift
        proba[1] += shift * 0.35  # 35% to Minor
        proba[2] += shift * 0.40  # 40% to Moderate
        proba[3] += shift * 0.25  # 25% to Serious

    # ── Factor 2: Absolute Speed (kinetic energy) ──
    # KE = ½mv², so danger grows quadratically with speed
    # Training data max ~80mph. Corrections kick in beyond that.
    if speed > 80:
        # Sigmoid-like: ramps from 0 at 80mph to ~1.0 at 200+mph
        extreme_factor = min(1.0, (speed - 80) / 120)
        extreme_factor = extreme_factor ** 0.7  # Make it ramp faster

        # At extreme speeds, shift heavily toward Serious
        shift = proba[0] * extreme_factor * 0.85
        proba[0] -= shift
        minor_shift = proba[1] * extreme_factor * 0.5
        proba[1] -= minor_shift
        proba[2] += (shift + minor_shift) * 0.35
        proba[3] += (shift + minor_shift) * 0.65

    # ── Factor 3: Combined Risk Conditions ──
    # Multiple hazards compound: night + wet + dark + speeding = very dangerous
    risk_count = sum([
        req.isNight, req.isWet, req.isDark, req.isBadWeather,
        1 if speed_ratio > 1.5 else 0,
        req.airbag,  # Airbag deployed = impact was severe
    ])
    if risk_count >= 3:
        compound_factor = min((risk_count - 2) / 4.0, 0.6)
        shift = proba[0] * compound_factor
        proba[0] -= shift
        proba[1] += shift * 0.3
        proba[2] += shift * 0.4
        proba[3] += shift * 0.3

    # ── Factor 4: Airbag as strong severity indicator ──
    if req.airbag:
        # If airbag deployed, at minimum this is a significant crash
        if proba[0] > 0.5:
            shift = (proba[0] - 0.3) * 0.5
            proba[0] -= shift
            proba[1] += shift * 0.5
            proba[2] += shift * 0.35
            proba[3] += shift * 0.15

    # Ensure valid probability distribution
    proba = np.clip(proba, 0, 1)
    total = proba.sum()
    if total > 0:
        proba /= total

    return proba


@app.post("/api/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """Run a LIVE prediction using the trained RF model + domain corrections."""
    try:
        rf, scaler = load_rf_model()
        data = load_model_results()
        feat_names = data["dataset"]["feature_names"]
        severity_labels = ["POD", "Minor", "Moderate", "Serious"]

        # Build feature row
        row = {col: 0.0 for col in feat_names}
        speed_ratio = req.precrashSpeed / req.speedLimit if req.speedLimit > 0 else 0
        speed_night = req.precrashSpeed * req.isNight

        scaled = scaler.transform(
            [[req.speedLimit, req.mileage, req.precrashSpeed, speed_ratio, speed_night]]
        )[0]

        row["Posted Speed Limit (MPH)"] = float(scaled[0])
        row["Mileage"] = float(scaled[1])
        row["SV Precrash Speed (MPH)"] = float(scaled[2])
        row["Speed_Ratio"] = float(scaled[3])
        row["Speed_Night"] = float(scaled[4])
        row["Is_Night"] = req.isNight
        row["Is_Wet"] = req.isWet
        row["Is_Highway"] = req.isHighway
        row["Is_Dark"] = req.isDark
        row["Is_BadWeather"] = req.isBadWeather
        row["AirBag_Deployed"] = req.airbag
        row["Incident_Year"] = 2024
        row["Incident_Month"] = 6
        row["Is_OldVehicle"] = 0

        X_row = pd.DataFrame([row])[feat_names].astype(float)
        proba_raw = rf.predict_proba(X_row)[0]
        proba_full = np.zeros(4)
        for i, c in enumerate(rf.classes_):
            proba_full[c] = proba_raw[i]

        # Apply domain-knowledge corrections for out-of-range inputs
        proba_corrected = apply_domain_corrections(proba_full, req)

        risk_index = sum(p * idx for idx, p in enumerate(proba_corrected))

        return PredictionResponse(
            probabilities={
                severity_labels[i]: round(float(proba_corrected[i]), 4)
                for i in range(4)
            },
            risk_index=round(float(risk_index), 4),
            predicted_class=severity_labels[int(np.argmax(proba_corrected))],
            model="Random Forest (500 trees) + Domain Corrections",
            source="live_python_fastapi",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/retrain")
def retrain():
    """Re-run the training script to refresh all models and results."""
    try:
        run_training_script()
        return {"status": "success", "message": "Models retrained and results updated!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dataset")
def get_dataset_info():
    """Return raw dataset info directly from the Excel file."""
    try:
        df = pd.read_excel(str(EXCEL_FILE))
        return {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "severity_counts": df["Severity"].value_counts().to_dict(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_rows": json.loads(
                df.head(20).fillna("N/A").to_json(orient="records")
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Serve Frontend Static Files
# ─────────────────────────────────────────────────────────────────────────────
# Mount frontend as static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend_static")


# Serve model_results.json from backend dir (frontend fetches this)
@app.get("/model_results.json")
def serve_model_results():
    if MODEL_RESULTS_FILE.exists():
        return FileResponse(str(MODEL_RESULTS_FILE))
    raise HTTPException(status_code=404, detail="model_results.json not found. Call /api/retrain first.")


# Serve HTML pages
@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/{page}.html")
def serve_page(page: str):
    filepath = FRONTEND_DIR / f"{page}.html"
    if filepath.exists():
        return FileResponse(str(filepath))
    raise HTTPException(status_code=404, detail=f"Page {page}.html not found")


# ─────────────────────────────────────────────────────────────────────────────
# Startup Event
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    print("=" * 60)
    print("  CrashPredict AI — FastAPI Backend v2.0")
    print("=" * 60)
    if not MODEL_RESULTS_FILE.exists():
        print("[BACKEND] model_results.json not found. Training models...")
        try:
            run_training_script()
        except Exception as e:
            print(f"[BACKEND] Warning: {e}")
    else:
        print(f"[BACKEND] Found {MODEL_RESULTS_FILE.name}")

    if SAVED_MODEL_PATH.exists():
        print(f"[BACKEND] Found saved RF model")
    else:
        print("[BACKEND] Warning: No saved model. Call /api/retrain or run preprocessrohan.py")

    print(f"[BACKEND] Frontend dir: {FRONTEND_DIR}")
    print(f"[BACKEND] Server ready at http://localhost:8000")
    print()
