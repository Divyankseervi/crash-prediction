"""
CrashPredict AI — Flask Backend Server
Runs the model, serves the website, and provides API endpoints.

Usage:
    cd "c:\\Users\\divya\\OneDrive\\Desktop\\prob project"
    .venv\\Scripts\\python.exe app.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory

# ─────────────────────────────────────────────────────────────────────────────
# Flask App Setup
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.', template_folder='.')

MODEL_RESULTS_FILE = 'model_results.json'
EXCEL_FILE = 'AV_accident_data__1_.xlsx'

def get_model_results():
    """Load model results from JSON file, regenerating if needed."""
    if not os.path.exists(MODEL_RESULTS_FILE):
        print("[SERVER] model_results.json not found. Running preprocessrohan.py ...")
        run_model_script()
    with open(MODEL_RESULTS_FILE, 'r') as f:
        return json.load(f)

def run_model_script():
    """Execute the preprocessing/training script to generate model_results.json"""
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__) or '.', 'preprocessrohan.py')
    print(f"[SERVER] Running {script_path} with {python_exe} ...")
    result = subprocess.run(
        [python_exe, script_path],
        cwd=os.path.dirname(__file__) or '.',
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    )
    if result.returncode != 0:
        print(f"[SERVER] Script FAILED:\n{result.stderr[-1000:]}")
        return False
    print("[SERVER] Script completed successfully!")
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Routes — Serve HTML Pages
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_file(filename):
    """Serve any file from the project directory (HTML, CSS, JS, images)."""
    return send_from_directory('.', filename)

# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints — Live Data from Python
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/results')
def api_results():
    """Return all model results as JSON."""
    try:
        data = get_model_results()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Run a live prediction using the trained model."""
    try:
        import joblib
        params = request.get_json()
        speed = float(params.get('speedLimit', 35))
        precrash = float(params.get('precrashSpeed', 10))
        mileage = float(params.get('mileage', 500))
        is_night = int(params.get('isNight', 0))
        is_wet = int(params.get('isWet', 0))
        is_highway = int(params.get('isHighway', 0))
        is_dark = int(params.get('isDark', 0))
        is_bad_weather = int(params.get('isBadWeather', 0))
        airbag = int(params.get('airbag', 0))

        # Load saved model and scaler
        rf = joblib.load('static/rf_model.joblib')
        scaler = joblib.load('static/scaler.joblib')

        # Load model_results to get feature names
        data = get_model_results()
        feat_names = data['dataset']['feature_names']

        # Build feature row
        row = {col: 0.0 for col in feat_names}
        speed_ratio = precrash / speed if speed > 0 else 0
        speed_night = precrash * is_night
        scaled = scaler.transform([[speed, mileage, precrash, speed_ratio, speed_night]])[0]
        row['Posted Speed Limit (MPH)'] = scaled[0]
        row['Mileage'] = scaled[1]
        row['SV Precrash Speed (MPH)'] = scaled[2]
        row['Speed_Ratio'] = scaled[3]
        row['Speed_Night'] = scaled[4]
        row['Is_Night'] = is_night
        row['Is_Wet'] = is_wet
        row['Is_Highway'] = is_highway
        row['Is_Dark'] = is_dark
        row['Is_BadWeather'] = is_bad_weather
        row['AirBag_Deployed'] = airbag
        row['Incident_Year'] = 2024
        row['Incident_Month'] = 6
        row['Is_OldVehicle'] = 0

        X_row = pd.DataFrame([row])[feat_names].astype(float)
        proba_raw = rf.predict_proba(X_row)[0]
        proba_full = np.zeros(4)
        for i, c in enumerate(rf.classes_):
            proba_full[c] = proba_raw[i]
        risk_index = sum(p * idx for idx, p in enumerate(proba_full))

        severity_labels = ['POD', 'Minor', 'Moderate', 'Serious']
        return jsonify({
            'probabilities': {severity_labels[i]: round(float(proba_full[i]), 4) for i in range(4)},
            'risk_index': round(float(risk_index), 4),
            'predicted_class': severity_labels[int(np.argmax(proba_full))],
            'model': 'Random Forest (500 trees)',
            'source': 'live_python_prediction'
        })
    except FileNotFoundError:
        return jsonify({"error": "Model files not found. Run preprocessrohan.py first."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    """Re-run the training script to refresh all results."""
    success = run_model_script()
    if success:
        return jsonify({"status": "success", "message": "Models retrained and results updated!"})
    else:
        return jsonify({"status": "error", "message": "Training script failed. Check console."}), 500

@app.route('/api/dataset-info')
def api_dataset_info():
    """Return raw dataset info directly from the Excel file."""
    try:
        df = pd.read_excel(EXCEL_FILE)
        return jsonify({
            "shape": list(df.shape),
            "columns": list(df.columns),
            "severity_counts": df['Severity'].value_counts().to_dict(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_rows": df.head(20).fillna('N/A').to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
# Main — Start the Server
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  CrashPredict AI — Flask Server")
    print("=" * 60)

    # Auto-generate model_results.json if it doesn't exist
    if not os.path.exists(MODEL_RESULTS_FILE):
        print("[SERVER] No model_results.json found. Running training script...")
        run_model_script()
    else:
        print(f"[SERVER] Found {MODEL_RESULTS_FILE}")

    if not os.path.exists('static/rf_model.joblib'):
        print("[SERVER] No saved model found. Running training script...")
        run_model_script()

    print("\n[SERVER] Starting Flask server...")
    print("[SERVER] Open http://localhost:5000 in your browser")
    print("[SERVER] Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
