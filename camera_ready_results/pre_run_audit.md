# Pre-Run Audit

## 1. Features Currently Used
- `Posted Speed Limit (MPH)`, `Mileage`, `SV Precrash Speed (MPH)`
- `Is_Night`, `Is_Wet`, `Is_Highway`, `Is_Dark`, `Is_BadWeather`, `AirBag_Deployed`
- `Incident_Year`, `Incident_Month`, `Is_OldVehicle`
- `Speed_Ratio` (`SV Precrash Speed (MPH)` / `Posted Speed Limit (MPH)`)
- `Speed_Night` (`SV Precrash Speed (MPH)` * `Is_Night`)
- One-hot encoded categorical variables: `Roadway_Type`, `Weather`, `Lighting`, `Crash_With`

## 2. Model Configurations Stated in the Paper
- **Random Forest Baseline**: 500 trees, Gini index, balanced class weights.
- **Random Forest + SMOTE**: 500 trees, Gini index.
- **Gradient Boosting**: 300 estimators, learning rate 0.05, max depth 4.
- **DNN / MLP**: 3 hidden layers (128, 64, 32), ReLU, Adam, learning rate = 0.001, dropout = 0.3, early stopping based on validation loss.
- **SVM**: RBF kernel, C=1.0, probability=True (Platt scaling).
- **KNN**: n_neighbors=7, distance weighted voting.
- **Naive Bayes**: Gaussian Naive Bayes.
- **Custom OLR**: 4 ordered severity classes, L-BFGS-B, cumulative-logit formulation.

## 3. Configurations Actually Present in the Old Code (`full_paper_analysis.py`)
- **Random Forest Baseline**: `RandomForestClassifier(n_estimators=500, random_state=42)` (No class weights).
- **Random Forest + SMOTE**: `RandomForestClassifier(n_estimators=500, random_state=42)` on SMOTE data.
- **Gradient Boosting**: `GradientBoostingClassifier(n_estimators=200, random_state=42)` (Default LR=0.1, max_depth=3).
- **DNN / MLP**: `MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)` (No dropout, no early stopping).
- **SVM**: `SVC(probability=True, random_state=42)`.
- **KNN**: `KNeighborsClassifier(n_neighbors=5)` (Uniform weights).
- **Naive Bayes**: `GaussianNB()`.
- **Custom OLR**: Custom BFGS implementation.

## 4. Inconsistencies that Exist
- **Data Leakage**: The old code scaled and imputed the entire dataset before applying `train_test_split`.
- **Model Configurations**: The old code did not implement the exact configurations claimed in the paper (e.g., missing balanced class weights for RF, wrong depth/estimators for GBM, missing dropout/early stopping/layers for DNN, wrong K and weights for KNN).
- **Cross-Validation**: 5-fold CV was never actually executed in a loop in the old code.
- **SMOTE during CV**: SMOTE was applied globally before any CV instead of independently inside each training fold (if CV had existed).
- **Is_OldVehicle Definition**: The code defines it as `< median(Model Year)` (which is prior to 2021). The paper refers to it differently.

## 5. Exactly What Will Be Corrected in the NEW Analysis Script
- I will enforce a strict order of operations: Train/Test split first, then fitting preprocessors (Scalers, Imputers) only on the training set, and applying them to the test set to completely eliminate data leakage.
- I will implement exactly the model configurations stated in the paper (including a Keras-based DNN to properly support Dropout and Early Stopping).
- I will implement a proper 5-fold Stratified CV loop for RF+SMOTE, ensuring SMOTE is applied *only* inside each training fold.
- I will calculate and report proper Multiclass Brier Scores and Weighted F1 metrics.
- I will report actual Bayesian risks and Chi-Square values directly from the unmodified dataset.
- I will verify the custom OLR's odds ratios mathematically.

## 6. Confirmation
I confirm that the ORIGINAL project files (including original scripts, JSONs, and notebooks) will remain **100% untouched**. All operations will be strictly confined to the new script `camera_ready_reproducible_analysis.py` and the outputs will be placed entirely in the `camera_ready_results/` directory.
