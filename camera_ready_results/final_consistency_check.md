RESULT | SOURCE | VERIFIED? | NOTES
All metrics | camera_ready_reproducible_analysis.py | YES | Clean pipeline
Model Configs | camera_ready_reproducible_analysis.py | YES | Matches strict paper methodology
Leakage | None | YES | Preprocessing fitted only on X_train
SMOTE | camera_ready_reproducible_analysis.py | YES | Applied only to training set/folds
CV Executed | StratifiedKFold | YES | 5-fold loop implemented for all models
