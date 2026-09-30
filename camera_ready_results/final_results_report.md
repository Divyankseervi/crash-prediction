## 1. DATASET
* Total records: 891
* Class counts: {'POD': 808, 'Minor': 62, 'Moderate': 14, 'Serious': 7}
## 2. FEATURE ENGINEERING
* `Is_OldVehicle` exact definition: `Model Year < median(Model Year)` (Median = 2021.0). If paper says '>5 years', this is an inconsistency.
* `Speed_Ratio` exact definition: `SV Precrash Speed (MPH) / Posted Speed Limit (MPH)`. It is completely unitless.
## 3. TRAIN/TEST SPLIT
* test_size=0.2, random_state=42, stratify=y
* Train size: 712
* Test size: 179
* Training class distribution: {0: 646, 1: 49, 2: 11, 3: 6}
* Test class distribution: {0: 162, 1: 13, 2: 3, 3: 1}
* Serious/Fatal samples in hold-out test set: 1
## 4. PREPROCESSING
* Scaler and median imputers were strictly fitted only on X_train and transformed on X_test to avoid data leakage.
## 5. SMOTE
* Applied strictly to the training data.
* Original train counts: {0: 646, 1: 49, 2: 11, 3: 6}
* Post-SMOTE train counts: {0: 646, 1: 646, 3: 646, 2: 646}
## 6. MODEL CONFIGURATIONS
* RF Baseline: 500 trees, gini, balanced weights.
* RF+SMOTE: 500 trees, gini, trained on SMOTE split.
* GBM: 300 estimators, lr=0.05, max_depth=4.
* DNN: Keras Sequential, (128, 64, 32) layers, ReLU, Dropout 0.3, Adam 0.001, early stopping with 20% train validation split. Deterministic ops enabled.
* SVM: RBF, C=1.0, probability=True (sklearn uses Platt scaling internally).
* KNN: K=7, weights=distance.
* Naive Bayes: GaussianNB.
* OLR: L-BFGS-B, cumulative-logit.
## 9. BRIER SCORE
* Multiclass Brier Formula: `mean(sum((one_hot_true - predicted_probability)^2, axis=1))`
## 7. HOLD-OUT RESULTS
**Random Forest Baseline**
Acc: 0.9050 | W-F1: 0.8599 | M-F1: 0.2375 | Brier: 0.1726 | W-Kappa: 0.0000
CM:
[[162   0   0   0]
 [ 13   0   0   0]
 [  3   0   0   0]
 [  1   0   0   0]]

**Random Forest + SMOTE**
Acc: 0.8603 | W-F1: 0.8371 | M-F1: 0.2312 | Brier: 0.2318 | W-Kappa: -0.0536
CM:
[[154   6   0   2]
 [ 13   0   0   0]
 [  3   0   0   0]
 [  1   0   0   0]]

**Gradient Boosting**
Acc: 0.8771 | W-F1: 0.8483 | M-F1: 0.2343 | Brier: 0.2252 | W-Kappa: 0.0288
CM:
[[157   3   0   2]
 [ 13   0   0   0]
 [  2   1   0   0]
 [  1   0   0   0]]

**DNN / MLP**
Acc: 0.9050 | W-F1: 0.8599 | M-F1: 0.2375 | Brier: 0.1856 | W-Kappa: 0.0000
CM:
[[162   0   0   0]
 [ 13   0   0   0]
 [  3   0   0   0]
 [  1   0   0   0]]

**SVM**
Acc: 0.9050 | W-F1: 0.8599 | M-F1: 0.2375 | Brier: 0.1792 | W-Kappa: 0.0000
CM:
[[162   0   0   0]
 [ 13   0   0   0]
 [  3   0   0   0]
 [  1   0   0   0]]

**KNN**
Acc: 0.9050 | W-F1: 0.8624 | M-F1: 0.2382 | Brier: 0.1931 | W-Kappa: 0.1080
CM:
[[162   0   0   0]
 [ 13   0   0   0]
 [  2   1   0   0]
 [  1   0   0   0]]

**Naive Bayes**
Acc: 0.0838 | W-F1: 0.1235 | M-F1: 0.0672 | Brier: 1.8122 | W-Kappa: 0.0156
CM:
[[ 11  22 106  23]
 [  0   2   8   3]
 [  0   0   2   1]
 [  0   0   1   0]]

**Custom OLR**
Acc: 0.9050 | W-F1: 0.8599 | M-F1: 0.2375 | Brier: 0.1753 | W-Kappa: 0.0000
CM:
[[162   0   0   0]
 [ 13   0   0   0]
 [  3   0   0   0]
 [  1   0   0   0]]


## 8. SERIOUS/FATAL RESULTS
**Random Forest Baseline** (Serious): TP=0, FP=0, FN=1 | Prec: 0.0000, Rec: 0.0000, F1: 0.0000 | Predicted as Class 3: 0
**Random Forest + SMOTE** (Serious): TP=0, FP=2, FN=1 | Prec: 0.0000, Rec: 0.0000, F1: 0.0000 | Predicted as Class 3: 2
**Gradient Boosting** (Serious): TP=0, FP=2, FN=1 | Prec: 0.0000, Rec: 0.0000, F1: 0.0000 | Predicted as Class 3: 2
**DNN / MLP** (Serious): TP=0, FP=0, FN=1 | Prec: 0.0000, Rec: 0.0000, F1: 0.0000 | Predicted as Class 3: 0
**SVM** (Serious): TP=0, FP=0, FN=1 | Prec: 0.0000, Rec: 0.0000, F1: 0.0000 | Predicted as Class 3: 0
**KNN** (Serious): TP=0, FP=0, FN=1 | Prec: 0.0000, Rec: 0.0000, F1: 0.0000 | Predicted as Class 3: 0
**Naive Bayes** (Serious): TP=0, FP=27, FN=1 | Prec: 0.0000, Rec: 0.0000, F1: 0.0000 | Predicted as Class 3: 27
**Custom OLR** (Serious): TP=0, FP=0, FN=1 | Prec: 0.0000, Rec: 0.0000, F1: 0.0000 | Predicted as Class 3: 0

*If all models have 0% recall, it is because there is exactly 1 test example and it was not detected.*
## 10. CROSS-VALIDATION
**Random Forest Baseline**
Folds: [0.8994, 0.9101, 0.9045, 0.8989, 0.9045]
Mean: 0.9035
Std: 0.0041

**Random Forest + SMOTE**
Folds: [0.8603, 0.882, 0.8764, 0.8764, 0.8989]
Mean: 0.8788
Std: 0.0124

**Gradient Boosting**
Folds: [0.8883, 0.8708, 0.8764, 0.8764, 0.8764]
Mean: 0.8777
Std: 0.0057

**DNN / MLP**
Folds: [0.905, 0.9101, 0.9101, 0.9045, 0.9045]
Mean: 0.9068
Std: 0.0027

**SVM**
Folds: [0.905, 0.9101, 0.9101, 0.9045, 0.9045]
Mean: 0.9068
Std: 0.0027

**KNN**
Folds: [0.905, 0.9101, 0.9045, 0.8989, 0.9045]
Mean: 0.9046
Std: 0.0036

**Naive Bayes**
Folds: [0.0838, 0.0899, 0.0618, 0.0562, 0.0787]
Mean: 0.0741
Std: 0.0129

**Custom OLR**
Folds: [0.905, 0.9101, 0.9045, 0.9045, 0.9045]
Mean: 0.9057
Std: 0.0022


## 11. OLR ODDS RATIOS
* The cumulative logit form used is `prob_lower = expit(thresh - X*beta)`.
* Therefore, `prob_higher = expit(X*beta - thresh)`. A positive Beta increases the probability of higher severity classes.
* Thus, `OR = exp(beta) > 1` means HIGHER cumulative odds of severity.

                  Feature      Beta  OddsRatio
 Posted Speed Limit (MPH)  0.521184   1.684020
                  Mileage  0.045476   1.046526
  SV Precrash Speed (MPH) -0.569052   0.566062
              Speed_Ratio  0.390629   1.477910
              Speed_Night  0.233787   1.263376
                 Is_Night -0.134881   0.873820
                   Is_Wet -0.387334   0.678864
               Is_Highway -0.306006   0.736382
                  Is_Dark -0.075011   0.927733
            Is_BadWeather -0.464359   0.628538
          AirBag_Deployed  1.251838   3.496765
            Incident_Year -0.000774   0.999226
           Incident_Month -0.047175   0.953921
            Is_OldVehicle  0.452500   1.572238
Roadway_Type_Intersection  0.470570   1.600907
      Roadway_Type_Street -0.164167   0.848600
       Weather_Cloudy/fog  0.128168   1.136744
        Weather_Snow/rain -0.592533   0.552925
          Weather_Unknown -0.344222   0.708771
Lighting_Dark/Not_Lighted -0.462491   0.629713
        Lighting_Daylight  0.075414   1.078330
         Lighting_Unknown  0.000000   1.000000
        Crash_With_Object  1.342823   3.829841
            Crash_With_PC  0.371737   1.450252
    Crash_With_Pedestrian  1.014723   2.758601
       Crash_With_Unknown -1.116281   0.327496
*Note: These are associative, not causal.*
## 12. CHI-SQUARE
- Crash_With: Chi2 = 54.57, p-value = 2.1589e-07, dof = 12, table shape = (5, 4)
- Lighting: Chi2 = 8.04, p-value = 5.3033e-01, dof = 9, table shape = (4, 4)
- Roadway_Type: Chi2 = 6.70, p-value = 3.4950e-01, dof = 6, table shape = (3, 4)
- Is_BadWeather: Chi2 = 6.33, p-value = 9.6593e-02, dof = 3, table shape = (2, 4)
- Is_OldVehicle: Chi2 = 12.55, p-value = 5.7063e-03, dof = 3, table shape = (2, 4)

## 13. BAYESIAN CONDITIONAL RISK
* Baseline: 7 / 891 = 0.00786
* Night: 4 / 234 = 0.01709
* Highway: 1 / 59 = 0.01695
* Bad Weather: 2 / 153 = 0.01307
* Old Vehicle: 0 / 157 = 0.00000
## 14. PERMUTATION IMPORTANCE
Top 10:
                  Feature  Importance
      Roadway_Type_Street    0.012291
Roadway_Type_Intersection    0.011732
 Posted Speed Limit (MPH)    0.006704
           Incident_Month    0.002793
                  Mileage    0.002235
        Lighting_Daylight    0.001676
        Crash_With_Object    0.001117
          AirBag_Deployed    0.000559
            Crash_With_PC    0.000559
                   Is_Wet    0.000000
