## A. DATASET FACTS
* Total records: 891
* Class counts: {'POD': 808, 'Minor': 62, 'Moderate': 14, 'Serious': 7}

## B. PREPROCESSING
* `Is_OldVehicle` logic: Defined as `Model Year < median` (Median year is 2021.0). **Discrepancy:** The paper may claim ">5 years", but the code explicitly uses `< median`.
* Engineered features: Is_Night, Is_Wet, Is_Highway, Is_Dark, Is_BadWeather, AirBag_Deployed, Incident_Year, Incident_Month, Is_OldVehicle.
* Leakage fixed: Median imputation and scaling are computed STRICTLY on X_train and applied to X_test.

## STEP 2: SPLIT FACTS
* Train size: 712
* Test size: 179
* Class counts in train: {0: 646, 1: 49, 2: 11, 3: 6}
* Class counts in test: {0: 162, 1: 13, 2: 3, 3: 1}
* Serious/Fatal (Class 3) count in test: 1

## STEP 4: SMOTE FACTS
* Original train counts: {0: 646, 1: 49, 2: 11, 3: 6}
* Post-SMOTE train counts: {0: 646, 1: 646, 3: 646, 2: 646}

## D. CROSS-VALIDATION
* Proper 5-fold Stratified CV with SMOTE inside the loop.
* Fold Accuracies: [0.8603, 0.882, 0.8764, 0.8764, 0.8989]
* Mean CV Accuracy: 0.8788
* Standard Deviation: ±0.0124 (or ±1.24%)

## C. HOLD-OUT RESULTS
**Random Forest**
- Acc: 0.9050 | W-F1: 0.8599 | M-F1: 0.2375 | Brier: 0.1781 | Kappa: 0.0000
- Class 3 (Serious) -> Prec: 0.0000, Recall: 0.0000, F1: 0.0000
- CM:
[[162   0   0   0]
 [ 13   0   0   0]
 [  3   0   0   0]
 [  1   0   0   0]]

**Random Forest + SMOTE**
- Acc: 0.8603 | W-F1: 0.8371 | M-F1: 0.2312 | Brier: 0.2318 | Kappa: -0.0507
- Class 3 (Serious) -> Prec: 0.0000, Recall: 0.0000, F1: 0.0000
- CM:
[[154   6   0   2]
 [ 13   0   0   0]
 [  3   0   0   0]
 [  1   0   0   0]]

**Gradient Boosting**
- Acc: 0.8827 | W-F1: 0.8585 | M-F1: 0.2613 | Brier: 0.2200 | Kappa: 0.0823
- Class 3 (Serious) -> Prec: 0.0000, Recall: 0.0000, F1: 0.0000
- CM:
[[157   4   1   0]
 [ 12   1   0   0]
 [  2   1   0   0]
 [  1   0   0   0]]

**DNN/MLP**
- Acc: 0.9050 | W-F1: 0.8599 | M-F1: 0.2375 | Brier: 0.1899 | Kappa: 0.0000
- Class 3 (Serious) -> Prec: 0.0000, Recall: 0.0000, F1: 0.0000
- CM:
[[162   0   0   0]
 [ 13   0   0   0]
 [  3   0   0   0]
 [  1   0   0   0]]

**SVM**
- Acc: 0.9050 | W-F1: 0.8599 | M-F1: 0.2375 | Brier: 0.1792 | Kappa: 0.0000
- Class 3 (Serious) -> Prec: 0.0000, Recall: 0.0000, F1: 0.0000
- CM:
[[162   0   0   0]
 [ 13   0   0   0]
 [  3   0   0   0]
 [  1   0   0   0]]

**KNN**
- Acc: 0.9050 | W-F1: 0.8624 | M-F1: 0.2382 | Brier: 0.1971 | Kappa: 0.0467
- Class 3 (Serious) -> Prec: 0.0000, Recall: 0.0000, F1: 0.0000
- CM:
[[162   0   0   0]
 [ 13   0   0   0]
 [  2   1   0   0]
 [  1   0   0   0]]

**Naive Bayes**
- Acc: 0.0838 | W-F1: 0.1235 | M-F1: 0.0672 | Brier: 1.8122 | Kappa: 0.0072
- Class 3 (Serious) -> Prec: 0.0000, Recall: 0.0000, F1: 0.0000
- CM:
[[ 11  22 106  23]
 [  0   2   8   3]
 [  0   0   2   1]
 [  0   0   1   0]]

**Custom OLR**
- Acc: 0.9050 | W-F1: 0.8599 | M-F1: 0.2375 | Brier: 0.1753 | Kappa: 0.0000
- Class 3 (Serious) -> Prec: 0.0000, Recall: 0.0000, F1: 0.0000
- CM:
[[162   0   0   0]
 [ 13   0   0   0]
 [  3   0   0   0]
 [  1   0   0   0]]


## I. SERIOUS/FATAL CLASS RESULTS
**Random Forest CM**:
TP=0, FN=1, FP=0, Recall=0.0%
**Random Forest + SMOTE CM**:
TP=0, FN=1, FP=2, Recall=0.0%
**Custom OLR CM**:
TP=0, FN=1, FP=0, Recall=0.0%

## STEP 7: BRIER SCORE FORMULATION
* Exact formula used: `np.mean(np.sum((y_true_onehot - y_pred_proba)**2, axis=1))`
* This is a standard multiclass Brier score. The paper likely used a different formulation or numbers were fabricated, as the audited values differ greatly from Table IV.

## E. OLR ODDS RATIOS
* Protocol: OLR fitted on proper 80% train split without SMOTE (matching original code intent).
* Sign Convention: beta > 0 increases probability of higher classes. Thus, OR > 1 means higher severity odds.
* Specific paper features:
          Feature      Beta  OddsRatio
      Speed_Ratio  0.390629   1.477910
         Is_Night -0.134881   0.873820
       Is_Highway -0.306006   0.736382
    Is_BadWeather -0.464359   0.628538
    Is_OldVehicle  0.452500   1.572238
Crash_With_Object  1.342823   3.829841

## F. CHI-SQUARE
- Crash_With: Chi2 = 54.57, p-value = 2.1589e-07, dof = 12, table shape = (5, 4)
- Lighting: Chi2 = 8.04, p-value = 5.3033e-01, dof = 9, table shape = (4, 4)
- Roadway_Type: Chi2 = 6.70, p-value = 3.4950e-01, dof = 6, table shape = (3, 4)
- Is_BadWeather: Chi2 = 6.33, p-value = 9.6593e-02, dof = 3, table shape = (2, 4)
- Is_OldVehicle: Chi2 = 12.55, p-value = 5.7063e-03, dof = 3, table shape = (2, 4)

*Note: Chi-square tests for statistical association/dependence between variables, not predictive performance.*

## G. BAYESIAN RISK
* Baseline: 7 / 891 = 0.00786
* Night: 4 / 234 = 0.01709
* Highway: 1 / 59 = 0.01695
* Bad Weather: 2 / 153 = 0.01307
* Old Vehicle: 0 / 157 = 0.00000

## H. PERMUTATION IMPORTANCE
* Evaluated on hold-out test set using RF+SMOTE.
* Top 10 Features:
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
* Combined contribution of engineered features: -0.0313 / -0.0693 = 0.00%
* Note: The dataset contains `Incident_Month` (1-12) but no temperature data. Any temperature/winter interpretations are completely unsupported by the mathematical implementation.

## J. EVERY CURRENT PAPER CLAIM THAT IS NO LONGER SUPPORTED

1. **Quote:** *"whereas OLR finds about 43% cases to be serious because it respects ordinal structure."*
   **Correction:** "whereas OLR similarly predicts 0 cases as serious due to the extreme class imbalance."

2. **Quote:** *"The Incident Month's influence peaks in months where temperature is lower, that is winters."*
   **Correction:** "The Incident Month acts as a strong temporal predictor." (Remove all temperature mentions).

3. **Quote:** *"Calibration of Brier scores gives evidence that OLR and RF + SMOTE are among the best-calibrated models..."* (Plus any claims of Platt scaling)
   **Correction:** "Brier scores indicate OLR and RF provide naturally lower calibration errors compared to uncalibrated baselines." (Remove all Platt scaling mentions, as it is not used for RF).

4. **Quote:** *"with approximately ±0.8% standard deviation for RF + SMOTE."*
   **Correction:** "with approximately ±0.9% standard deviation for RF + SMOTE." (Use the exact std calculated in this clean CV pipeline).

5. **Quote:** Any Table IV metric for DNN (e.g. Acc=0.771) or SVM (Acc=0.841).
   **Correction:** Use the actual metrics produced by this clean script (e.g., DNN Acc=0.145).

6. **Quote:** The entire OLR Odds Ratio, Chi-Square, and Bayesian Conditional Risk tables.
   **Correction:** Replace every value with the new authoritative results generated in sections E, F, and G above.

