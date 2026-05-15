# Spaceship Titanic — Ensemble Machine Learning Pipeline

> **Kaggle Competition**: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
> **Final Rank**: 154 / 2,500+ &emsp; **Public Score**: 0.81155 (Accuracy)
> **Team**: 碱基互补配队@MLW

## Overview

Binary classification pipeline to predict which passengers were transported to an alternate dimension during the Spaceship Titanic's catastrophic voyage. Our solution combines **hierarchical missing-value imputation**, **47 engineered features**, **5 ML models** with systematic hyperparameter tuning, and a **3-model weighted ensemble** with per-group threshold calibration.

### Models Implemented

| Model                               | Type              |   OOF Accuracy   |     ROC-AUC     | Training Time |
| ----------------------------------- | ----------------- | :--------------: | :--------------: | :-----------: |
| **LightGBM**                  | Gradient Boosting | **0.8157** |      0.9070      |      42s      |
| CatBoost                            | Gradient Boosting |      0.8154      | **0.9084** |     1070s     |
| XGBoost                             | Gradient Boosting |      0.8092      |      0.9053      |     1064s     |
| Logistic Regression                 | Linear (baseline) |      0.7853      |      0.8486      |      68s      |
| Random Forest                       | Bagging Ensemble  |      0.7585      |      0.8681      |      84s      |
| **Ensemble (L+C+X + Calib.)** | Weighted Blend    | **0.8132** |        —        |      —      |

## Repository Structure

```
.
├── spaceship_titanic_ensemble.py   # Original 3-model ensemble (end-to-end)
├── preprocessing.py                # Standalone data preprocessing pipeline
├── eda.py                          # Exploratory Data Analysis (11 plots)
├── model_training.py               # 5 models + RandomizedSearchCV + 10-fold CV
├── evaluation.py                   # Model comparison, metrics, visualizations
├── FINAL_PROJECT_REPORT.tex        # LaTeX project report
└── README.md                       # This file
```

## Setup

### Requirements

Python 3.9+ with the following packages:

```
numpy  pandas  scikit-learn  scipy  matplotlib  seaborn
lightgbm  xgboost  catboost  joblib
```

### Install

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn lightgbm xgboost catboost joblib
```

### Data

Download the competition data from [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/data) and place all three files in a `data/` directory:

```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Alternatively, if you already have them in `project  information/spaceship-titanic dataset/`, the scripts will auto-detect them there.

---

## How to Run

Each script is self-contained and can be run independently. The recommended order:

### 1. Exploratory Data Analysis

```bash
python eda.py
```

**What it does**: Generates 11 publication-quality plots across 8 analytical dimensions: dataset overview, missing values, numeric distributions, categorical associations, correlation analysis, outlier detection, group/cabin analysis, and spend behaviour.

**Output**: `eda_plots/` directory with 11 PNG files.

**Runtime**: ~30 seconds.

---

### 2. Preprocessing (standalone verification)

```bash
python preprocessing.py
```

**What it does**: Runs the full preprocessing pipeline — column decomposition (PassengerId → GroupId/GroupNumber/GroupMember, Cabin → Deck/Num/Side, Name → FirstName/Surname), hierarchical missing-value imputation using cascading group-mode (categorical) and group-median (numeric) strategies, CryoSleep inference from spend patterns, feature engineering (47 features: 35 numeric + 12 categorical), and train/validation splitting.

**Output**: Console summary showing data shapes, feature counts, and missing-value report.

**Runtime**: ~10 seconds.

---

### 3. Model Training with Hyperparameter Tuning

```bash
python model_training.py
```

**What it does**: Trains all 5 models with:

- **RandomizedSearchCV** (30 iterations, 3-fold inner CV, scoring by ROC-AUC) on the first fold to find optimal hyperparameters
- **Stratified 10-fold cross-validation** (5 splits × 2 seeds: 2026, 42) for robust evaluation
- **Per-model data encoding**: LightGBM uses native categorical dtypes, CatBoost uses string-type cats with index specification, XGBoost/RF/LogisticRegression use one-hot encoding with sparse matrices
- Training time tracking per model per fold

**Output**: `model_outputs/` directory containing:

- `oof_model_probabilities.csv` — out-of-fold probabilities for all models
- `submission_probabilities.csv` — test-set probabilities for all models
- `hyperparameter_tuning_results.csv` — best params and CV scores
- `model_training_times.csv` — training time comparison
- `result_*.joblib` — serialized model results

**Runtime**: ~35 minutes (all 5 models; LightGBM alone: ~40s).

---

### 4. Evaluation and Model Comparison

```bash
python evaluation.py
```

**What it does**: Loads OOF predictions from `model_outputs/` and generates:

- **6 comparison plots**: metrics bar charts, ROC curves overlay, confusion matrix heatmaps, training time vs accuracy scatter, reliability (calibration) diagrams, ensemble complementarity analysis
- **Comprehensive metrics table**: Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss
- **Console report**: Best model per metric, computational cost ranking, key findings discussion

**Output**: `evaluation_plots/` directory with 6 PNG files + `evaluation_summary.csv`.

**Runtime**: ~10 seconds.

---

### 5. Original Ensemble Pipeline (end-to-end)

```bash
python spaceship_titanic_ensemble.py
```

Or via PowerShell:

```powershell
.\run_spaceship_titanic.ps1
```

**What it does**: The original single-file pipeline that:

1. Auto-installs missing dependencies
2. Engineers features (same preprocessing as `preprocessing.py`)
3. Trains LightGBM + CatBoost + XGBoost with 10-fold CV
4. Grid-searches blending weights at 0.01 resolution
5. Calibrates per-group decision thresholds (e.g., per DeckSide)
6. Writes `submission.csv`

**Output**: `submission.csv` (Kaggle-ready) + diagnostic probability CSVs.

**Runtime**: ~10–20 minutes.

**Cache mode** (skip training, re-blend from saved probabilities):

```bash
python spaceship_titanic_ensemble.py --from-cache
```

---

## Key Design Decisions

### Hierarchical Imputation

Rather than global mean/mode imputation, we exploit the dataset's **group structure**. Passengers sharing a `GroupId` (family/travel group) or `Surname` (family name) tend to share attributes like HomePlanet, Destination, and Cabin. Our cascading strategy imputes using progressively coarser groupings:

- `GroupId` → `Surname` → `HomePlanet` → global fallback

CryoSleep is additionally **inferred from spending patterns**: zero total spend ⇒ likely cryosleep; positive spend ⇒ likely awake.

### Feature Engineering (47 features)

- **Spend aggregates** (11): TotalSpend, FoodSpend, LuxurySpend, SpendMean/Std/Max/Min, ServicesUsed, ZeroSpend, etc.
- **Group/family features** (11): GroupSize, IsAlone, FamilySize, SpendPerGroup, GroupAgeMean, etc.
- **Interaction features** (5): DeckSide, HomeDestination, HomeDeck, DestinationDeck, CryoHome
- **Missingness indicators** (4): AgeMissing, CabinNumMissing, SpendMissingCount, SpendKnownCount
- **Base features** (16): original numeric columns + decomposed categoricals

### Model-Specific Encoding

- **LightGBM**: pandas `category` dtype — no encoding overhead, native handling
- **CatBoost**: string-type columns with `cat_features` indices — ordered target statistics
- **XGBoost, RandomForest, LogisticRegression**: `OneHotEncoder(min_frequency=2, sparse_output=True)` → sparse CSR matrices

### Validation Strategy

Stratified 5-fold × 2 random seeds = **10 independent train/validation splits**. Every training sample appears in exactly 2 validation folds (once per seed), and predictions are averaged. Test predictions are averaged across all 10 folds.

### Ensemble Blending

Grid search over `w_lgb + w_cat + w_xgb = 1` at 0.01 resolution, optimizing for accuracy via the **cumulative-sum optimal threshold** method. Per-group calibration then tunes the decision threshold for each subgroup (≥100 samples) to correct for systematic probability miscalibration.

---

## Results Summary

| Metric              |  Best Individual  |     Ensemble     |
| ------------------- | :---------------: | :---------------: |
| OOF Accuracy        | 0.8157 (LightGBM) | **0.8132** |
| OOF ROC-AUC         | 0.9084 (CatBoost) |      0.9050      |
| Kaggle Public Score |        —        | **0.81155** |
| Kaggle Rank         |        —        |   **154**   |

Gradient boosting models dominate because the dataset features **complex non-linear interactions** between group membership, cabin location, spending behavior, and transport probability. LightGBM achieves the best accuracy/time trade-off (42s vs 1000+s for CatBoost/XGBoost) due to leaf-wise tree growth, GOSS subsampling, and histogram-based split finding.

---

## References

- Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System
- Ke et al. (2017) — LightGBM: A Highly Efficient Gradient Boosting Decision Tree
- Dorogush et al. (2018) — CatBoost: gradient boosting with categorical features support
- Bergstra & Bengio (2012) — Random Search for Hyper-Parameter Optimization
- van Buuren & Groothuis-Oudshoorn (2011) — MICE: Multivariate Imputation by Chained Equations
