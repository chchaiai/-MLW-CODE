"""
Model Selection, Training, and Validation
==========================================
Implements 5 ML models with systematic hyperparameter tuning (RandomizedSearchCV),
stratified k-fold cross-validation, and training-time tracking.

Models:
  1. Logistic Regression  (baseline linear model)
  2. Random Forest        (bagging ensemble)
  3. LightGBM             (gradient boosting)
  4. CatBoost             (gradient boosting with ordered target encoding)
  5. XGBoost              (gradient boosting)

Outputs:
  - trained_model_*.joblib         saved model objects
  - oof_model_probabilities.csv    out-of-fold probability predictions
  - hyperparameter_tuning_results.csv  best params & CV scores
  - model_training_times.csv       training time per model
"""

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import uniform, randint, loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "project  information" / "spaceship-titanic dataset"
OUT_DIR = BASE_DIR / "model_outputs"
OUT_DIR.mkdir(exist_ok=True)

SEED = 2026
N_SPLITS = 5
N_HP_ITER = 30  # RandomizedSearch iterations per model
CV_SEEDS = [2026, 42]

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


# ── Preprocessing (self-contained for standalone use) ─────────────────────


def load_engineered_data() -> dict:
    """Load data and apply the full feature-engineering pipeline. Returns a dict of arrays/indices."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y = train["Transported"].astype(int).to_numpy()
    train_x = train.drop(columns=["Transported"])
    n_train = len(train_x)

    # Use inline preprocessing (self-contained for standalone use)

    full = pd.concat([train_x, test.copy()], axis=0, ignore_index=True)
    full = _split_columns(full)
    full = _impute_all(full)
    X_full = full.iloc[:n_train].reset_index(drop=True)
    X_test = full.iloc[n_train:].reset_index(drop=True)

    num_features = [
        "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "GroupNumber", "GroupMember", "CabinNum", "CabinKnown",
        "SpendMissingCount", "SpendKnownCount", "TotalSpend",
        "FoodSpend", "LuxurySpend", "SpendMean", "SpendStd",
        "SpendMax", "SpendMin", "ServicesUsed", "ZeroSpend",
        "GroupSize", "IsAlone", "FamilySize", "CabinGroupSize",
        "SpendPerGroup", "SpendPerFamily", "CabinNumPerGroup",
        "GroupAgeMean", "GroupSpendMean", "GroupSpendMax",
        "FamilyAgeMean", "FamilySpendMean", "AgeMissing", "CabinNumMissing",
    ]
    cat_features = [
        "HomePlanet", "CryoSleep", "Destination", "VIP",
        "CabinDeck", "CabinSide", "DeckSide", "HomeDestination",
        "HomeDeck", "DestinationDeck", "CryoHome", "Surname",
    ]

    features = num_features + cat_features
    X_full = X_full[features].reset_index(drop=True)
    X_test = X_test[features].reset_index(drop=True)

    return {
        "X": X_full, "X_test": X_test, "y": y,
        "num_features": num_features, "cat_features": cat_features,
        "test_ids": test["PassengerId"].values,
    }


def _split_columns(full: pd.DataFrame) -> pd.DataFrame:
    df = full.copy()
    p = df["PassengerId"].str.split("_", expand=True)
    df["GroupId"] = p[0].astype("string")
    df["GroupNumber"] = pd.to_numeric(p[0], errors="coerce")
    df["GroupMember"] = pd.to_numeric(p[1], errors="coerce")
    c = df["Cabin"].str.split("/", expand=True)
    df["CabinDeck"] = c[0].astype("string")
    df["CabinNum"] = pd.to_numeric(c[1], errors="coerce")
    df["CabinSide"] = c[2].astype("string")
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    n = df["Name"].fillna("Unknown Unknown").str.split(" ", n=1, expand=True)
    df["FirstName"] = n[0].astype("string")
    df["Surname"] = n[1].fillna("Unknown").astype("string")
    df.loc[df["Name"].isna(), "Surname"] = pd.NA
    return df


def _impute_all(full: pd.DataFrame) -> pd.DataFrame:
    from preprocessing import (
        impute_home_planet, impute_categoricals, impute_cryosleep,
        impute_spend, impute_vip, impute_numeric,
        engineer_spend_features, engineer_group_features,
        engineer_interaction_features, finalize_categoricals,
    )
    full["SpendMissingCount"] = full[SPEND_COLS].isna().sum(axis=1)
    full["SpendKnownCount"] = full[SPEND_COLS].notna().sum(axis=1)
    full["TotalSpendRaw"] = full[SPEND_COLS].sum(axis=1, min_count=1)
    full["AgeMissing"] = full["Age"].isna().astype(int)
    full["CabinNumMissing"] = full["CabinNum"].isna().astype(int)

    cat_base = ["HomePlanet", "CryoSleep", "Destination", "VIP", "CabinDeck", "CabinSide", "Surname"]
    impute_home_planet(full)
    impute_categoricals(full)
    impute_cryosleep(full)
    impute_spend(full)
    full["TotalSpend"] = full[SPEND_COLS].sum(axis=1)
    full.loc[full["TotalSpend"].gt(0) & full["CryoSleep"].eq(True), "CryoSleep"] = False
    impute_vip(full)
    impute_numeric(full)
    full = finalize_categoricals(full, cat_base)
    full = engineer_spend_features(full)
    full = engineer_group_features(full)
    full = engineer_interaction_features(full)
    full = finalize_categoricals(full, cat_base)
    return full


# ── Model factory ─────────────────────────────────────────────────────────


def _make_lgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=6000, learning_rate=0.018,
        num_leaves=31, max_depth=6, min_child_samples=45,
        subsample=0.82, subsample_freq=1, colsample_bytree=0.86,
        reg_alpha=0.35, reg_lambda=6.0, random_state=SEED,
        n_jobs=-1, verbose=-1,
    )


def _make_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=3200, learning_rate=0.032, depth=5, l2_leaf_reg=7.0,
        random_strength=0.7, bagging_temperature=0.3,
        loss_function="Logloss", eval_metric="Logloss",
        bootstrap_type="Bayesian", allow_writing_files=False,
        random_seed=SEED, thread_count=-1, verbose=False,
        od_type="Iter", od_wait=150,
    )


def _make_xgboost() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        n_estimators=5000, learning_rate=0.016, max_depth=4,
        min_child_weight=6, subsample=0.86, colsample_bytree=0.82,
        reg_alpha=0.25, reg_lambda=7.0, gamma=0.03,
        tree_method="hist", random_state=SEED, n_jobs=-1,
        early_stopping_rounds=250,
    )


def _make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400, max_depth=18, min_samples_split=10,
        min_samples_leaf=4, max_features="sqrt", bootstrap=True,
        random_state=SEED, n_jobs=-1, verbose=0,
    )


def _make_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        penalty="l2", C=1.0, solver="saga", max_iter=2000,
        random_state=SEED, n_jobs=-1,
    )


# ── Hyperparameter search spaces ──────────────────────────────────────────


HP_SPACES = {
    "LightGBM": {
        "num_leaves": randint(15, 127),
        "max_depth": randint(3, 12),
        "learning_rate": loguniform(0.005, 0.05),
        "min_child_samples": randint(10, 80),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": loguniform(0.01, 2.0),
        "reg_lambda": loguniform(0.1, 20.0),
    },
    "CatBoost": {
        "depth": randint(3, 9),
        "learning_rate": loguniform(0.01, 0.08),
        "l2_leaf_reg": uniform(1, 20),
        "random_strength": uniform(0.2, 2.0),
        "bagging_temperature": uniform(0.1, 1.5),
    },
    "XGBoost": {
        "max_depth": randint(3, 10),
        "learning_rate": loguniform(0.005, 0.05),
        "min_child_weight": randint(1, 15),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": loguniform(0.01, 2.0),
        "reg_lambda": loguniform(0.1, 20.0),
        "gamma": uniform(0, 0.5),
    },
    "RandomForest": {
        "n_estimators": randint(100, 800),
        "max_depth": randint(5, 40),
        "min_samples_split": randint(2, 30),
        "min_samples_leaf": randint(1, 15),
        "max_features": ["sqrt", "log2", 0.5, 0.8],
    },
    "LogisticRegression": {
        "C": loguniform(0.001, 10),
        "solver": ["saga", "lbfgs"],
        "max_iter": [2000, 3000, 5000],
    },
}

MODEL_FACTORIES = {
    "LightGBM": _make_lgbm,
    "CatBoost": _make_catboost,
    "XGBoost": _make_xgboost,
    "RandomForest": _make_random_forest,
    "LogisticRegression": _make_logistic_regression,
}


# ── Data preparation per model type ───────────────────────────────────────


def prepare_tree_data(X_train, X_valid, X_test, num_features, cat_features):
    """For LightGBM: return category-typed frames."""
    def _to_lgb(df):
        out = df.copy()
        for c in cat_features:
            out[c] = out[c].astype("category")
        return out
    return _to_lgb(X_train), _to_lgb(X_valid), _to_lgb(X_test)


def prepare_catboost_data(X_train, X_valid, X_test, cat_features):
    """For CatBoost: return string-typed frames with cat indices."""
    def _to_cat(df):
        out = df.copy()
        for c in cat_features:
            out[c] = out[c].astype(str)
        return out
    cat_idx = [X_train.columns.get_loc(c) for c in cat_features]
    return _to_cat(X_train), _to_cat(X_valid), _to_cat(X_test), cat_idx


def prepare_onehot_data(X_train, X_valid, X_test, num_features, cat_features):
    """For XGBoost/RF/LogReg: one-hot encode categoricals, optionally scale numerics."""
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
    Xtr_cat = encoder.fit_transform(X_train[cat_features].astype(str))
    Xva_cat = encoder.transform(X_valid[cat_features].astype(str))
    Xte_cat = encoder.transform(X_test[cat_features].astype(str))

    Xtr_num = sparse.csr_matrix(X_train[num_features].astype(float).to_numpy())
    Xva_num = sparse.csr_matrix(X_valid[num_features].astype(float).to_numpy())
    Xte_num = sparse.csr_matrix(X_test[num_features].astype(float).to_numpy())

    return (
        sparse.hstack([Xtr_num, Xtr_cat], format="csr"),
        sparse.hstack([Xva_num, Xva_cat], format="csr"),
        sparse.hstack([Xte_num, Xte_cat], format="csr"),
    )


# ── Cross-validation with hyperparameter tuning ───────────────────────────


def run_model_cv(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    num_features: list[str],
    cat_features: list[str],
    n_hp_iter: int = N_HP_ITER,
) -> dict:
    """Run stratified k-fold CV with RandomizedSearchCV for one model.

    Returns a dict with OOF predictions, test predictions, best params, scores, and timing.
    """
    print(f"\n{'─' * 60}")
    print(f"Model: {name}")
    print(f"{'─' * 60}")

    total_folds = N_SPLITS * len(CV_SEEDS)
    oof_preds = np.zeros(len(X), dtype=float)
    test_preds = np.zeros(len(X_test), dtype=float)
    fold_scores = []
    total_time = 0.0
    best_params_list = []

    is_tree_native = name in ("LightGBM", "CatBoost")
    is_xgb = name == "XGBoost"

    for seed_idx, seed in enumerate(CV_SEEDS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            fold_start = time.time()
            print(f"  Seed {seed} Fold {fold}/{N_SPLITS} ...", end=" ", flush=True)

            X_tr_raw, X_va_raw = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # Prepare data
            if name == "LightGBM":
                X_tr, X_va, X_te = prepare_tree_data(X_tr_raw, X_va_raw, X_test, num_features, cat_features)
                base_model = _make_lgbm()
                fit_kwargs = dict(
                    categorical_feature=cat_features,
                    eval_set=[(X_va, y_va)],
                    eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
                )
            elif name == "CatBoost":
                X_tr, X_va, X_te, cat_idx = prepare_catboost_data(X_tr_raw, X_va_raw, X_test, cat_features)
                base_model = _make_catboost()
                fit_kwargs = dict(
                    cat_features=cat_idx,
                    eval_set=(X_va, y_va),
                    use_best_model=True,
                )
            elif name == "XGBoost":
                X_tr, X_va, X_te = prepare_onehot_data(X_tr_raw, X_va_raw, X_test, num_features, cat_features)
                base_model = _make_xgboost()
                fit_kwargs = dict(eval_set=[(X_va, y_va)], verbose=False)
            else:
                X_tr, X_va, X_te = prepare_onehot_data(X_tr_raw, X_va_raw, X_test, num_features, cat_features)
                base_model = MODEL_FACTORIES[name]()
                fit_kwargs = {}

            # Hyperparameter search (first fold only, to save time)
            if fold == 1 and seed == CV_SEEDS[0] and name in HP_SPACES:
                print("tuning...", end=" ", flush=True)
                search = RandomizedSearchCV(
                    base_model, HP_SPACES[name], n_iter=n_hp_iter,
                    scoring="roc_auc", cv=3, random_state=SEED, n_jobs=-1,
                    verbose=0, error_score="raise",
                )
                search.fit(X_tr, y_tr, **fit_kwargs)
                model = search.best_estimator_
                best_params_list.append(search.best_params_)
                print(f"best={search.best_score_:.4f}", end=" ", flush=True)
            else:
                model = base_model
                model.fit(X_tr, y_tr, **fit_kwargs)

            # Predict
            va_prob = model.predict_proba(X_va)[:, 1]
            te_prob = model.predict_proba(X_te)[:, 1]
            oof_preds[va_idx] += va_prob
            test_preds += te_prob / total_folds

            fold_acc = accuracy_score(y_va, va_prob >= 0.5)
            fold_scores.append(fold_acc)
            elapsed = time.time() - fold_start
            total_time += elapsed
            print(f"acc={fold_acc:.4f}  ({elapsed:.1f}s)")

    # Average OOF (each sample appears once per seed)
    oof_preds /= len(CV_SEEDS)

    oof_acc = accuracy_score(y, oof_preds >= 0.5)
    oof_auc = roc_auc_score(y, oof_preds)
    oof_ll = log_loss(y, oof_preds)

    print(f"  OOF accuracy: {oof_acc:.5f}  |  ROC-AUC: {oof_auc:.5f}  |  LogLoss: {oof_ll:.5f}")
    print(f"  Total training time: {total_time:.1f}s  ({total_time / total_folds:.1f}s/fold)")

    return {
        "name": name,
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "fold_scores": fold_scores,
        "oof_accuracy": oof_acc,
        "oof_roc_auc": oof_auc,
        "oof_log_loss": oof_ll,
        "total_time_s": total_time,
        "best_params": best_params_list[0] if best_params_list else {},
        "avg_fold_score": np.mean(fold_scores),
        "std_fold_score": np.std(fold_scores),
    }


# ── Main entry point ──────────────────────────────────────────────────────


def main() -> None:
    print("Loading and preprocessing data...")
    data = load_engineered_data()
    X, X_test, y = data["X"], data["X_test"], data["y"]
    num_features = data["num_features"]
    cat_features = data["cat_features"]

    model_names = ["LogisticRegression", "RandomForest", "LightGBM", "CatBoost", "XGBoost"]
    results = {}

    for name in model_names:
        res = run_model_cv(name, X, y, X_test, num_features, cat_features)
        results[name] = res

        # Save model OOF predictions for blending
        joblib.dump(res, OUT_DIR / f"result_{name}.joblib")

    # ── Save summary CSVs ─────────────────────────────────────────────────
    oof_df = pd.DataFrame({"Transported": y.astype(bool)})
    test_df = pd.DataFrame({"PassengerId": data["test_ids"]})
    for name in model_names:
        oof_df[f"p_{name}"] = results[name]["oof_preds"]
        test_df[f"p_{name}"] = results[name]["test_preds"]
    oof_df.to_csv(OUT_DIR / "oof_model_probabilities.csv", index=False)
    test_df.to_csv(OUT_DIR / "submission_probabilities.csv", index=False)

    # Hyperparameter tuning summary
    hp_rows = []
    for name in model_names:
        r = results[name]
        hp_rows.append({
            "Model": name,
            "Best_Params": str(r["best_params"]),
            "OOF_Accuracy": f"{r['oof_accuracy']:.5f}",
            "OOF_ROC_AUC": f"{r['oof_roc_auc']:.5f}",
            "OOF_LogLoss": f"{r['oof_log_loss']:.5f}",
            "CV_Mean_Acc": f"{r['avg_fold_score']:.5f}",
            "CV_Std_Acc": f"{r['std_fold_score']:.5f}",
            "Training_Time_s": f"{r['total_time_s']:.1f}",
        })
    pd.DataFrame(hp_rows).to_csv(OUT_DIR / "hyperparameter_tuning_results.csv", index=False)

    # Timing comparison
    time_rows = []
    for name in model_names:
        r = results[name]
        time_rows.append({
            "Model": name,
            "Total_Time_s": r["total_time_s"],
            "Time_per_Fold_s": r["total_time_s"] / (N_SPLITS * len(CV_SEEDS)),
            "OOF_Accuracy": r["oof_accuracy"],
        })
    pd.DataFrame(time_rows).to_csv(OUT_DIR / "model_training_times.csv", index=False)

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUT_DIR}/")
    print("\nSummary:")
    for row in hp_rows:
        print(f"  {row['Model']:<22s}  OOF Acc={row['OOF_Accuracy']}  AUC={row['OOF_ROC_AUC']}  Time={row['Training_Time_s']}s")


if __name__ == "__main__":
    main()
