import importlib.util
import subprocess
import sys
import time
from pathlib import Path


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
}


def ensure_dependencies() -> None:
    missing = [pkg for module, pkg in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *missing])


ensure_dependencies()

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy import sparse
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

SEED = 2026
N_SPLITS = 5
CV_SEEDS = [2026, 42]
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def mode_value(values: pd.Series):
    mode = values.dropna().mode()
    return mode.iloc[0] if len(mode) else np.nan


def optimal_threshold_for_predictions(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    order = np.argsort(pred)
    sorted_pred = pred[order]
    sorted_y = y[order]
    correct_if_all_true = int(sorted_y.sum())
    deltas = np.where(sorted_y == 0, 1, -1)
    correct_after_cut = correct_if_all_true + np.cumsum(deltas)

    best_correct = correct_if_all_true
    threshold = float(np.nextafter(sorted_pred[0], -np.inf))
    cut_idx = int(np.argmax(correct_after_cut))
    if int(correct_after_cut[cut_idx]) > best_correct:
        best_correct = int(correct_after_cut[cut_idx])
        threshold = float(np.nextafter(sorted_pred[cut_idx], np.inf))
    return threshold, best_correct / len(y)


def fill_by_mode(df: pd.DataFrame, col: str, by_cols: list[str]) -> None:
    key = df[by_cols].astype("string").fillna("__NA__").agg("__".join, axis=1)
    valid = df[col].notna()
    if valid.sum() == 0:
        return
    mode_map = pd.DataFrame({"key": key[valid], col: df.loc[valid, col]}).groupby("key")[col].agg(mode_value)
    df[col] = df[col].fillna(key.map(mode_map))


def fill_numeric_by_median(df: pd.DataFrame, col: str, group_sets: list[list[str]]) -> None:
    for group_cols in group_sets:
        df[col] = df[col].fillna(df.groupby(group_cols, dropna=False, observed=False)[col].transform("median"))
    df[col] = df[col].fillna(df[col].median())


def split_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    passenger = out["PassengerId"].str.split("_", expand=True)
    out["GroupId"] = passenger[0].astype("string")
    out["GroupNumber"] = pd.to_numeric(passenger[0], errors="coerce")
    out["GroupMember"] = pd.to_numeric(passenger[1], errors="coerce")

    cabin = out["Cabin"].str.split("/", expand=True)
    out["CabinDeck"] = cabin[0].astype("string")
    out["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    out["CabinSide"] = cabin[2].astype("string")
    out["CabinKnown"] = out["Cabin"].notna().astype(int)

    name = out["Name"].fillna("Unknown Unknown").str.split(" ", n=1, expand=True)
    out["FirstName"] = name[0].astype("string")
    out["Surname"] = name[1].fillna("Unknown").astype("string")
    out.loc[out["Name"].isna(), "Surname"] = pd.NA

    return out


def engineer_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train_x = train.drop(columns=["Transported"]).copy()
    full = pd.concat([train_x, test.copy()], axis=0, ignore_index=True)
    n_train = len(train_x)
    full = split_core_columns(full)

    full["SpendMissingCount"] = full[SPEND_COLS].isna().sum(axis=1)
    full["SpendKnownCount"] = full[SPEND_COLS].notna().sum(axis=1)
    full["TotalSpendRaw"] = full[SPEND_COLS].sum(axis=1, min_count=1)
    full["AgeMissing"] = full["Age"].isna().astype(int)
    full["CabinNumMissing"] = full["CabinNum"].isna().astype(int)

    deck_home = {"A": "Europa", "B": "Europa", "C": "Europa", "T": "Europa", "G": "Earth"}
    full["HomePlanet"] = full["HomePlanet"].fillna(full["CabinDeck"].map(deck_home))
    for group_cols in [["GroupId"], ["Surname"], ["CabinDeck"], ["CabinSide"]]:
        fill_by_mode(full, "HomePlanet", group_cols)

    for col in ["Destination", "CabinDeck", "CabinSide"]:
        for group_cols in [["GroupId"], ["Surname"], ["HomePlanet"], ["HomePlanet", "Destination"]]:
            fill_by_mode(full, col, group_cols)

    full.loc[full["TotalSpendRaw"].fillna(0).gt(0) & full["CryoSleep"].isna(), "CryoSleep"] = False
    full.loc[full[SPEND_COLS].fillna(0).sum(axis=1).eq(0) & full["CryoSleep"].isna(), "CryoSleep"] = True
    for group_cols in [["GroupId"], ["Surname"], ["HomePlanet"], ["Destination"]]:
        fill_by_mode(full, "CryoSleep", group_cols)
    full["CryoSleep"] = full["CryoSleep"].fillna(False)

    full.loc[full["CryoSleep"].eq(True), SPEND_COLS] = full.loc[full["CryoSleep"].eq(True), SPEND_COLS].fillna(0)
    for col in SPEND_COLS:
        fill_numeric_by_median(
            full,
            col,
            [["HomePlanet", "Destination", "CryoSleep"], ["HomePlanet", "CryoSleep"], ["Destination", "CryoSleep"]],
        )
        full[col] = full[col].fillna(0)

    full["TotalSpend"] = full[SPEND_COLS].sum(axis=1)
    full.loc[full["TotalSpend"].gt(0) & full["CryoSleep"].eq(True), "CryoSleep"] = False

    for group_cols in [["GroupId"], ["Surname"], ["HomePlanet"]]:
        fill_by_mode(full, "VIP", group_cols)
    full["VIP"] = full["VIP"].fillna(False)

    fill_numeric_by_median(full, "Age", [["GroupId"], ["Surname"], ["HomePlanet", "Destination"], ["HomePlanet"]])
    fill_numeric_by_median(full, "CabinNum", [["GroupId"], ["Surname"], ["CabinDeck", "CabinSide"], ["CabinDeck"]])

    full["GroupSize"] = full["GroupId"].map(full["GroupId"].value_counts()).astype(float)
    full["IsAlone"] = full["GroupSize"].eq(1).astype(int)
    surname_counts = full["Surname"].dropna().value_counts()
    full["FamilySize"] = full["Surname"].map(surname_counts).fillna(1).astype(float)
    full["CabinGroupSize"] = full["Cabin"].map(full["Cabin"].value_counts()).fillna(1).astype(float)

    full["FoodSpend"] = full["FoodCourt"] + full["ShoppingMall"]
    full["LuxurySpend"] = full["RoomService"] + full["Spa"] + full["VRDeck"]
    full["SpendMean"] = full[SPEND_COLS].mean(axis=1)
    full["SpendStd"] = full[SPEND_COLS].std(axis=1).fillna(0)
    full["SpendMax"] = full[SPEND_COLS].max(axis=1)
    full["SpendMin"] = full[SPEND_COLS].min(axis=1)
    full["ServicesUsed"] = full[SPEND_COLS].gt(0).sum(axis=1)
    full["ZeroSpend"] = full["TotalSpend"].eq(0).astype(int)
    full["SpendPerGroup"] = full["TotalSpend"] / full["GroupSize"].clip(lower=1)
    full["SpendPerFamily"] = full["TotalSpend"] / full["FamilySize"].clip(lower=1)
    full["CabinNumPerGroup"] = full["CabinNum"] / full["GroupNumber"].clip(lower=1)
    full["GroupAgeMean"] = full.groupby("GroupId", dropna=False, observed=False)["Age"].transform("mean")
    full["GroupSpendMean"] = full.groupby("GroupId", dropna=False, observed=False)["TotalSpend"].transform("mean")
    full["GroupSpendMax"] = full.groupby("GroupId", dropna=False, observed=False)["TotalSpend"].transform("max")
    full["FamilyAgeMean"] = full.groupby("Surname", dropna=False, observed=False)["Age"].transform("mean")
    full["FamilySpendMean"] = full.groupby("Surname", dropna=False, observed=False)["TotalSpend"].transform("mean")

    category_base = ["HomePlanet", "CryoSleep", "Destination", "VIP", "CabinDeck", "CabinSide", "Surname"]
    for col in category_base:
        full[col] = full[col].astype("string").fillna("Unknown")

    full["DeckSide"] = full["CabinDeck"] + "_" + full["CabinSide"]
    full["HomeDestination"] = full["HomePlanet"] + "_" + full["Destination"]
    full["HomeDeck"] = full["HomePlanet"] + "_" + full["CabinDeck"]
    full["DestinationDeck"] = full["Destination"] + "_" + full["CabinDeck"]
    full["CryoHome"] = full["CryoSleep"] + "_" + full["HomePlanet"]

    cat_features = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "DeckSide",
        "HomeDestination",
        "HomeDeck",
        "DestinationDeck",
        "CryoHome",
        "Surname",
    ]

    num_features = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "GroupNumber",
        "GroupMember",
        "CabinNum",
        "CabinKnown",
        "SpendMissingCount",
        "SpendKnownCount",
        "TotalSpend",
        "FoodSpend",
        "LuxurySpend",
        "SpendMean",
        "SpendStd",
        "SpendMax",
        "SpendMin",
        "ServicesUsed",
        "ZeroSpend",
        "GroupSize",
        "IsAlone",
        "FamilySize",
        "CabinGroupSize",
        "SpendPerGroup",
        "SpendPerFamily",
        "CabinNumPerGroup",
        "GroupAgeMean",
        "GroupSpendMean",
        "GroupSpendMax",
        "FamilyAgeMean",
        "FamilySpendMean",
        "AgeMissing",
        "CabinNumMissing",
    ]

    for col in num_features:
        full[col] = pd.to_numeric(full[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        full[col] = full[col].fillna(full[col].median()).astype(float)

    features = num_features + cat_features
    return full.iloc[:n_train][features].reset_index(drop=True), full.iloc[n_train:][features].reset_index(drop=True), num_features, cat_features


def as_lgb_frame(df: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_features:
        out[col] = out[col].astype("category")
    return out


def as_cat_frame(df: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_features:
        out[col] = out[col].astype(str)
    return out


def fit_predict_lgbm(x_train, y_train, x_valid, y_valid, x_test, cat_features, seed):
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=6000,
        learning_rate=0.018,
        num_leaves=31,
        max_depth=6,
        min_child_samples=45,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.86,
        reg_alpha=0.35,
        reg_lambda=6.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="binary_logloss",
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(0)],
    )
    return model.predict_proba(x_valid)[:, 1], model.predict_proba(x_test)[:, 1]


def fit_predict_catboost(x_train, y_train, x_valid, y_valid, x_test, cat_features, seed):
    cat_idx = [x_train.columns.get_loc(col) for col in cat_features]
    model = CatBoostClassifier(
        iterations=3200,
        learning_rate=0.032,
        depth=5,
        l2_leaf_reg=7.0,
        random_strength=0.7,
        bagging_temperature=0.3,
        loss_function="Logloss",
        eval_metric="Logloss",
        bootstrap_type="Bayesian",
        allow_writing_files=False,
        random_seed=seed,
        thread_count=-1,
        verbose=False,
        od_type="Iter",
        od_wait=150,
    )
    model.fit(x_train, y_train, cat_features=cat_idx, eval_set=(x_valid, y_valid), use_best_model=True)
    return model.predict_proba(x_valid)[:, 1], model.predict_proba(x_test)[:, 1]


def fit_predict_xgboost(x_train, y_train, x_valid, y_valid, x_test, num_features, cat_features, seed):
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
    xtr_cat = encoder.fit_transform(x_train[cat_features].astype(str))
    xva_cat = encoder.transform(x_valid[cat_features].astype(str))
    xte_cat = encoder.transform(x_test[cat_features].astype(str))

    xtr_num = sparse.csr_matrix(x_train[num_features].astype(float).to_numpy())
    xva_num = sparse.csr_matrix(x_valid[num_features].astype(float).to_numpy())
    xte_num = sparse.csr_matrix(x_test[num_features].astype(float).to_numpy())

    xtr = sparse.hstack([xtr_num, xtr_cat], format="csr")
    xva = sparse.hstack([xva_num, xva_cat], format="csr")
    xte = sparse.hstack([xte_num, xte_cat], format="csr")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=5000,
        learning_rate=0.016,
        max_depth=4,
        min_child_weight=6,
        subsample=0.86,
        colsample_bytree=0.82,
        reg_alpha=0.25,
        reg_lambda=7.0,
        gamma=0.03,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=250,
    )
    model.fit(xtr, y_train, eval_set=[(xva, y_valid)], verbose=False)
    return model.predict_proba(xva)[:, 1], model.predict_proba(xte)[:, 1]


def tune_blend(oof: dict[str, np.ndarray], y: np.ndarray) -> tuple[dict[str, float], float, float]:
    best_score = -1.0
    best_weights = {"lgbm": 0.4, "cat": 0.4, "xgb": 0.2}
    best_threshold_value = 0.5
    grid = np.round(np.arange(0.01, 0.99, 0.01), 2)

    for w_lgb in grid:
        for w_cat in grid:
            w_xgb = round(1.0 - w_lgb - w_cat, 2)
            if w_xgb < 0.01:
                continue
            pred = w_lgb * oof["lgbm"] + w_cat * oof["cat"] + w_xgb * oof["xgb"]
            threshold, score = optimal_threshold_for_predictions(pred, y)
            if score > best_score:
                best_score = score
                best_weights = {"lgbm": float(w_lgb), "cat": float(w_cat), "xgb": float(w_xgb)}
                best_threshold_value = float(threshold)

    return best_weights, best_threshold_value, best_score


def tune_group_thresholds(
    pred: np.ndarray,
    y: np.ndarray,
    groups: pd.Series,
    default_threshold: float,
    min_count: int = 100,
) -> tuple[dict[str, float], float]:
    group_key = groups.astype("string").fillna("Unknown")
    thresholds: dict[str, float] = {}
    calibrated = np.zeros(len(y), dtype=bool)

    for value, idxs in group_key.groupby(group_key, dropna=False).groups.items():
        idx = np.fromiter(idxs, dtype=int)
        if len(idx) >= min_count:
            group_threshold, _ = optimal_threshold_for_predictions(pred[idx], y[idx])
        else:
            group_threshold = default_threshold
        thresholds[str(value)] = float(group_threshold)
        calibrated[idx] = pred[idx] >= group_threshold

    return thresholds, accuracy_score(y, calibrated)


def apply_group_thresholds(pred: np.ndarray, groups: pd.Series, thresholds: dict[str, float], default_threshold: float):
    group_key = groups.astype("string").fillna("Unknown")
    used_thresholds = group_key.map(thresholds).astype(float).fillna(default_threshold).to_numpy()
    return pred >= used_thresholds


def main() -> None:
    base = Path(__file__).resolve().parent
    train = pd.read_csv(base / "train.csv")
    test = pd.read_csv(base / "test.csv")
    sample = pd.read_csv(base / "sample_submission.csv")

    y = train["Transported"].astype(int).to_numpy()
    x, x_test, num_features, cat_features = engineer_features(train, test)

    if "--from-cache" in sys.argv:
        oof_cache = pd.read_csv(base / "oof_model_probabilities.csv")
        test_cache = pd.read_csv(base / "submission_probabilities.csv")
        oof = {name: oof_cache[f"p_{name}"].to_numpy() for name in ["lgbm", "cat", "xgb"]}
        test_pred = {name: test_cache[f"p_{name}"].to_numpy() for name in ["lgbm", "cat", "xgb"]}
        weights, threshold, score = tune_blend(oof, y)
        blended_oof = sum(weights[name] * oof[name] for name in weights)
        blended_test = sum(weights[name] * test_pred[name] for name in weights)
        write_submission(base, sample, test, x, x_test, y, oof, test_pred, blended_oof, blended_test, weights, threshold, score)
        return

    x_lgb = as_lgb_frame(x, cat_features)
    xt_lgb = as_lgb_frame(x_test, cat_features)
    x_cat = as_cat_frame(x, cat_features)
    xt_cat = as_cat_frame(x_test, cat_features)

    oof_sum = {name: np.zeros(len(x), dtype=float) for name in ["lgbm", "cat", "xgb"]}
    oof_count = {name: np.zeros(len(x), dtype=float) for name in ["lgbm", "cat", "xgb"]}
    test_pred = {name: np.zeros(len(x_test), dtype=float) for name in ["lgbm", "cat", "xgb"]}

    total_folds = N_SPLITS * len(CV_SEEDS)
    run_fold = 0
    for seed in CV_SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y), start=1):
            run_fold += 1
            fold_start = time.time()
            print(f"Seed {seed} fold {fold}/{N_SPLITS} ({run_fold}/{total_folds})", flush=True)

            lgb_valid, lgb_test = fit_predict_lgbm(
                x_lgb.iloc[train_idx],
                y[train_idx],
                x_lgb.iloc[valid_idx],
                y[valid_idx],
                xt_lgb,
                cat_features,
                seed,
            )
            oof_sum["lgbm"][valid_idx] += lgb_valid
            oof_count["lgbm"][valid_idx] += 1
            test_pred["lgbm"] += lgb_test / total_folds

            cat_valid, cat_test = fit_predict_catboost(
                x_cat.iloc[train_idx],
                y[train_idx],
                x_cat.iloc[valid_idx],
                y[valid_idx],
                xt_cat,
                cat_features,
                seed,
            )
            oof_sum["cat"][valid_idx] += cat_valid
            oof_count["cat"][valid_idx] += 1
            test_pred["cat"] += cat_test / total_folds

            xgb_valid, xgb_test = fit_predict_xgboost(
                x.iloc[train_idx],
                y[train_idx],
                x.iloc[valid_idx],
                y[valid_idx],
                x_test,
                num_features,
                cat_features,
                seed,
            )
            oof_sum["xgb"][valid_idx] += xgb_valid
            oof_count["xgb"][valid_idx] += 1
            test_pred["xgb"] += xgb_test / total_folds

            fold_blend = (lgb_valid + cat_valid + xgb_valid) / 3
            print(
                f"  equal-blend accuracy: {accuracy_score(y[valid_idx], fold_blend >= 0.5):.5f}; "
                f"fold seconds: {time.time() - fold_start:.1f}",
                flush=True,
            )

    oof = {name: oof_sum[name] / np.maximum(oof_count[name], 1) for name in oof_sum}

    for name, pred in oof.items():
        print(f"{name} OOF accuracy@0.5: {accuracy_score(y, pred >= 0.5):.5f}; logloss: {log_loss(y, pred):.5f}")

    pd.DataFrame(
        {
            "Transported": y.astype(bool),
            "p_lgbm": oof["lgbm"],
            "p_cat": oof["cat"],
            "p_xgb": oof["xgb"],
        }
    ).to_csv(base / "oof_model_probabilities.csv", index=False)

    weights, threshold, score = tune_blend(oof, y)
    blended_oof = sum(weights[name] * oof[name] for name in weights)
    blended_test = sum(weights[name] * test_pred[name] for name in weights)

    write_submission(base, sample, test, x, x_test, y, oof, test_pred, blended_oof, blended_test, weights, threshold, score)


def write_submission(
    base: Path,
    sample: pd.DataFrame,
    test: pd.DataFrame,
    x: pd.DataFrame,
    x_test: pd.DataFrame,
    y: np.ndarray,
    oof: dict[str, np.ndarray],
    test_pred: dict[str, np.ndarray],
    blended_oof: np.ndarray,
    blended_test: np.ndarray,
    weights: dict[str, float],
    threshold: float,
    score: float,
) -> None:
    print(f"Best OOF blend accuracy: {score:.5f}")
    print(f"Blend weights: {weights}")
    print(f"Decision threshold: {threshold:.3f}")
    print(f"OOF logloss: {log_loss(y, blended_oof):.5f}")

    calibration_candidates = ["DeckSide", "CabinDeck", "Destination", "HomeDestination", "CryoHome", "ServicesUsed"]
    best_calibration_col = None
    best_calibration_thresholds: dict[str, float] = {}
    best_calibration_score = score
    for col in calibration_candidates:
        thresholds, cal_score = tune_group_thresholds(blended_oof, y, x[col], threshold, min_count=100)
        print(f"Calibration {col}: {cal_score:.5f}")
        if cal_score > best_calibration_score:
            best_calibration_score = cal_score
            best_calibration_col = col
            best_calibration_thresholds = thresholds

    if best_calibration_col is None:
        final_pred = blended_test >= threshold
        print("Selected calibration: global")
    else:
        final_pred = apply_group_thresholds(
            blended_test,
            x_test[best_calibration_col],
            best_calibration_thresholds,
            threshold,
        )
        print(f"Selected calibration: {best_calibration_col} ({best_calibration_score:.5f})")

    submission = sample.copy()
    submission["Transported"] = final_pred.astype(bool)
    submission.to_csv(base / "submission.csv", index=False)

    diagnostic = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"],
            "p_lgbm": test_pred["lgbm"],
            "p_cat": test_pred["cat"],
            "p_xgb": test_pred["xgb"],
            "p_blend": blended_test,
            "Transported": submission["Transported"],
        }
    )
    diagnostic.to_csv(base / "submission_probabilities.csv", index=False)
    pd.DataFrame(
        {
            "Transported": y.astype(bool),
            "p_lgbm": oof["lgbm"],
            "p_cat": oof["cat"],
            "p_xgb": oof["xgb"],
            "p_blend": blended_oof,
        }
    ).to_csv(base / "oof_probabilities.csv", index=False)
    print(f"Saved: {base / 'submission.csv'}")
    print(submission["Transported"].value_counts(normalize=True).to_dict())


if __name__ == "__main__":
    main()
