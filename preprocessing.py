"""
Data Preparation and Preprocessing Module
==========================================
Handles missing data imputation, categorical encoding, feature engineering,
normalization/standardization, and train/test splitting for the Spaceship Titanic dataset.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# ── Helper utilities ──────────────────────────────────────────────────────


def mode_value(series: pd.Series):
    """Return the first mode of a series, or NaN if empty."""
    modes = series.dropna().mode()
    return modes.iloc[0] if len(modes) else np.nan


def fill_by_mode(df: pd.DataFrame, col: str, group_cols: list[str]) -> None:
    """Fill missing values in `col` using the mode within each group defined by `group_cols`."""
    key = df[group_cols].astype("string").fillna("__NA__").agg("__".join, axis=1)
    valid = df[col].notna()
    if valid.sum() == 0:
        return
    mode_map = pd.DataFrame({"key": key[valid], col: df.loc[valid, col]}).groupby("key")[col].agg(mode_value)
    df[col] = df[col].fillna(key.map(mode_map))


def fill_numeric_by_median(df: pd.DataFrame, col: str, group_sets: list[list[str]]) -> None:
    """Fill missing numeric values using group medians in cascading order, falling back to global median."""
    for group_cols in group_sets:
        df[col] = df[col].fillna(df.groupby(group_cols, dropna=False, observed=False)[col].transform("median"))
    df[col] = df[col].fillna(df[col].median())


# ── Core column splitting ─────────────────────────────────────────────────


def split_passenger_id(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose PassengerId into GroupId, GroupNumber, and GroupMember."""
    parts = df["PassengerId"].str.split("_", expand=True)
    df = df.copy()
    df["GroupId"] = parts[0].astype("string")
    df["GroupNumber"] = pd.to_numeric(parts[0], errors="coerce")
    df["GroupMember"] = pd.to_numeric(parts[1], errors="coerce")
    return df


def split_cabin(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose Cabin into Deck, Num, and Side."""
    parts = df["Cabin"].str.split("/", expand=True)
    df = df.copy()
    df["CabinDeck"] = parts[0].astype("string")
    df["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
    df["CabinSide"] = parts[2].astype("string")
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    return df


def split_name(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose Name into FirstName and Surname."""
    parts = df["Name"].fillna("Unknown Unknown").str.split(" ", n=1, expand=True)
    df = df.copy()
    df["FirstName"] = parts[0].astype("string")
    df["Surname"] = parts[1].fillna("Unknown").astype("string")
    df.loc[df["Name"].isna(), "Surname"] = pd.NA
    return df


# ── Missing-value analysis ────────────────────────────────────────────────


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with column-level missing-value statistics."""
    total = len(df)
    report = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing": df.isna().sum().values,
            "Percent": (df.isna().sum() / total * 100).values,
            "Dtype": df.dtypes.values,
        }
    )
    return report.sort_values("Missing", ascending=False).reset_index(drop=True)


# ── Imputation pipeline ───────────────────────────────────────────────────


def impute_home_planet(full: pd.DataFrame) -> None:
    """Impute HomePlanet via deck → home mapping then cascading group mode."""
    deck_home = {"A": "Europa", "B": "Europa", "C": "Europa", "T": "Europa", "G": "Earth"}
    full["HomePlanet"] = full["HomePlanet"].fillna(full["CabinDeck"].map(deck_home))
    for group in [["GroupId"], ["Surname"], ["CabinDeck"], ["CabinSide"]]:
        fill_by_mode(full, "HomePlanet", group)


def impute_categoricals(full: pd.DataFrame) -> None:
    """Impute Destination, CabinDeck, and CabinSide via cascading group mode."""
    for col in ["Destination", "CabinDeck", "CabinSide"]:
        for group in [["GroupId"], ["Surname"], ["HomePlanet"], ["HomePlanet", "Destination"]]:
            fill_by_mode(full, col, group)


def impute_cryosleep(full: pd.DataFrame) -> None:
    """Infer CryoSleep from spend patterns, then backfill via group mode."""
    full.loc[full["TotalSpendRaw"].fillna(0).gt(0) & full["CryoSleep"].isna(), "CryoSleep"] = False
    full.loc[full[SPEND_COLS].fillna(0).sum(axis=1).eq(0) & full["CryoSleep"].isna(), "CryoSleep"] = True
    for group in [["GroupId"], ["Surname"], ["HomePlanet"], ["Destination"]]:
        fill_by_mode(full, "CryoSleep", group)
    full["CryoSleep"] = full["CryoSleep"].fillna(False)


def impute_spend(full: pd.DataFrame) -> None:
    """Zero-fill spend for CryoSleep passengers, then hierarchical median imputation for the rest."""
    full.loc[full["CryoSleep"].eq(True), SPEND_COLS] = full.loc[full["CryoSleep"].eq(True), SPEND_COLS].fillna(0)
    for col in SPEND_COLS:
        fill_numeric_by_median(
            full,
            col,
            [["HomePlanet", "Destination", "CryoSleep"], ["HomePlanet", "CryoSleep"], ["Destination", "CryoSleep"]],
        )
        full[col] = full[col].fillna(0)


def impute_vip(full: pd.DataFrame) -> None:
    """Impute VIP via group mode; fallback to False."""
    for group in [["GroupId"], ["Surname"], ["HomePlanet"]]:
        fill_by_mode(full, "VIP", group)
    full["VIP"] = full["VIP"].fillna(False)


def impute_numeric(full: pd.DataFrame) -> None:
    """Impute Age and CabinNum via cascading group medians."""
    fill_numeric_by_median(full, "Age", [["GroupId"], ["Surname"], ["HomePlanet", "Destination"], ["HomePlanet"]])
    fill_numeric_by_median(full, "CabinNum", [["GroupId"], ["Surname"], ["CabinDeck", "CabinSide"], ["CabinDeck"]])


# ── Feature engineering ───────────────────────────────────────────────────


def engineer_spend_features(full: pd.DataFrame) -> pd.DataFrame:
    """Create aggregated spend features."""
    df = full.copy()
    df["SpendMissingCount"] = df[SPEND_COLS].isna().sum(axis=1)
    df["SpendKnownCount"] = df[SPEND_COLS].notna().sum(axis=1)
    df["TotalSpendRaw"] = df[SPEND_COLS].sum(axis=1, min_count=1)
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    df["FoodSpend"] = df["FoodCourt"] + df["ShoppingMall"]
    df["LuxurySpend"] = df["RoomService"] + df["Spa"] + df["VRDeck"]
    df["SpendMean"] = df[SPEND_COLS].mean(axis=1)
    df["SpendStd"] = df[SPEND_COLS].std(axis=1).fillna(0)
    df["SpendMax"] = df[SPEND_COLS].max(axis=1)
    df["SpendMin"] = df[SPEND_COLS].min(axis=1)
    df["ServicesUsed"] = df[SPEND_COLS].gt(0).sum(axis=1)
    df["ZeroSpend"] = df["TotalSpend"].eq(0).astype(int)
    return df


def engineer_group_features(full: pd.DataFrame) -> pd.DataFrame:
    """Create group-based and family-based features."""
    df = full.copy()
    df["GroupSize"] = df["GroupId"].map(df["GroupId"].value_counts()).astype(float)
    df["IsAlone"] = df["GroupSize"].eq(1).astype(int)
    surname_counts = df["Surname"].dropna().value_counts()
    df["FamilySize"] = df["Surname"].map(surname_counts).fillna(1).astype(float)
    df["CabinGroupSize"] = df["Cabin"].map(df["Cabin"].value_counts()).fillna(1).astype(float)
    df["SpendPerGroup"] = df["TotalSpend"] / df["GroupSize"].clip(lower=1)
    df["SpendPerFamily"] = df["TotalSpend"] / df["FamilySize"].clip(lower=1)
    df["CabinNumPerGroup"] = df["CabinNum"] / df["GroupNumber"].clip(lower=1)
    df["GroupAgeMean"] = df.groupby("GroupId", dropna=False, observed=False)["Age"].transform("mean")
    df["GroupSpendMean"] = df.groupby("GroupId", dropna=False, observed=False)["TotalSpend"].transform("mean")
    df["GroupSpendMax"] = df.groupby("GroupId", dropna=False, observed=False)["TotalSpend"].transform("max")
    df["FamilyAgeMean"] = df.groupby("Surname", dropna=False, observed=False)["Age"].transform("mean")
    df["FamilySpendMean"] = df.groupby("Surname", dropna=False, observed=False)["TotalSpend"].transform("mean")
    return df


def engineer_interaction_features(full: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between categorical columns."""
    df = full.copy()
    df["DeckSide"] = df["CabinDeck"] + "_" + df["CabinSide"]
    df["HomeDestination"] = df["HomePlanet"] + "_" + df["Destination"]
    df["HomeDeck"] = df["HomePlanet"] + "_" + df["CabinDeck"]
    df["DestinationDeck"] = df["Destination"] + "_" + df["CabinDeck"]
    df["CryoHome"] = df["CryoSleep"] + "_" + df["HomePlanet"]
    return df


def finalize_categoricals(full: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Cast categorical columns to string and fill remaining nulls."""
    df = full.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown")
    return df


# ── Main preprocessing entry point ────────────────────────────────────────


def load_and_preprocess(
    data_dir: str | Path = ".",
    standardize: bool = True,
    test_size: float = 0.2,
    random_state: int = 2026,
) -> dict:
    """
    Full preprocessing pipeline for the Spaceship Titanic dataset.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing train.csv, test.csv, sample_submission.csv.
    standardize : bool
        Whether to apply StandardScaler to numeric features.
    test_size : float
        Fraction of training data to hold out for validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: X_train, X_val, y_train, y_val, X_test, test_passenger_ids,
                    num_features, cat_features, scaler, encoder, preprocessor_info
    """
    base = Path(data_dir)
    train = pd.read_csv(base / "train.csv")
    test = pd.read_csv(base / "test.csv")

    y_full = train["Transported"].astype(int).to_numpy()
    train_x = train.drop(columns=["Transported"])
    n_train = len(train_x)
    full = pd.concat([train_x, test.copy()], axis=0, ignore_index=True)

    # Step 1: Split core columns
    full = split_passenger_id(full)
    full = split_cabin(full)
    full = split_name(full)

    # Step 2: Pre-imputation derived columns
    full["SpendMissingCount"] = full[SPEND_COLS].isna().sum(axis=1)
    full["SpendKnownCount"] = full[SPEND_COLS].notna().sum(axis=1)
    full["TotalSpendRaw"] = full[SPEND_COLS].sum(axis=1, min_count=1)
    full["AgeMissing"] = full["Age"].isna().astype(int)
    full["CabinNumMissing"] = full["CabinNum"].isna().astype(int)

    # Step 3: Hierarchical imputation
    impute_home_planet(full)
    impute_categoricals(full)
    impute_cryosleep(full)
    impute_spend(full)
    # Compute TotalSpend early so we can correct CryoSleep
    full["TotalSpend"] = full[SPEND_COLS].sum(axis=1)
    full.loc[full["TotalSpend"].gt(0) & full["CryoSleep"].eq(True), "CryoSleep"] = False
    impute_vip(full)
    impute_numeric(full)

    # Step 4: Feature engineering (base categoricals must be strings first)
    cat_base = ["HomePlanet", "CryoSleep", "Destination", "VIP", "CabinDeck", "CabinSide", "Surname"]
    full = finalize_categoricals(full, cat_base)
    full = engineer_spend_features(full)
    full = engineer_group_features(full)
    full = engineer_interaction_features(full)

    # Step 5: Categorical finalization (re-apply to ensure all are clean)
    full = finalize_categoricals(full, cat_base)

    # Feature list definitions
    cat_features = [
        "HomePlanet", "CryoSleep", "Destination", "VIP",
        "CabinDeck", "CabinSide", "DeckSide", "HomeDestination",
        "HomeDeck", "DestinationDeck", "CryoHome", "Surname",
    ]

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

    # Ensure numeric columns are clean
    for col in num_features:
        full[col] = pd.to_numeric(full[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        full[col] = full[col].fillna(full[col].median()).astype(float)

    # Split back into train + test
    features = num_features + cat_features
    X_full = full.iloc[:n_train][features].reset_index(drop=True)
    X_test = full.iloc[n_train:][features].reset_index(drop=True)
    test_ids = test["PassengerId"].values

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=test_size, random_state=random_state, stratify=y_full
    )

    result = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "X_test": X_test,
        "test_passenger_ids": test_ids,
        "num_features": num_features,
        "cat_features": cat_features,
        "scaler": None,
        "encoder": None,
        "preprocessor_info": missing_value_report(train),
    }

    # Standardization (optional, not applied to tree-based features by default)
    if standardize:
        scaler = StandardScaler()
        num_cols = [c for c in num_features if c in X_train.columns]
        X_train_num = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
        X_val_num = pd.DataFrame(scaler.transform(X_val[num_cols]), columns=num_cols, index=X_val.index)
        X_test_num = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)
        for col in num_cols:
            X_train[col] = X_train_num[col]
            X_val[col] = X_val_num[col]
            X_test[col] = X_test_num[col]
        result["scaler"] = scaler

    return result


if __name__ == "__main__":
    data = load_and_preprocess(
        data_dir=Path(__file__).resolve().parent / "project  information" / "spaceship-titanic dataset",
        standardize=False,
    )
    print("Preprocessing complete.")
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  X_val:   {data['X_val'].shape}")
    print(f"  X_test:  {data['X_test'].shape}")
    print(f"  Numeric features: {len(data['num_features'])}")
    print(f"  Categorical features: {len(data['cat_features'])}")
    missing = data["preprocessor_info"]
    print(f"\nMissing value summary (train):\n{missing[missing['Missing'] > 0].to_string(index=False)}")
