"""
Exploratory Data Analysis (EDA)
================================
Generates visualizations and statistical summaries for the Spaceship Titanic dataset.
Outputs plots to `eda_plots/` and prints interpretive findings to the console.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
PLOT_DIR = Path(__file__).resolve().parent / "eda_plots"
RANDOM_STATE = 2026

# ── Styling ───────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def _ensure_dir() -> Path:
    PLOT_DIR.mkdir(exist_ok=True)
    return PLOT_DIR


# ── Data loading helper ───────────────────────────────────────────────────


def load_data(data_dir: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir)
    train = pd.read_csv(base / "train.csv")
    test = pd.read_csv(base / "test.csv")
    return train, test


# ── Section 1: Dataset overview ───────────────────────────────────────────


def dataset_overview(train: pd.DataFrame, test: pd.DataFrame) -> None:
    out_dir = _ensure_dir()
    print("=" * 70)
    print("SECTION 1: DATASET OVERVIEW")
    print("=" * 70)

    print(f"\nTraining samples:   {len(train):,}")
    print(f"Test samples:       {len(test):,}")
    print(f"Training columns:   {len(train.columns)}")
    print(f"Features (excl. target): {len(train.columns) - 2}")

    target_counts = train["Transported"].value_counts()
    print(f"\nTarget distribution (Transported):")
    print(f"  True:  {target_counts.get(True, 0):,}  ({target_counts.get(True, 0) / len(train):.1%})")
    print(f"  False: {target_counts.get(False, 0):,}  ({target_counts.get(False, 0) / len(train):.1%})")

    # Dtype breakdown
    dtypes = train.dtypes.value_counts()
    print(f"\nColumn data types:")
    for dtype, count in dtypes.items():
        print(f"  {dtype}: {count}")

    # Pie chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].pie(
        target_counts.values, labels=["Not Transported", "Transported"],
        autopct="%1.1f%%", colors=["#E74C3C", "#2ECC71"], startangle=90, explode=(0, 0.03),
    )
    axes[0].set_title("Target Variable (Transported)", fontweight="bold")
    axes[1].bar(dtypes.index.astype(str), dtypes.values, color=sns.color_palette("muted")[:len(dtypes)])
    axes[1].set_title("Column Data Types", fontweight="bold")
    axes[1].set_ylabel("Count")
    fig.suptitle("Dataset Overview", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "01_dataset_overview.png")
    plt.close(fig)
    print(f"  -> Saved {out_dir / '01_dataset_overview.png'}")


# ── Section 2: Missing value analysis ─────────────────────────────────────


def missing_value_analysis(train: pd.DataFrame) -> None:
    out_dir = _ensure_dir()
    print("\n" + "=" * 70)
    print("SECTION 2: MISSING VALUE ANALYSIS")
    print("=" * 70)

    total = len(train)
    missing = (train.isna().sum() / total * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    print(f"\nColumns with missing values ({len(missing)}/{len(train.columns)}):")
    for col, pct in missing.items():
        print(f"  {col:<20s}: {pct:5.1f}%  ({int(train[col].isna().sum()):,} missing)")

    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(missing.index, missing.values, color=sns.color_palette("Reds_r", len(missing)))
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Value Percentages by Column", fontweight="bold")
    ax.invert_yaxis()
    for bar, pct in zip(bars, missing.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "02_missing_values.png")
    plt.close(fig)
    print(f"  -> Saved {out_dir / '02_missing_values.png'}")

    # Missing-value heatmap pattern
    missing_cols = missing.index.tolist()
    if missing_cols:
        fig, ax = plt.subplots(figsize=(14, 5))
        sample_missing = train[missing_cols].isna().sample(min(2000, len(train)), random_state=RANDOM_STATE)
        sns.heatmap(sample_missing.T, cmap=["#F0F0F0", "#C0392B"], cbar=False, xticklabels=False, ax=ax)
        ax.set_title("Missing Value Pattern (sample of 2,000 rows)", fontweight="bold")
        ax.set_xlabel("Passenger (sampled)")
        fig.tight_layout()
        fig.savefig(out_dir / "02b_missing_pattern.png")
        plt.close(fig)
        print(f"  -> Saved {out_dir / '02b_missing_pattern.png'}")
    else:
        print("  (No missing values to visualize)")


# ── Section 3: Numeric distributions ──────────────────────────────────────


def numeric_distributions(train: pd.DataFrame) -> None:
    out_dir = _ensure_dir()
    print("\n" + "=" * 70)
    print("SECTION 3: NUMERIC FEATURE DISTRIBUTIONS")
    print("=" * 70)

    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["PassengerId"]]
    # Split into two groups: spend columns and others
    spend_cols_in = [c for c in SPEND_COLS if c in numeric_cols]
    other_num = [c for c in numeric_cols if c not in spend_cols_in]

    # Histograms by Transported status
    _plot_feature_hists(train, spend_cols_in, "03_spend_distributions.png", "Spend Feature Distributions by Transported", ncols=3)
    _plot_feature_hists(train, other_num, "03b_other_numeric_distributions.png", "Other Numeric Distributions by Transported", ncols=3)

    # Skewness / kurtosis table
    print("\nSkewness and kurtosis of numeric features:")
    for col in numeric_cols[:12]:
        col_data = train[col].dropna()
        skew = col_data.skew()
        kurt = col_data.kurtosis()
        flag = " *** HIGHLY SKEWED" if abs(skew) > 1.5 else ""
        print(f"  {col:<20s}  skew={skew:+.3f}  kurtosis={kurt:+.3f}{flag}")
    if len(numeric_cols) > 12:
        print(f"  ... ({len(numeric_cols) - 12} more columns)")


def _plot_feature_hists(train: pd.DataFrame, cols: list[str], fname: str, suptitle: str, ncols: int = 3) -> None:
    out_dir = _ensure_dir()
    n = len(cols)
    if n == 0:
        return
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        for label_val, color, lbl in [(0, "#E74C3C", "Not Transported"), (1, "#2ECC71", "Transported")]:
            subset = train.loc[train["Transported"].astype(int) == label_val, col].dropna()
            if len(subset):
                ax.hist(subset, bins=40, alpha=0.55, color=color, label=lbl, density=True)
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(suptitle, fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / fname)
    plt.close(fig)
    print(f"  -> Saved {out_dir / fname}")


# ── Section 4: Categorical feature analysis ───────────────────────────────


def categorical_analysis(train: pd.DataFrame) -> None:
    out_dir = _ensure_dir()
    print("\n" + "=" * 70)
    print("SECTION 4: CATEGORICAL FEATURE ANALYSIS")
    print("=" * 70)

    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "CabinDeck", "CabinSide"]
    cat_cols = [c for c in cat_cols if c in train.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for i, col in enumerate(cat_cols):
        ax = axes_flat[i]
        ct = pd.crosstab(train[col].fillna("Missing"), train["Transported"], normalize="index")
        ct = ct.sort_values(True, ascending=False)
        ct.plot(kind="barh", stacked=True, ax=ax, color=["#E74C3C", "#2ECC71"], alpha=0.85)
        ax.set_title(f"{col} x Transported", fontweight="bold")
        ax.set_xlabel("Proportion")
        ax.legend(["Not Transported", "Transported"], fontsize=8)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Categorical Features vs. Target", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "04_categorical_vs_target.png")
    plt.close(fig)
    print(f"  -> Saved {out_dir / '04_categorical_vs_target.png'}")

    # Chi-squared association
    from scipy.stats import chi2_contingency
    print("\nChi-squared association with Transported:")
    for col in cat_cols:
        ct = pd.crosstab(train[col].fillna("Missing"), train["Transported"])
        chi2, p, dof, _ = chi2_contingency(ct)
        strength = "*** VERY STRONG" if p < 1e-20 else ("** STRONG" if p < 1e-6 else ("* MODERATE" if p < 0.01 else "weak"))
        print(f"  {col:<18s}  chi2={chi2:8.1f}  df={dof}  p={p:.2e}  [{strength}]")


# ── Section 5: Correlation analysis ───────────────────────────────────────


def correlation_analysis(train: pd.DataFrame) -> None:
    out_dir = _ensure_dir()
    print("\n" + "=" * 70)
    print("SECTION 5: CORRELATION ANALYSIS")
    print("=" * 70)

    # Build a numeric-only dataframe (encode binary/categorical where possible)
    df_num = train.select_dtypes(include=[np.number]).copy()
    # Encode Transported
    df_num["Transported_int"] = train["Transported"].astype(int)
    # Encode binary categoricals
    for col in ["CryoSleep", "VIP"]:
        if col in train.columns:
            df_num[f"{col}_int"] = train[col].map({True: 1, False: 0}).astype(float)

    # Correlation with target
    corr_with_target = df_num.corr()["Transported_int"].drop("Transported_int").sort_values(key=abs, ascending=False)
    print("\nTop correlations with Transported:")
    for col, val in corr_with_target.head(15).items():
        direction = "^ transported" if val > 0 else "v transported"
        print(f"  {col:<28s}: {val:+.4f}  ({direction})")
    for col, val in corr_with_target.tail(5).items():
        pass  # already covered

    # Correlation heatmap
    top_feats = corr_with_target.head(20).index.tolist() + ["Transported_int"]
    corr_mat = df_num[top_feats].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    sns.heatmap(
        corr_mat, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, linewidths=0.5, square=True,
        annot_kws={"size": 7}, ax=ax,
    )
    ax.set_title("Correlation Matrix (Top 20 Features vs. Transported)", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "05_correlation_heatmap.png")
    plt.close(fig)
    print(f"  -> Saved {out_dir / '05_correlation_heatmap.png'}")

    # Pairplot of top features
    top4 = corr_with_target.head(4).index.tolist()
    pair_df = df_num[top4 + ["Transported_int"]].sample(min(1500, len(train)), random_state=RANDOM_STATE).copy()
    pair_df["Transported"] = pair_df["Transported_int"].map({0: "Not Transported", 1: "Transported"})

    g = sns.pairplot(
        pair_df, hue="Transported", diag_kind="kde",
        palette={"Not Transported": "#E74C3C", "Transported": "#2ECC71"},
        plot_kws={"alpha": 0.5, "s": 12}, corner=True,
    )
    g.fig.suptitle("Pairwise Relationships: Top 4 Correlated Features", fontweight="bold", fontsize=14, y=1.01)
    g.fig.savefig(out_dir / "05b_pairplot.png")
    plt.close(g.fig)
    print(f"  -> Saved {out_dir / '05b_pairplot.png'}")


# ── Section 6: Outlier detection ──────────────────────────────────────────


def outlier_analysis(train: pd.DataFrame) -> None:
    out_dir = _ensure_dir()
    print("\n" + "=" * 70)
    print("SECTION 6: OUTLIER DETECTION")
    print("=" * 70)

    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    # Focus on spend columns + Age
    focus_cols = [c for c in SPEND_COLS + ["Age"] if c in numeric_cols]

    outlier_summary = []
    for col in focus_cols:
        col_data = train[col].dropna()
        Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = int(((col_data < lower) | (col_data > upper)).sum())
        pct = n_outliers / len(col_data) * 100
        outlier_summary.append((col, n_outliers, pct, lower, upper))
        print(f"  {col:<18s}: {n_outliers:5d} outliers ({pct:5.1f}%)  bounds=[{lower:.1f}, {upper:.1f}]")

    # Boxplots
    n = len(focus_cols)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()
    for i, (col, n_out, pct, lo, hi) in enumerate(outlier_summary):
        ax = axes_flat[i]
        data_by_target = [
            train.loc[train["Transported"] == True, col].dropna(),
            train.loc[train["Transported"] == False, col].dropna(),
        ]
        bp = ax.boxplot(data_by_target, vert=True, patch_artist=True, labels=["Transported", "Not Trans."],
                        widths=0.5, flierprops=dict(marker=".", markersize=3, alpha=0.4))
        bp["boxes"][0].set_facecolor("#2ECC71")
        bp["boxes"][1].set_facecolor("#E74C3C")
        ax.set_title(f"{col}\n({n_out} outliers, {pct:.1f}%)", fontsize=10)
        ax.set_ylabel("Value")
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Outlier Analysis: Spend Features + Age", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "06_outlier_boxplots.png")
    plt.close(fig)
    print(f"  -> Saved {out_dir / '06_outlier_boxplots.png'}")


# ── Section 7: Group / Cabin analysis ─────────────────────────────────────


def group_and_cabin_analysis(train: pd.DataFrame) -> None:
    out_dir = _ensure_dir()
    print("\n" + "=" * 70)
    print("SECTION 7: GROUP AND CABIN ANALYSIS")
    print("=" * 70)

    df = train.copy()
    # Parse group and cabin
    df["GroupId"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df["GroupId"].map(df["GroupId"].value_counts())
    df["CabinDeck"] = df["Cabin"].str.split("/").str[0]
    df["CabinSide"] = df["Cabin"].str.split("/").str[2]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Group size vs transport rate
    gs = df.groupby("GroupSize")["Transported"].agg(["mean", "count"]).reset_index()
    gs = gs[gs["GroupSize"] <= 10]
    ax = axes[0, 0]
    ax.bar(gs["GroupSize"].astype(str), gs["mean"], color=["#2ECC71" if v > 0.5 else "#E74C3C" for v in gs["mean"]])
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Transport Rate by Group Size", fontweight="bold")
    ax.set_ylabel("Transport Rate")
    ax.set_xlabel("Group Size")

    # Group size distribution
    ax = axes[0, 1]
    gs_all = df["GroupSize"].value_counts().sort_index()
    gs_all = gs_all[gs_all.index <= 12]
    ax.bar(gs_all.index.astype(str), gs_all.values, color="#3498DB", alpha=0.85)
    ax.set_title("Group Size Distribution", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Group Size")

    # Cabin deck vs transport
    ax = axes[0, 2]
    deck_order = ["A", "B", "C", "D", "E", "F", "G", "T"]
    ct_deck = pd.crosstab(df["CabinDeck"], df["Transported"], normalize="index")
    ct_deck = ct_deck.reindex([d for d in deck_order if d in ct_deck.index])
    ct_deck.plot(kind="barh", stacked=True, ax=ax, color=["#E74C3C", "#2ECC71"], alpha=0.85)
    ax.set_title("Transport Rate by Cabin Deck", fontweight="bold")
    ax.set_xlabel("Proportion")
    ax.legend(["Not Transported", "Transported"], fontsize=8)

    # Cabin side vs transport
    ax = axes[1, 0]
    ct_side = pd.crosstab(df["CabinSide"], df["Transported"], normalize="index")
    ct_side.plot(kind="bar", stacked=True, ax=ax, color=["#E74C3C", "#2ECC71"], alpha=0.85)
    ax.set_title("Transport Rate by Cabin Side", fontweight="bold")
    ax.set_ylabel("Proportion")
    ax.legend(["Not Transported", "Transported"], fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Home planet vs destination
    ax = axes[1, 1]
    ct_hd = pd.crosstab(df["HomePlanet"].fillna("Unknown"), df["Destination"].fillna("Unknown"), normalize="index")
    sns.heatmap(ct_hd, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Home Planet -> Destination Heatmap", fontweight="bold")

    # Age distribution by transported
    ax = axes[1, 2]
    for label_val, color, lbl in [(True, "#2ECC71", "Transported"), (False, "#E74C3C", "Not Transported")]:
        subset = df.loc[df["Transported"] == label_val, "Age"].dropna()
        ax.hist(subset, bins=35, alpha=0.55, color=color, label=lbl, density=True)
    ax.set_title("Age Distribution by Transported", fontweight="bold")
    ax.set_xlabel("Age")
    ax.legend(fontsize=8)

    fig.suptitle("Group, Cabin & Demographic Analysis", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "07_group_cabin_analysis.png")
    plt.close(fig)
    print(f"  -> Saved {out_dir / '07_group_cabin_analysis.png'}")

    # Key findings
    solo_rate = gs[gs["GroupSize"] == 1]["mean"].values[0] if len(gs[gs["GroupSize"] == 1]) else float("nan")
    large_group = gs[gs["GroupSize"] >= 6]["mean"].mean() if len(gs[gs["GroupSize"] >= 6]) else float("nan")
    print(f"\nKey insights:")
    print(f"  Solo passenger transport rate:    {solo_rate:.3f}")
    print(f"  Large group (6+) transport rate:  {large_group:.3f}")
    for deck in ["A", "B", "C", "D", "E", "F", "G", "T"]:
        if deck in ct_deck.index:
            print(f"  Deck {deck} transport rate:         {ct_deck.loc[deck, True]:.3f}")


# ── Section 8: Spend behaviour analysis ───────────────────────────────────


def spend_analysis(train: pd.DataFrame) -> None:
    out_dir = _ensure_dir()
    print("\n" + "=" * 70)
    print("SECTION 8: SPEND BEHAVIOUR ANALYSIS")
    print("=" * 70)

    df = train.copy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Total spend vs transport
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1, min_count=1)
    for label_val, color, lbl in [(True, "#2ECC71", "Transported"), (False, "#E74C3C", "Not Transported")]:
        subset = df.loc[df["Transported"] == label_val, "TotalSpend"].dropna()
        subset_clipped = subset[subset < subset.quantile(0.95)]
        axes[0, 0].hist(subset_clipped, bins=50, alpha=0.55, color=color, label=lbl, density=True)
    axes[0, 0].set_title("Total Spend Distribution (95th %ile clipped)", fontweight="bold", fontsize=10)
    axes[0, 0].legend(fontsize=8)

    # Zero spend rate
    df["ZeroSpend"] = df["TotalSpend"].eq(0)
    zero_rate = df.groupby("Transported")["ZeroSpend"].mean()
    axes[0, 1].bar(["Not Transported", "Transported"], [zero_rate.get(False, 0), zero_rate.get(True, 0)],
                   color=["#E74C3C", "#2ECC71"], alpha=0.85)
    axes[0, 1].set_title("Zero-Spend Rate by Transport Status", fontweight="bold", fontsize=10)
    axes[0, 1].set_ylabel("Proportion with $0 spend")

    # Spend per service
    spend_means = df.groupby("Transported")[SPEND_COLS].mean()
    spend_means.T.plot(kind="bar", ax=axes[0, 2], color=["#E74C3C", "#2ECC71"], alpha=0.85)
    axes[0, 2].set_title("Mean Spend per Service by Transport Status", fontweight="bold", fontsize=10)
    axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=30, ha="right")
    axes[0, 2].legend(["Not Transported", "Transported"], fontsize=8)

    # CryoSleep x spend
    df["CryoStr"] = df["CryoSleep"].map({True: "CryoSleep", False: "Awake"}).fillna("Unknown")
    for i, (cryo_val, cryo_label) in enumerate([("CryoSleep", "#9B59B6"), ("Awake", "#F39C12")]):
        subset = df.loc[df["CryoStr"] == cryo_val, "TotalSpend"].dropna()
        subset_c = subset[subset < subset.quantile(0.95)] if len(subset) > 0 else subset
        axes[1, 0].hist(subset_c, bins=40, alpha=0.55, color=cryo_label, label=cryo_val, density=True)
    axes[1, 0].set_title("Total Spend by CryoSleep Status", fontweight="bold", fontsize=10)
    axes[1, 0].legend(fontsize=8)

    # Spend correlation heatmap
    corr_spend = df[SPEND_COLS + ["TotalSpend"]].corr()
    sns.heatmap(corr_spend, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1, 1], linewidths=0.5, square=True,
                annot_kws={"size": 8})
    axes[1, 1].set_title("Spend Feature Inter-Correlations", fontweight="bold", fontsize=10)

    # Services used
    df["ServicesUsed"] = df[SPEND_COLS].gt(0).sum(axis=1)
    svc_rate = df.groupby("ServicesUsed")["Transported"].agg(["mean", "count"])
    axes[1, 2].bar(svc_rate.index.astype(str), svc_rate["mean"],
                   color=["#2ECC71" if v > 0.5 else "#E74C3C" for v in svc_rate["mean"]], alpha=0.85)
    axes[1, 2].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[1, 2].set_title("Transport Rate by # of Services Used", fontweight="bold", fontsize=10)
    axes[1, 2].set_ylabel("Transport Rate")
    axes[1, 2].set_xlabel("Number of Services Used")

    fig.suptitle("Spend Behaviour Analysis", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "08_spend_analysis.png")
    plt.close(fig)
    print(f"  -> Saved {out_dir / '08_spend_analysis.png'}")

    print(f"\nKey spend insights:")
    print(f"  Mean spend (transported):     {df.loc[df['Transported'] == True, 'TotalSpend'].mean():.1f}")
    print(f"  Mean spend (not transported): {df.loc[df['Transported'] == False, 'TotalSpend'].mean():.1f}")
    print(f"  Zero-spend transport rate:    {df.loc[df['ZeroSpend'] == True, 'Transported'].mean():.3f}")
    print(f"  Nonzero-spend transport rate: {df.loc[df['ZeroSpend'] == False, 'Transported'].mean():.3f}")


# ── Main entry point ──────────────────────────────────────────────────────


def run_eda(data_dir: str | Path = ".") -> None:
    """Run the full EDA pipeline and generate all plots."""
    base = Path(data_dir)
    train_path = base / "train.csv"
    test_path = base / "test.csv"

    if not train_path.exists():
        # Try alternative path
        alt = Path(__file__).resolve().parent / "project  information" / "spaceship-titanic dataset"
        train_path = alt / "train.csv"
        test_path = alt / "test.csv"

    train, test = load_data(train_path.parent)
    print(f"Loaded train={train.shape}, test={test.shape}")
    print(f"All plots will be saved to: {_ensure_dir()}\n")

    dataset_overview(train, test)
    missing_value_analysis(train)
    numeric_distributions(train)
    categorical_analysis(train)
    correlation_analysis(train)
    outlier_analysis(train)
    group_and_cabin_analysis(train)
    spend_analysis(train)

    print("\n" + "=" * 70)
    print("EDA COMPLETE -- all plots saved to eda_plots/")
    print("=" * 70)


if __name__ == "__main__":
    run_eda()
