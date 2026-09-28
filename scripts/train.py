"""End-to-end pipeline runner.

Runs: load -> clean -> EDA -> baseline & improved models -> evaluation ->
predictive bed-allocation simulation. Saves all figures, metrics, and the
best model into `results/`.

Usage:
    python scripts/train.py
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, TARGET_COL  # noqa: E402
from src.data_utils import clean_data, describe_cleaning, load_raw_data  # noqa: E402
from src.features import prepare_train_test  # noqa: E402
from src.models import (  # noqa: E402
    plot_feature_importance,
    plot_metrics,
    save_model,
    train_and_evaluate,
)
from src.scheduling import build_arrival_scenario, compare_policies  # noqa: E402


def run_eda(df_clean: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Generate the standard EDA figures used in the report."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- cleaning summary ------------------------------------------------
    summary = describe_cleaning(df_raw, df_clean)
    summary.to_csv(RESULTS_DIR / "cleaning_summary.csv", index_label="stage")
    print("\n[cleaning summary]")
    print(summary.to_string())

    # --- target distribution ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    df_clean[TARGET_COL].hist(bins=range(0, 16), ax=axes[0], color="#4C72B0", edgecolor="white")
    axes[0].set_title("Distribution of time_in_hospital (days)")
    axes[0].set_xlabel("days")
    (np.log1p(df_clean[TARGET_COL])).hist(bins=40, ax=axes[1], color="#DD8452", edgecolor="white")
    axes[1].set_title("log1p(time_in_hospital)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- missingness -------------------------------------------------------
    missing = (df_clean.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        fig, ax = plt.subplots(figsize=(8, max(3, len(missing) * 0.35)))
        ax.barh(missing.index, missing.values, color="#C44E52")
        ax.set_title("Missing values (%) after cleaning")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "missing_values.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- numeric correlations with target ---------------------------------
    num_cols = [c for c in df_clean.columns if df_clean[c].dtype in ("int64", "float64")]
    corr = df_clean[num_cols].corrwith(df_clean[TARGET_COL]).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, len(corr) * 0.35)))
    colors = ["#C44E52" if v < 0 else "#55A868" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.set_title(f"Correlation with {TARGET_COL}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- diagnosis category mix -------------------------------------------
    diag_cols = [c for c in df_clean.columns if c.startswith("diag_") and c.endswith("_cat")]
    if diag_cols:
        top = df_clean[diag_cols[0]].value_counts().head(12)
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.bar(range(len(top)), top.values, color="#4C72B0")
        ax.set_title("Primary diagnosis category (top 12)")
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(top.index, rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "diagnosis_categories.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- LOS by clinical workload ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col in zip(axes, ["num_lab_procedures", "num_medications"]):
        binned = pd.cut(df_clean[col], bins=10)
        means = df_clean.groupby(binned, observed=True)[TARGET_COL].mean()
        means.plot(kind="line", marker="o", ax=ax, color="#4C72B0")
        ax.set_title(f"Mean LOS vs {col}")
        ax.set_xlabel(col)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "los_vs_workload.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[EDA figures saved to {FIGURES_DIR}]")


def main() -> None:
    t_start = time.time()

    print("== 1/5 load raw data ==")
    df_raw = load_raw_data()
    print(f"  raw shape: {df_raw.shape}")

    print("\n== 2/5 clean & feature engineering ==")
    df_clean = clean_data(df_raw)
    print(f"  clean shape: {df_clean.shape}")

    print("\n== 3/5 exploratory data analysis ==")
    run_eda(df_clean, df_raw)

    print("\n== 4/5 model training & evaluation ==")
    X_train, X_test, y_train, y_test = prepare_train_test(df_clean)
    metrics, fitted, X_train_enc = train_and_evaluate(X_train, X_test, y_train, y_test)
    print(metrics.to_string())
    metrics.to_csv(RESULTS_DIR / "metrics.csv")

    plot_metrics(metrics, save_path=FIGURES_DIR / "model_comparison.png")

    # feature importance from the best tree model
    tree_name = next(
        (m for m in ["HistGradientBoosting", "RandomForest", "XGBoost"] if m in fitted),
        None,
    )
    if tree_name:
        from src.features import encode_features

        X_test_enc = encode_features(X_test).reindex(columns=X_train_enc.columns, fill_value=0)
        X_test_enc = X_test_enc.set_axis(X_test.index, axis=0)  # align with y_test
        plot_feature_importance(
            fitted[tree_name], X_test_enc, y_test,
            save_path=FIGURES_DIR / "feature_importance.png",
        )
        best_model = fitted[tree_name]
        print(f"\n[feature importance plot from {tree_name}]")
    else:
        best_model = fitted["LinearRegression"]
        print("\n[no tree model available; skipping feature importance]")

    saved = save_model(best_model, tree_name or "LinearRegression")
    print(f"  best model saved: {saved}")

    print("\n== 5/5 predictive bed-allocation simulation ==")
    arrivals = build_arrival_scenario(df_clean, n_patients=1200, n_beds=80)
    sim_df = compare_policies(arrivals, n_beds=80)
    print(sim_df.to_string(index=False))
    sim_df.to_csv(RESULTS_DIR / "scheduling_summary.csv", index=False)

    # occupancy curves
    from src.scheduling import BedSimulator

    sim_fifo = BedSimulator(n_beds=80).simulate(arrivals, policy="fifo")
    sim_spt = BedSimulator(n_beds=80).simulate(arrivals, policy="spt")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sim_fifo["daily_occupancy"], label="FIFO", color="#C44E52")
    ax.plot(sim_spt["daily_occupancy"], label="SPT (prediction-driven)", color="#55A868")
    ax.axhline(80, color="grey", ls="--", lw=0.8)
    ax.set_title("Daily bed occupancy: FIFO vs prediction-driven scheduling")
    ax.set_xlabel("day")
    ax.set_ylabel("occupied beds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scheduling_occupancy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  scheduling figure saved to {FIGURES_DIR / 'scheduling_occupancy.png'}")

    print(f"\nDone in {time.time() - t_start:.1f}s — outputs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
