"""Model training, evaluation, and visualisation.

Design follows a 'baseline reproduction -> improvement' narrative that is
easy to explain in a statement of purpose:

  1. Baseline: Linear Regression on raw features (simple, interpretable).
  2. Improvement 1 (model): tree ensembles that capture non-linear
     interactions (Random Forest, HistGradientBoosting, optionally XGBoost).
  3. Improvement 2 (data): the same best model is re-run on a
     feature-ablation version (without engineered clinical features) to
     quantify what the feature engineering adds.
"""
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_SEED
from src.features import encode_features


def get_models() -> dict:
    """Return the model zoo: name -> unfitted estimator.

    XGBoost is included automatically when it is installed.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=None, n_jobs=-1,
            random_state=RANDOM_SEED, min_samples_leaf=5,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            random_state=RANDOM_SEED,
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.08, max_depth=6,
            subsample=0.9, colsample_bytree=0.8, n_jobs=-1,
            random_state=RANDOM_SEED, verbosity=0,
        )
    except ImportError:
        pass  # optional dependency
    return models


def _score(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, use_scaling_linear=True):
    """Train every model and return (metrics_df, fitted_models).

    Linear models receive a standardised feature matrix; tree models use the
    raw encoded matrix. `feature_engineered` flags which feature set was used.
    """
    X_train_enc = encode_features(X_train)
    X_test_enc = encode_features(X_test)

    # keep the mapping between encodings identical
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    metrics_rows = []
    fitted = {}

    for name, est in get_models().items():
        t0 = time.time()
        if name == "LinearRegression" and use_scaling_linear:
            pipe = Pipeline(
                [("scale", StandardScaler()), ("reg", est)]
            )
            pipe.fit(X_train_enc, y_train)
            y_pred = pipe.predict(X_test_enc)
            fitted[name] = pipe
        else:
            est.fit(X_train_enc, y_train)
            y_pred = est.predict(X_test_enc)
            fitted[name] = est

        score = _score(y_test, y_pred)
        score["model"] = name
        score["train_time_s"] = round(time.time() - t0, 1)
        metrics_rows.append(score)

    metrics = pd.DataFrame(metrics_rows).set_index("model")
    metrics = metrics[["MAE", "RMSE", "R2", "train_time_s"]].sort_values("MAE")
    return metrics, fitted, X_train_enc


def plot_metrics(metrics: pd.DataFrame, save_path=None) -> plt.Figure:
    """Grouped bar chart of MAE / RMSE / R2 per model."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        values = metrics[metric].astype(float)
        bars = ax.barh(metrics.index, values, color="#4C72B0")
        ax.set_title(f"{metric} (lower is better)" if metric != "R2" else "R2 (higher is better)")
        for bar, v in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", ha="left", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    model,
    X_test_enc: pd.DataFrame,
    y_test: pd.Series,
    save_path=None,
    top_n: int = 25,
    n_samples: int = 5000,
    n_repeats: int = 3,
) -> plt.Figure:
    """Feature importance.

    Tree models with a native `feature_importances_` (RandomForest, XGBoost)
    use it directly. Others (e.g. HistGradientBoosting in recent scikit-learn
    versions) fall back to permutation importance, which is model-agnostic and
    arguably more trustworthy for a portfolio.
    """
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        from sklearn.inspection import permutation_importance

        X_sub = X_test_enc.sample(n=min(n_samples, len(X_test_enc)), random_state=RANDOM_SEED)
        y_sub = y_test.loc[X_sub.index]
        print("  computing permutation importance (sampled test set)...")
        pi = permutation_importance(
            model, X_sub, y_sub,
            scoring="neg_mean_absolute_error",
            n_repeats=n_repeats, random_state=RANDOM_SEED, n_jobs=-1,
        )
        importances = pi.importances_mean

    feature_names = list(X_test_enc.columns)
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.32)))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        importances[order][::-1],
        color="#55A868",
    )
    ax.set_title("Top %d features (tree importance or permutation importance)" % top_n)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def save_model(model, name: str, path=None) -> str:
    path = path or (MODELS_DIR / f"{name}.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return str(path)
