"""
Modeling utilities for the credit card affinity model (Phase 3).

Provides:
  - Feature set definitions (baseline, static, all)
  - Leakage-free sklearn pipelines
  - Cross-validation and evaluation helpers
  - ROC, PR, Lift, Top-N plotting
  - Feature importance visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.features import BASELINE_FEATURE_COLS, STATIC_FEATURE_COLS

# ── Constants ────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER_RANDOM_SEARCH = 30
SCORING_PRIMARY = "roc_auc"
SCORING_METRICS = ["roc_auc", "average_precision", "f1", "precision", "recall"]
TOP_N_PERCENTILES = [0.05, 0.10]


# ── Feature Sets ─────────────────────────────────────────────────────────────


def get_feature_sets(fm: pd.DataFrame) -> dict[str, list[str]]:
    """
    Return dict of feature set names to column lists.

    - 'baseline': 5 assignment-required features
    - 'static_all': 22 static/contextual features (superset of baseline)
    - 'all_features': all ~1576 features
    """
    meta_cols = {"account_id", "target", "split"}
    all_features = [c for c in fm.columns if c not in meta_cols]

    # Static features that actually exist in fm
    static_all = [c for c in STATIC_FEATURE_COLS if c in fm.columns]

    return {
        "baseline": [c for c in BASELINE_FEATURE_COLS if c in fm.columns],
        "static_all": static_all,
        "all_features": all_features,
    }


# ── Pipeline Construction ────────────────────────────────────────────────────


def build_pipeline(
    model, feature_cols: list[str], scale: bool = False
) -> Pipeline:
    """
    Build a leakage-free sklearn Pipeline.

    Steps: ColumnSelector → SimpleImputer(median) → [StandardScaler] → Model
    """
    steps = []

    # Imputer
    steps.append(("imputer", SimpleImputer(strategy="median")))

    # Optional scaling (for linear models)
    if scale:
        steps.append(("scaler", StandardScaler()))

    # Model
    steps.append(("model", model))

    pipe = Pipeline(steps)

    return pipe


# ── Cross-Validation ─────────────────────────────────────────────────────────


def get_cv_splitter() -> StratifiedKFold:
    """Return the shared CV splitter for fair comparison across models."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# ── Test-Set Evaluation ──────────────────────────────────────────────────────


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a fitted pipeline on the test set.

    Returns dict with: y_proba, roc_auc, avg_precision, f1, precision, recall,
                        classification_report_str, confusion_matrix.
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    return {
        "y_proba": y_proba,
        "y_pred": y_pred,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
    }


# ── ROC Curves ───────────────────────────────────────────────────────────────


def plot_roc_curves(results: dict[str, dict], ax=None) -> plt.Figure:
    """
    Plot overlaid ROC curves for all models.

    Parameters
    ----------
    results : dict mapping model_name -> evaluate_on_test output (must include y_proba)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.get_figure()

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
        ax.plot(fpr, tpr, label=f'{name} (AUC={res["roc_auc"]:.3f})')

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")

    return fig


# ── Precision-Recall Curves ──────────────────────────────────────────────────


def plot_pr_curves(results: dict[str, dict], ax=None) -> plt.Figure:
    """Plot overlaid Precision-Recall curves."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.get_figure()

    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_proba"])
        ax.plot(rec, prec, label=f'{name} (AP={res["avg_precision"]:.3f})')

    # Baseline: prevalence
    prevalence = np.mean(list(results.values())[0]["y_true"])
    ax.axhline(y=prevalence, color="k", linestyle="--", alpha=0.5, label=f"Random ({prevalence:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Test Set")
    ax.legend(loc="upper right")

    return fig


# ── Lift Curve ───────────────────────────────────────────────────────────────


def compute_lift_curve(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 100
) -> pd.DataFrame:
    """
    Compute lift curve data.

    Returns DataFrame with columns: pct_contacted, cumulative_positives, lift.
    """
    order = np.argsort(-y_proba)
    y_sorted = np.array(y_true)[order]
    n = len(y_sorted)
    n_pos = y_sorted.sum()

    pct_contacted = np.arange(1, n + 1) / n
    cum_positives = np.cumsum(y_sorted) / n_pos
    lift = cum_positives / pct_contacted

    # Subsample for plotting
    step = max(1, n // n_bins)
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)

    return pd.DataFrame(
        {
            "pct_contacted": pct_contacted[idx],
            "cumulative_positives": cum_positives[idx],
            "lift": lift[idx],
        }
    )


def plot_lift_curve(results: dict[str, dict]) -> plt.Figure:
    """Plot lift curves for all models + random baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Cumulative Gains
    ax = axes[0]
    for name, res in results.items():
        lc = compute_lift_curve(res["y_true"], res["y_proba"])
        ax.plot(lc["pct_contacted"], lc["cumulative_positives"], label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("% Kunden kontaktiert")
    ax.set_ylabel("% Käufer erreicht (kumulativ)")
    ax.set_title("Cumulative Gains Chart")
    ax.legend(loc="lower right")

    # Lift
    ax = axes[1]
    for name, res in results.items():
        lc = compute_lift_curve(res["y_true"], res["y_proba"])
        ax.plot(lc["pct_contacted"], lc["lift"], label=name)
    ax.axhline(y=1, color="k", linestyle="--", alpha=0.5, label="Random (Lift=1)")
    ax.set_xlabel("% Kunden kontaktiert")
    ax.set_ylabel("Lift")
    ax.set_title("Lift Curve")
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


# ── Top-N% Customer Lists ───────────────────────────────────────────────────


def compute_top_n_list(
    ids: np.ndarray,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    pct: float,
) -> dict:
    """
    Compute Top-N% customer list.

    Returns dict with: ids, n_total, n_selected, n_positive, precision_at_n.
    """
    n = len(ids)
    n_select = max(1, int(n * pct))
    order = np.argsort(-y_proba)[:n_select]

    selected_ids = set(np.array(ids)[order])
    selected_true = np.array(y_true)[order]
    n_positive = int(selected_true.sum())
    precision_at_n = n_positive / n_select

    return {
        "ids": selected_ids,
        "n_total": n,
        "n_selected": n_select,
        "n_positive": n_positive,
        "precision_at_n": precision_at_n,
    }


def compare_top_n_lists(top_n_dict: dict[str, dict]) -> pd.DataFrame:
    """
    Compute Jaccard overlap between all model pairs' Top-N lists.

    Parameters
    ----------
    top_n_dict : dict mapping model_name -> compute_top_n_list output

    Returns
    -------
    Symmetric DataFrame of Jaccard similarity scores.
    """
    names = list(top_n_dict.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            s1 = top_n_dict[n1]["ids"]
            s2 = top_n_dict[n2]["ids"]
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            matrix[i, j] = intersection / union if union > 0 else 0

    return pd.DataFrame(matrix, index=names, columns=names)


# ── Feature Importance ───────────────────────────────────────────────────────


def plot_feature_importance(
    importances: pd.Series, name: str, top_n: int = 20, ax=None
) -> plt.Figure:
    """Plot horizontal bar chart of top-N feature importances."""
    top = importances.nlargest(top_n).sort_values()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    ax.barh(top.index, top.values)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{top_n} Feature Importance — {name}")
    ax.tick_params(axis="y", labelsize=9)

    return fig


# ── Confusion Matrix Grid ───────────────────────────────────────────────────


def plot_confusion_matrices(results: dict[str, dict]) -> plt.Figure:
    """Plot confusion matrices for all models in a grid."""
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_2d(axes)

    for idx, (name, res) in enumerate(results.items()):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ConfusionMatrixDisplay(
            confusion_matrix=res["cm"],
            display_labels=["Non-Buyer", "Buyer"],
        ).plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(name)

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    return fig
