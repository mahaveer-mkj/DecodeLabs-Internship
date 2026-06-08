"""
DecodeLabs — Project 2: Data Classification Using AI
Supervised Learning pipeline for the Iris benchmark dataset
using K‑Nearest Neighbors (KNN).
--------------------------------------------------
All requirements met:
- Shuffling, 80/20 split, StandardScaler (mean=0, var=1)
- KNN with elbow method using 5‑fold cross‑validation (NO test set leakage)
- Confusion matrix, precision, recall, F1 score
- Fully configurable, modular, PEP‑8, Google docstrings
- Saves plots to ./outputs/ directory
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Configuration & output directory
# ----------------------------------------------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "font.family": "DejaVu Sans"})


# ----------------------------------------------------------------------
# 1. Data ingestion & preprocessing (shuffle, split, scale)
# ----------------------------------------------------------------------
def load_and_prepare(
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    """
    Load Iris, shuffle, split into train/test, and apply StandardScaler.

    Args:
        test_size: Fraction for testing (default 0.2).
        random_state: Seed for reproducibility.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, target_names
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = list(iris.target_names)

    # Shuffle to eliminate order bias
    indices = np.arange(len(X))
    rng = np.random.default_rng(seed=random_state)
    rng.shuffle(indices)
    X, y = X[indices], y[indices]

    # Stratified split ensures class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale: fit only on training data, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=" * 60)
    print("  DATA PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total samples      : {len(X)}")
    print(f"  Training samples   : {len(X_train_scaled)} ({int((1-test_size)*100)}%)")
    print(f"  Testing samples    : {len(X_test_scaled)} ({int(test_size*100)}%)")
    print(f"  Classes            : {target_names}")
    print(f"  Scaled feature range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    print("=" * 60)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, target_names


# ----------------------------------------------------------------------
# 2. Elbow method with cross‑validation (CORRECT – no test set leakage)
# ----------------------------------------------------------------------
def find_optimal_k_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_k: int = 30,
    cv_folds: int = 5,
    random_state: int = 42,
    save_fig: bool = True,
) -> int:
    """
    Determine optimal K using k‑fold cross‑validation on the training set.

    Args:
        X_train, y_train: Training data (already scaled).
        max_k: Maximum number of neighbours to try.
        cv_folds: Number of cross‑validation folds.
        random_state: Seed for reproducible folds.
        save_fig: Whether to save the elbow plot.

    Returns:
        Optimal K (lowest cross‑validation error rate).
    """
    k_range = np.arange(1, max_k + 1)
    cv_errors = []

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=skf, scoring='accuracy')
        error = 1 - scores.mean()
        cv_errors.append(error)

    optimal_k = k_range[np.argmin(cv_errors)]

    # Plot elbow curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_range, cv_errors, color="steelblue", linestyle="--", marker="o",
            markerfacecolor="tomato", markersize=8, linewidth=2, label="CV Error Rate")
    ax.axvline(x=optimal_k, color="green", linestyle=":", linewidth=2,
               label=f"Optimal K = {optimal_k}  (error={cv_errors[optimal_k-1]:.3f})")
    ax.set_xlabel("K (Number of Neighbours)", fontsize=12)
    ax.set_ylabel("Cross‑Validation Error Rate", fontsize=12)
    ax.set_title("Elbow Method — 5‑fold CV Error vs. K", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(range(1, max_k + 1, 2))
    fig.tight_layout()

    if save_fig:
        path = os.path.join(OUTPUT_DIR, "elbow_plot.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"\n  [Elbow plot] saved → {path}")

    plt.show()
    print(f"\n  Optimal K (via CV) : {optimal_k}")
    print(f"  Lowest CV error    : {cv_errors[optimal_k-1]:.4f}")
    return optimal_k


# ----------------------------------------------------------------------
# 3. Train final model
# ----------------------------------------------------------------------
def train_model(X_train: np.ndarray, y_train: np.ndarray, k: int) -> KNeighborsClassifier:
    """Train KNN classifier with the chosen K."""
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    print(f"\n  Model trained : KNN(K={k})")
    return model


# ----------------------------------------------------------------------
# 4. Evaluation: confusion matrix, classification report, per‑class table
# ----------------------------------------------------------------------
def evaluate_model(
    model: KNeighborsClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str],
    save_fig: bool = True,
) -> None:
    """
    Generate all evaluation metrics. Explains why accuracy alone is insufficient.

    Metrics:
      - Confusion matrix (with annotations)
      - Precision, Recall, F1 per class
      - Per‑class F1 table for readability
    """
    y_pred = model.predict(X_test)

    # Raw accuracy (for reference, but not the main metric)
    accuracy = np.mean(y_pred == y_test)
    print(f"\n  Raw accuracy      : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print("  (Accuracy can be misleading – see precision/recall/F1 below)\n")

    # Classification report
    print("=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion matrix with heatmap
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix\n(Diagonal = correct predictions)", fontsize=11, fontweight="bold")
    fig.tight_layout()

    if save_fig:
        path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        fig.savefig(path, bbox_inches="tight")
        print("  [Confusion matrix] saved → confusion_matrix.png")

    plt.show()

    # Per‑class F1 table (pandas for nice display)
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    rows = {
        cls: {
            "Precision": report_dict[cls]["precision"],
            "Recall": report_dict[cls]["recall"],
            "F1-Score": report_dict[cls]["f1-score"],
            "Support": int(report_dict[cls]["support"]),
        }
        for cls in target_names
    }
    df_metrics = pd.DataFrame(rows).T.round(4)
    print("\n  Per‑class metrics table:")
    print(df_metrics.to_string())
    print()


# ----------------------------------------------------------------------
# 5. Main orchestrator
# ----------------------------------------------------------------------
def run_pipeline(
    test_size: float = 0.20,
    random_state: int = 42,
    max_k: int = 30,
    cv_folds: int = 5,
) -> None:
    """
    Execute the complete KNN classification pipeline.

    Args:
        test_size: Fraction for testing (default 0.2).
        random_state: Global seed for reproducibility.
        max_k: Maximum K to evaluate in elbow method.
        cv_folds: Number of cross‑validation folds.
    """
    print("\n" + "▓" * 60)
    print("  DecodeLabs · Project 2 · Correct KNN Pipeline")
    print("▓" * 60)

    # Step 1: Load, shuffle, split, scale
    X_train, X_test, y_train, y_test, scaler, target_names = load_and_prepare(
        test_size=test_size, random_state=random_state
    )

    # Step 2: Find optimal K using cross‑validation (NO TEST SET LEAKAGE)
    optimal_k = find_optimal_k_cv(
        X_train, y_train,
        max_k=max_k,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    # Step 3: Train final model with optimal K
    model = train_model(X_train, y_train, k=optimal_k)

    # Step 4: Evaluate on untouched test set
    evaluate_model(model, X_test, y_test, target_names)

    print("▓" * 60)
    print("  Pipeline complete. All plots saved in './outputs/'")
    print("▓" * 60 + "\n")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline(
        test_size=0.20,      # configurable
        random_state=42,     # configurable
        max_k=30,            # configurable
        cv_folds=5,          # configurable
    )
