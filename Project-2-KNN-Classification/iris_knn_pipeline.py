"""
iris_knn_pipeline.py
====================
DecodeLabs Training Kit — Project 2: Data Classification Using AI
Supervised Learning pipeline for the Iris benchmark dataset
using the K-Nearest Neighbors (KNN) algorithm.

Author  : Generated for MaxiWoxi / DecodeLabs Project 2
Style   : PEP-8, Google-style docstrings
"""

# ── Standard library ────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets        import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ── Global style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "font.family": "DejaVu Sans"})

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1 · DATA INGESTION & PREPROCESSING                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_and_prepare(
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           StandardScaler, list[str]]:
    """Load the Iris dataset, shuffle it, split it, and scale features.

    Args:
        test_size (float): Fraction of samples reserved for testing.
            Defaults to 0.20 (20 %).
        random_state (int): Seed for reproducibility. Defaults to 42.

    Returns:
        A tuple of:
            X_train_sc  – Scaled training features (np.ndarray)
            X_test_sc   – Scaled testing  features (np.ndarray)
            y_train     – Training labels          (np.ndarray)
            y_test      – Testing  labels          (np.ndarray)
            scaler      – Fitted StandardScaler (reuse for new data)
            target_names– Human-readable class labels (list[str])
    """
    # ── Load ────────────────────────────────────────────────────────────────
    iris        = load_iris()
    X, y        = iris.data, iris.target
    target_names = list(iris.target_names)   # ['setosa', 'versicolor', 'virginica']

    # ── Shuffle → eliminates order bias present in the raw dataset ──────────
    # Without shuffling, the 80/20 split would put entire classes in one set.
    indices = np.arange(len(X))
    rng     = np.random.default_rng(seed=random_state)
    rng.shuffle(indices)
    X, y = X[indices], y[indices]

    # ── Train / Test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state,
        stratify     = y,   # ensures each split keeps the class ratio intact
    )

    # ── Feature Scaling (StandardScaler) ────────────────────────────────────
    # Raw features have different units (cm) and magnitudes.
    # KNN relies purely on Euclidean distance, so an unscaled large feature
    # (e.g., sepal length ~5–8 cm) would dominate a small one (petal width
    # ~0.1–2.5 cm), biasing every distance calculation.
    # StandardScaler transforms each feature → mean=0, std=1 (z-score).
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit ONLY on training data
    X_test_sc  = scaler.transform(X_test)        # apply same transform to test

    print("=" * 60)
    print("  DATA PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total samples    : {len(X)}")
    print(f"  Training samples : {len(X_train_sc)} ({int((1-test_size)*100)} %)")
    print(f"  Testing  samples : {len(X_test_sc)}  ({int(test_size*100)} %)")
    print(f"  Classes          : {target_names}")
    print(f"  Feature range (scaled) ≈ [{X_train_sc.min():.2f}, {X_train_sc.max():.2f}]")
    print("=" * 60)

    return X_train_sc, X_test_sc, y_train, y_test, scaler, target_names


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2 · ELBOW METHOD — FIND OPTIMAL K                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def find_optimal_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    k_max:   int = 30,
    save_fig: bool = True,
) -> int:
    """Calculate the error rate for each K and plot the Elbow curve.

    The "Elbow Method" helps choose K by plotting error rate against K.
    - Very low K (e.g. K=1): model memorises training data → overfitting,
      high variance, sensitive to noise.
    - Very high K (e.g. K=100): model ignores local structure → underfitting,
      high bias, overly simplistic.
    The optimal K sits at the "elbow" — where additional neighbours stop
    meaningfully reducing the error rate.

    Args:
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray): Training labels.
        X_test  (np.ndarray): Scaled testing features.
        y_test  (np.ndarray): Testing labels.
        k_max   (int): Upper bound for K search. Defaults to 30.
        save_fig (bool): Save elbow plot to disk. Defaults to True.

    Returns:
        int: The K value with the lowest error rate.
    """
    error_rates = []

    for k in range(1, k_max + 1):
        knn  = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds      = knn.predict(X_test)
        error_rate = np.mean(preds != y_test)   # fraction of wrong predictions
        error_rates.append(error_rate)

    optimal_k = int(np.argmin(error_rates)) + 1  # +1 because range starts at 1

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        range(1, k_max + 1), error_rates,
        color="steelblue", linestyle="--", marker="o",
        markerfacecolor="tomato", markersize=8, linewidth=2,
        label="Error Rate",
    )
    ax.axvline(
        x=optimal_k, color="green", linestyle=":", linewidth=2,
        label=f"Optimal K = {optimal_k}  (error={error_rates[optimal_k-1]:.3f})",
    )
    ax.set_xlabel("K (Number of Neighbours)", fontsize=12)
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_title("Elbow Method — KNN Error Rate vs. K Value", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(range(1, k_max + 1))
    fig.tight_layout()

    if save_fig:
        fig.savefig("/mnt/user-data/outputs/elbow_plot.png", bbox_inches="tight")
        print(f"\n  [Elbow Plot] saved → elbow_plot.png")

    plt.show()

    print(f"\n  Optimal K identified : {optimal_k}")
    print(f"  Corresponding error  : {error_rates[optimal_k-1]:.4f}")

    return optimal_k


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3 · MODEL TRAINING                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> KNeighborsClassifier:
    """Instantiate and train a KNN classifier with the given K.

    Args:
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray): Training labels.
        k (int): Number of neighbours to use.

    Returns:
        KNeighborsClassifier: Fitted model ready for inference.
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    print(f"\n  Model trained  : KNeighborsClassifier(n_neighbors={k})")
    return model


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4 · VALIDATION & METRICS                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def evaluate_model(
    model:        KNeighborsClassifier,
    X_test:       np.ndarray,
    y_test:       np.ndarray,
    target_names: list[str],
    save_fig:     bool = True,
) -> None:
    """Generate and display all evaluation metrics for the trained model.

    Why accuracy alone is misleading
    ---------------------------------
    On a balanced dataset like Iris, accuracy can look great even when the
    model fails systematically on one class.  The metrics below give a
    complete, honest picture:

    * Confusion Matrix  – counts of TP / FP / TN / FN per class.
    * Precision         – of all predicted positives, how many were truly positive?
                          (minimises false alarms)
    * Recall            – of all actual positives, how many did the model catch?
                          (minimises misses)
    * F1 Score          – harmonic mean of Precision and Recall; balanced metric
                          that penalises extremes and is robust to class imbalance.

    Args:
        model        (KNeighborsClassifier): Fitted KNN model.
        X_test       (np.ndarray): Scaled testing features.
        y_test       (np.ndarray): True labels.
        target_names (list[str]): Human-readable class names.
        save_fig     (bool): Save confusion matrix plot. Defaults to True.
    """
    y_pred = model.predict(X_test)

    # ── 4a · Raw accuracy (shown for reference; don't rely on it alone) ──────
    accuracy = np.mean(y_pred == y_test)
    print(f"\n  Raw Accuracy    : {accuracy:.4f}  ({accuracy*100:.1f} %)")
    print("  (Accuracy is a partial truth — see F1 scores below)\n")

    # ── 4b · Classification Report ───────────────────────────────────────────
    print("=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=target_names))

    # ── 4c · Confusion Matrix visualisation ─────────────────────────────────
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    # Annotate TP / FP / TN / FN interpretation in title
    ax.set_title(
        "Confusion Matrix\n"
        "(Diagonal = True Positives  |  Off-diagonal = Misclassifications)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    fig.tight_layout()

    if save_fig:
        fig.savefig("/mnt/user-data/outputs/confusion_matrix.png", bbox_inches="tight")
        print("  [Confusion Matrix] saved → confusion_matrix.png")

    plt.show()

    # ── 4d · Per-class F1 summary table (pandas for readability) ────────────
    report_dict = classification_report(
        y_test, y_pred,
        target_names=target_names,
        output_dict=True,
    )
    rows = {
        cls: {
            "Precision": report_dict[cls]["precision"],
            "Recall"   : report_dict[cls]["recall"],
            "F1-Score" : report_dict[cls]["f1-score"],
            "Support"  : int(report_dict[cls]["support"]),
        }
        for cls in target_names
    }
    df_metrics = pd.DataFrame(rows).T.round(4)
    print("\n  Per-class F1 Table:")
    print(df_metrics.to_string())
    print()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN ORCHESTRATOR                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_pipeline(
    test_size:    float = 0.20,
    random_state: int   = 42,
    k_max:        int   = 30,
) -> None:
    """End-to-end orchestrator for the DecodeLabs KNN Classification pipeline.

    Calls each stage in IPO order:
        Input  → load_and_prepare()
        Process→ find_optimal_k()  →  train_model()
        Output → evaluate_model()

    Args:
        test_size    (float): Fraction held out for testing. Default 0.20.
        random_state (int):   Global seed for reproducibility. Default 42.
        k_max        (int):   Max K value tested in Elbow Method. Default 30.
    """
    print("\n" + "▓" * 60)
    print("  DecodeLabs · Project 2 · KNN Classification Pipeline")
    print("▓" * 60)

    # ── Stage 1 · Input ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, target_names = load_and_prepare(
        test_size=test_size,
        random_state=random_state,
    )

    # ── Stage 2a · Elbow Method ───────────────────────────────────────────
    optimal_k = find_optimal_k(
        X_train, y_train,
        X_test,  y_test,
        k_max=k_max,
    )

    # ── Stage 2b · Train with optimal K ──────────────────────────────────
    model = train_model(X_train, y_train, k=optimal_k)

    # ── Stage 3 · Evaluate ────────────────────────────────────────────────
    evaluate_model(model, X_test, y_test, target_names)

    print("▓" * 60)
    print("  Pipeline complete.")
    print("▓" * 60 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline(
        test_size    = 0.20,   # ← configurable
        random_state = 42,     # ← configurable
        k_max        = 30,     # ← configurable
    )
