import math
from collections.abc import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def metrics_from_counts(tn, fp, fn, tp) -> dict:
    """Compute metrics from counts of tn, fp, fn, tp.

    Args:
        tn (int): Count of true negatives.
        fp (int): Count of false positives.
        fn (int): Count of false negatives.
        tp (int): Count of true positives.
    Returns:
        dict: A dictionary containing the computed metrics.
    """

    if (tp + fn) == 0:
        sensitivity = None
    else:
        sensitivity = tp / (tp + fn)

    if (tn + fp) == 0:
        specificity = None
    else:
        specificity = tn / (tn + fp)

    if (tp + fp) == 0:
        precision = None
    else:
        precision = tp / (tp + fp)

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
    }


def confusion_counts(y_true, y_score, threshold) -> tuple[int, int, int, int]:
    """Count classification numbers to return confusion counts.

    Args:
        y_true: True class labels
        y_score: Class probabilities returned by the model
        threshold: class threshold

    Returns:
        tuple: Confusion counts in the order (tn, fp, fn, tp), as in sklearn.
    """

    y_pred = y_score >= threshold

    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tp = ((y_pred == 1) & (y_true == 1)).sum()

    return tn, fp, fn, tp


def threshold_for_sensitivity(y_true, y_score, min_sensitivity=0.9) -> float:
    """Return the highest threshold that reaches the given sensitivity.

    Args:
        y_true: Ground truth labels (0/1) for one class.
        y_score: Predicted probabilities for that class.
        min_sensitivity: Minimum sensitivity to reach, in (0, 1].

    Returns:
        float: Highest threshold with sensitivity >= min_sensitivity.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if not 0 < min_sensitivity <= 1:
        raise ValueError("min_sensitivity must be in (0, 1].")

    # Scores of the positive samples, sorted from high to low
    positive_scores = np.sort(y_score[y_true == 1])[::-1]
    if len(positive_scores) == 0:
        raise ValueError("y_true contains no positive samples.")

    # We need at least k true positives (round up, 13.5 -> 14)
    k = math.ceil(min_sensitivity * len(positive_scores))

    # With score >= threshold, exactly the k highest positives are detected
    return float(positive_scores[k - 1])


def plot_roc_pr_curve(
    y_true_all: np.ndarray, y_score_all: np.ndarray, class_names: Sequence[str]
) -> Figure:
    """
    Generates a figure with two subplots for the PR- and ROC-Curve
    Args:
        y_true_all: Ground truth labels (0/1) for all classes
        y_score_all: Predicted probabilities for all classes
        class_names: Class names for all given classes

    Returns:
        fig: One figure including subplots for PR- and ROC-Curve
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, name in enumerate(class_names):
        PrecisionRecallDisplay.from_predictions(
            y_true_all[:, i], y_score_all[:, i], name=name, ax=axes[0]
        )
        RocCurveDisplay.from_predictions(
            y_true_all[:, i], y_score_all[:, i], name=name, ax=axes[1]
        )

        axes[0].set_title("Precision-Recall Curve")
        axes[0].set_title("Receiver Operating Characteristic Curve")

    return fig
