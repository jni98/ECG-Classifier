import numpy as np
import pytest

from ecg.evaluate import (
    confusion_counts,
    metrics_from_counts,
    threshold_for_sensitivity,
)


def test_metrics_from_counts():
    metrics = metrics_from_counts(tn=70, fp=10, fn=5, tp=15)

    assert metrics["sensitivity"] == 0.75, (
        f"Expected a sensitivity of 0.75 but got {metrics['sensitivity']} instead"
    )
    assert metrics["specificity"] == 0.875, (
        f"Expected a specificity of 0.875 but got {metrics['specificity']} instead"
    )
    assert metrics["precision"] == 0.6, (
        f"Expected a precision of 0.6 but got {metrics['precision']} instead"
    )


def test_confusion_counts():
    y_score = np.array([0.1, 0.2, 0.7, 0.05, 0.6])
    y_true = np.array([0, 1, 1, 0, 1])
    threshold = 0.55

    tn, fp, fn, tp = confusion_counts(
        y_true=y_true, y_score=y_score, threshold=threshold
    )

    assert tn == 2, f"Expected a true negative value of 2 but got {tn} instead"
    assert fp == 0, f"Expected a false positive value of 0 but got {fp} instead"
    assert fn == 1, f"Expected a false negative value of 1 but got {fn} instead"
    assert tp == 2, f"Expected a true positive value of 2 but got {tp} instead"


def test_threshold_for_sensitivity():
    y_score = np.array([0.9, 0.7, 0.4, 0.2, 0.1])
    y_true = np.array([1, 1, 1, 1, 0])

    threshold = threshold_for_sensitivity(y_true, y_score, min_sensitivity=0.75)

    assert threshold == 0.4, f"Expected a threshold of 0.4 but got {threshold} instead"


def test_threshold_for_sensitivity_():
    y_score = np.array([0.9, 0.7, 0.4, 0.2, 0.1])
    y_true = np.array([1, 1, 1, 1, 0])

    threshold = threshold_for_sensitivity(y_true, y_score, min_sensitivity=0.2)

    assert threshold == 0.9, f"Expected a threshold of 0.9 but got {threshold} instead"


def test_threshold_for_sensitivity_no_positives():
    y_score = np.array([0.9, 0.7, 0.4])
    y_true = np.array([0, 0, 0])

    with pytest.raises(ValueError):
        threshold_for_sensitivity(y_true, y_score)
