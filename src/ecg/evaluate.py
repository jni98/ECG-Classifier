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

    y_pred = y_score > threshold

    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tp = ((y_pred == 1) & (y_true == 1)).sum()

    return tn, fp, fn, tp
