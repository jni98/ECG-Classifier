from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from ecg.data import (
    get_superclasses,
    load_metadata,
    load_or_cache,
    load_statements,
    split_by_fold,
)
from ecg.evaluate import (
    confusion_counts,
    metrics_from_counts,
    plot_reliability_diagram,
    plot_roc_pr_curve,
    threshold_for_sensitivity,
)
from ecg.features import extract_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_FILENAME = "ptbxl_database.csv"
STATEMENTS_FILENAME = "scp_statements.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "figures"


def evaluate_baseline(
    data_dir: Path = DATA_DIR,
    data_filename: str = DATA_FILENAME,
    statements_filename: str = STATEMENTS_FILENAME,
    data_dir_processed: Path = DATA_DIR_PROCESSED,
    report_dir: Path = REPORT_DIR,
) -> None:
    """Train the baseline on folds 1-8 and evaluate it on validation fold 9.

    Prints the macro and per-class AUROC and, per class, the threshold that
    reaches 90 % sensitivity with its specificity and precision. Saves the ROC
    and PR curves to `report_dir`. Test fold 10 is not used.

    Args:
        data_dir: Raw dataset directory.
        data_filename: Database filename.
        statements_filename: SCP statements filename.
        data_dir_processed: Directory for cached signal arrays.
        report_dir: Directory for report figures.
    """

    df = load_metadata(data_dir / data_filename)
    statements = load_statements(data_dir / statements_filename)

    df["superclasses"] = df["scp_codes"].apply(
        lambda codes: get_superclasses(codes, statements)
    )

    # ECGs without a diagnostic superclass have no usable label
    len_classes = df["superclasses"].apply(len)
    df = df[len_classes > 0]

    # Fold 10 is deliberately not used; it is reserved for the final report
    train_df, val_df, _ = split_by_fold(df)

    # Fixed class order so the columns of y_val and y_score always match
    mlb = MultiLabelBinarizer(classes=["CD", "HYP", "MI", "NORM", "STTC"])
    y_train = mlb.fit_transform(train_df["superclasses"])
    y_val = mlb.transform(val_df["superclasses"])

    # Cached as .npy so later runs skip reading the raw WFDB files
    X_train = load_or_cache(data_dir_processed / "X_train.npy", train_df, data_dir)
    X_val = load_or_cache(data_dir_processed / "X_val.npy", val_df, data_dir)

    features_train = extract_features(X_train)
    features_val = extract_features(X_val)

    # Scaler inside the pipeline so it is fit on the training folds only (no leakage).
    # class_weight="balanced" because the superclasses are imbalanced (HYP ~12 %).
    model = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(LogisticRegression(class_weight="balanced", max_iter=1000)),
    )
    model.fit(features_train, y_train)

    y_score = model.predict_proba(features_val)

    roc_auc_classes = np.zeros(len(mlb.classes_))
    for i in range(len(mlb.classes_)):
        roc_auc_classes[i] = roc_auc_score(y_val[:, i], y_score[:, i])

    macro_auroc = roc_auc_score(y_val, y_score, average="macro")
    class_auroc = pd.Series(roc_auc_classes, index=mlb.classes_)
    print(f"Macro AUROC score (fold 9): {macro_auroc:.3f}")
    print(f"AUROC score per class (fold 9): \n{class_auroc}")

    # Thresholds are chosen on fold 9 only, never on the test fold
    thresholds = np.zeros(len(mlb.classes_))
    metrics = []
    brier_score = []
    for i in range(len(mlb.classes_)):
        thresholds[i] = threshold_for_sensitivity(
            y_val[:, i], y_score[:, i], min_sensitivity=0.9
        )
        tn, fp, fn, tp = confusion_counts(y_val[:, i], y_score[:, i], thresholds[i])
        metrics.append(metrics_from_counts(tn, fp, fn, tp))
        brier_score.append(brier_score_loss(y_val[:, i], y_score[:, i]))

    metrics_df = pd.DataFrame(metrics, index=mlb.classes_)
    metrics_df["thresholds"] = thresholds
    metrics_df["brier score"] = brier_score
    print(f"Metrics for a given threshold: \n{metrics_df.round(3)}")

    fig = plot_roc_pr_curve(y_val, y_score, mlb.classes_)
    fig_path = report_dir / "roc_pr_curves.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"ROC and PR curves were saved to {fig_path}")

    fig = plot_reliability_diagram(y_val, y_score, mlb.classes_)
    fig_path = report_dir / "reliability_diagram.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Reliability diagram was saved to {fig_path}")


if __name__ == "__main__":
    evaluate_baseline()
