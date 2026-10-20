import pandas as pd
import pytest

from ecg.data import TEST_FOLD, VAL_FOLD, get_superclasses, split_by_fold


@pytest.fixture
def statements():
    return pd.DataFrame(
        {"diagnostic_class": ["MI", "MI", "STTC"]},
        index=["IMI", "ASMI", "NDT"],
    )


def test_get_superclasses_empty_class(statements):
    assert get_superclasses({"SR": 0}, statements) == []


def test_get_superclasses_sr_ignored(statements):
    assert get_superclasses({"IMI": 100, "SR": 0}, statements) == ["MI"]


def test_get_superclasses_duplicate_superclass(statements):
    assert get_superclasses({"IMI": 100, "ASMI": 50}, statements) == ["MI"]


def test_get_superclasses_multiple_superclasses(statements):
    assert get_superclasses({"IMI": 100, "ASMI": 50, "NDT": 50}, statements) == [
        "MI",
        "STTC",
    ]


@pytest.fixture
def folds_df() -> pd.DataFrame:
    return pd.DataFrame({"strat_fold": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 10]})


def test_split_by_fold_training(folds_df):
    train_df, _, _ = split_by_fold(folds_df)

    assert len(train_df) == 8, f"Expected 8 training samples, but got {len(train_df)}"

    folds = set(train_df["strat_fold"])
    assert VAL_FOLD not in folds, (
        f"Validation fold {VAL_FOLD} should not be in training folds {folds}"
    )
    assert TEST_FOLD not in folds, (
        f"Test fold {TEST_FOLD} should not be in training folds {folds}"
    )


def test_split_by_fold_val(folds_df):
    _, val_df, _ = split_by_fold(folds_df)

    assert len(val_df) == 2, f"Expected 2 validation sample, but got {len(val_df)}"
    assert set(val_df["strat_fold"]) == {VAL_FOLD}, (
        f"Expected validation fold to be {VAL_FOLD}, "
        f"but got {set(val_df['strat_fold'])}"
    )


def test_split_by_fold_test(folds_df):
    _, _, test_df = split_by_fold(folds_df)

    assert len(test_df) == 2, f"Expected 2 test sample, but got {len(test_df)}"
    assert set(test_df["strat_fold"]) == {TEST_FOLD}, (
        f"Expected test fold to be {TEST_FOLD}, but got {set(test_df['strat_fold'])}"
    )


def test_split_by_fold_combined(folds_df):
    train_df, val_df, test_df = split_by_fold(folds_df)

    assert sum(len(df) for df in [train_df, val_df, test_df]) == len(folds_df), (
        "Total number of samples after splitting does not match original DataFrame"
    )

    # Check that all samples are accounted for
    all_samples = set()
    for df in [train_df, val_df, test_df]:
        all_samples.update(df.index)

    assert all_samples == set(folds_df.index), (
        "Not all samples are accounted for in the splits"
    )
