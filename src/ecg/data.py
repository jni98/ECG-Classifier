import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer

VAL_FOLD = 9
TEST_FOLD = 10
LABELS = ["CD", "HYP", "MI", "NORM", "STTC"]


def get_superclasses(codes: dict, statements: pd.DataFrame) -> list[str]:
    """Map SCP codes of one ECG to its diagnostic superclasses.

    Args:
        codes (dict): A dictionary of SCP codes.
        statements (pd.DataFrame): A DataFrame containing diagnostic statements.

    Returns:
        list[str]: A list of diagnostic superclasses.
    """
    superclasses = []
    for code in codes:
        if code in statements.index:
            superclasses.append(statements.loc[code, "diagnostic_class"])

    return sorted(set(superclasses))


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load metadata from CSV file and replace age 300 with 90.

    Args:
        path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded metadata.
    """
    df = pd.read_csv(path, index_col="ecg_id")
    df["age"] = df["age"].replace(300, 90)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def load_statements(path: str | Path) -> pd.DataFrame:
    """Load diagnostic statements from CSV file.

    Args:
        path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded diagnostic statements.
    """
    statements = pd.read_csv(path, index_col=0)
    statements = statements[statements["diagnostic"] == 1]
    return statements


def load_signals(df: pd.DataFrame, data_dir: str | Path) -> np.ndarray:
    """Load raw ECG signals from WFDB files.

    Args:
        df (pd.DataFrame): A DataFrame containing the ECG data.
        data_dir (str): The directory containing the WFDB files.

    Returns:
        np.ndarray: An array of raw ECG signals.
    """
    signals = []
    for filename in df["filename_lr"]:
        signal, _ = wfdb.rdsamp(str(Path(data_dir) / filename))
        signals.append(signal)
    return np.array(signals, dtype=np.float32)


def split_by_fold(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into training, validation and testing sets based on stratified folds.

    Args:
        df (pd.DataFrame): A DataFrame containing the ECG data.

    Returns:
        tuple: A tuple containing the training and testing DataFrames.
    """

    train_df = df[~df["strat_fold"].isin([VAL_FOLD, TEST_FOLD])]
    test_df = df[df["strat_fold"] == TEST_FOLD]
    val_df = df[df["strat_fold"] == VAL_FOLD]

    return train_df, val_df, test_df


def load_or_cache(cache: Path, data: pd.DataFrame, data_dir: str | Path) -> np.ndarray:
    """Load signals from cache if available, otherwise load from data and cache it.

    Args:
        cache (Path): The path to the cache file.
        data (pd.DataFrame): A DataFrame containing the ECG data.
        data_dir (str): The directory containing the WFDB files.

    Returns:
        np.ndarray: An array of raw ECG signals.
    """

    if cache.exists():
        X = np.load(cache)
    else:
        X = load_signals(data, data_dir)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, X)

    return X


def add_label_columns(df: pd.DataFrame, classes: list[str] = LABELS) -> pd.DataFrame:
    """Add one 0/1 column per class from the superclasses lists (multi-hot).

    Args:
        df (pd.DataFrame): DataFrame with a superclasses column of lists.
        classes (list[str]): Class names, in the column order to create.

    Returns:
        pd.DataFrame: A copy of df with one extra column per class.
    """
    label_df = df.copy()
    mlb = MultiLabelBinarizer(classes=classes)
    encoded = mlb.fit_transform(label_df["superclasses"])
    new_columns = pd.DataFrame(encoded, columns=classes, index=label_df.index)
    return pd.concat([label_df, new_columns], axis=1)
