import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

VAL_FOLD = 9
TEST_FOLD = 10


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


def load_metadata(path: str) -> pd.DataFrame:
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


def load_statements(path: str) -> pd.DataFrame:
    """Load diagnostic statements from CSV file.

    Args:
        path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded diagnostic statements.
    """
    statements = pd.read_csv(path, index_col=0)
    statements = statements[statements["diagnostic"] == 1]
    return statements


def load_signals(df: pd.DataFrame, data_dir: str) -> np.ndarray:
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
    return np.array(signals)


def split_by_fold(df) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into training and testing sets based on stratified folds.

    Args:
        df (pd.DataFrame): A DataFrame containing the ECG data.

    Returns:
        tuple: A tuple containing the training and testing DataFrames.
    """

    train_df = df[~df["strat_fold"].isin([VAL_FOLD, TEST_FOLD])]
    test_df = df[df["strat_fold"] == TEST_FOLD]
    val_df = df[df["strat_fold"] == VAL_FOLD]

    return train_df, val_df, test_df
