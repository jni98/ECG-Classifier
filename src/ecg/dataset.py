"""PyTorch Dataset for PTB-XL ECG records."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

LABELS = ["CD", "HYP", "MI", "NORM", "STTC"]


class PTBXLDataset(Dataset):
    """Loads one PTB-XL record per index from disk (lazy loading).

    Parameters
    ----------
    data : pandas.DataFrame
        One row per ECG. Needs the column ``filename_lr`` and the label columns.
    directory : str or Path
        Folder that ``filename_lr`` is relative to (e.g. ``data/raw``).
    label_cols : list of str
        Names of the label columns, in a fixed order.
    """

    def __init__(
        self, data: pd.DataFrame, directory: str | Path, label_cols: list[str] = LABELS
    ) -> None:
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.directory = Path(directory)
        self.label_cols = label_cols

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (signal, label) with shapes (12, 1000) and (5,), both float32."""
        row = self.data.iloc[index]

        # rdsamp returns (signal, metadata); signal has shape (time, leads)
        signal, _ = wfdb.rdsamp(str(self.directory / row["filename_lr"]))
        # Conv1d expects (channels, time), so transpose to (leads, time)
        ecg = torch.tensor(signal.T, dtype=torch.float32)

        label = torch.from_numpy(row[self.label_cols].to_numpy(dtype=np.float32))
        return ecg, label
