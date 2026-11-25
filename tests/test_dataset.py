import numpy as np
import pandas as pd
import torch
import wfdb

from ecg.dataset import PTBXLDataset


def test_dataset_returns_correct_shapes(tmp_path):
    # 1) Arrange: write 2 fake ECGs to a temporary folder
    for name in ["rec1", "rec2"]:
        signal = np.random.randn(1000, 12)
        wfdb.wrsamp(
            name,
            fs=100,
            units=["mV"] * 12,
            sig_name=[f"ch{i}" for i in range(12)],
            p_signal=signal,
            fmt=["16"] * 12,
            write_dir=str(tmp_path),
        )
    df = pd.DataFrame(
        {
            "filename_lr": ["rec1", "rec2"],
            "CD": [0, 0],
            "HYP": [1, 1],
            "MI": [1, 0],
            "NORM": [1, 1],
            "STTC": [0, 1],
        }
    )

    # 2) Act
    ds = PTBXLDataset(df, tmp_path)
    signal, label = ds[0]

    # 3) Assert
    assert len(ds) == 2, f"Expected length 2, but got {len(ds)}"
    assert signal.shape == (12, 1000), (
        f"Expected signal shape (12,1000), but got {signal.shape}"
    )
    assert label.shape == (5,), f"Expected label shape (5,), but got {label.shape}"
    assert signal.dtype == torch.float32, (
        f"Expected signal to have type torch.float32, but got {signal.dtype}"
    )
    assert label.dtype == torch.float32, (
        f"Expected label to have type torch.float32, but got {label.dtype}"
    )
