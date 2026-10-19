"""Download PTB-XL (100 Hz only) from PhysioNet's public S3 mirror."""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
S3_URI = "s3://physionet-open/ptb-xl/1.0.3/"


def download_ptbxl(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "tool",
        "run",
        "--from",
        "awscli",
        "aws",
        "s3",
        "sync",
        "--no-sign-request",
        S3_URI,
        str(data_dir),
        "--exclude",
        "records500/*",
    ]
    print(f"Syncing PTB-XL into {data_dir} ...")
    subprocess.run(cmd, check=True)
    print("Done.")


if __name__ == "__main__":
    download_ptbxl()
