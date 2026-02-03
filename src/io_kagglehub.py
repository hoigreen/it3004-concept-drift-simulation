from __future__ import annotations
from pathlib import Path
import kagglehub

DATASET_SLUGS = {
    "nslkdd": "hassan06/nslkdd",
    "unsw": "harshwardhanbhangale/unsw-complete-dataset",
    "cicids2018": "dhoogla/csecicids2018",
}


def get_dataset_folder(data_dir: Path, dataset: str) -> Path:
    """
    Returns a local dataset folder.
    Priority:
    1) Use an existing local folder if provided (data_dir or data_dir/dataset).
    2) Otherwise download via kagglehub and return its cache path.
    """
    if data_dir is not None and data_dir.exists():
        direct = data_dir / dataset
        if direct.exists():
            return direct
        if any(data_dir.rglob("*.csv")) or any(data_dir.rglob("*.TXT")):
            return data_dir

    slug = DATASET_SLUGS[dataset]
    path = kagglehub.dataset_download(slug)
    return Path(path)
