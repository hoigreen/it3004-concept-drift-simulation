from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    name: str


DATASETS = {
    "nslkdd": DatasetInfo(name="NSL-KDD"),
    "unsw": DatasetInfo(name="UNSW-NB15"),
    "cicids2018": DatasetInfo(name="CSE-CIC-IDS2018"),
}
