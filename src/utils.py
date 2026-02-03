from __future__ import annotations
from pathlib import Path
import json


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False,
                    indent=2), encoding="utf-8")
