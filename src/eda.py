from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils import ensure_dir, save_json
from src.io_kagglehub import get_dataset_folder
from src.loaders import load_dataset
from src.preprocess import basic_clean, maybe_sample


def run_eda(dataset: str, data_dir: Path, out_dir: Path, sample: int = 0):
    ensure_dir(out_dir / "figures")

    folder = get_dataset_folder(data_dir, dataset)
    loaded = load_dataset(folder, dataset)
    df = basic_clean(loaded.df)
    df = maybe_sample(df, sample, seed=42)

    # Basic stats
    info = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "label_col": loaded.label_col,
        "attack_type_col": loaded.attack_type_col,
        "label_distribution": df[loaded.label_col].value_counts(dropna=False).to_dict(),
        "top_attack_types": df[loaded.attack_type_col].astype(str).value_counts().head(15).to_dict(),
        "missing_ratio_top10": (df.isna().mean().sort_values(ascending=False).head(10)).to_dict(),
    }
    save_json(info, out_dir / "eda_summary.json")

    # Class distribution plot
    plt.figure()
    df[loaded.label_col].value_counts().plot(kind="bar")
    plt.title(f"{dataset} - label distribution (0 benign, 1 attack)")
    plt.xlabel("label")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "label_distribution.png")
    plt.close()

    # Attack type distribution (top 15)
    plt.figure()
    df[loaded.attack_type_col].astype(
        str).value_counts().head(15).plot(kind="bar")
    plt.title(f"{dataset} - top 15 attack types")
    plt.xlabel("attack type")
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "top_attack_types.png")
    plt.close()

    # Numeric correlation heatmap (top 20 numeric by variance)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(
        df[c]) and c != loaded.label_col]
    if len(num_cols) >= 5:
        sub = df[num_cols].copy()
        # pick top 20 by variance
        vars_ = sub.var(numeric_only=True).sort_values(ascending=False)
        pick = vars_.head(min(20, len(vars_))).index.tolist()
        corr = sub[pick].corr()

        plt.figure(figsize=(10, 8))
        plt.imshow(corr, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(pick)), pick, rotation=90)
        plt.yticks(range(len(pick)), pick)
        plt.title(f"{dataset} - correlation (top variance numeric features)")
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "corr_top_numeric.png")
        plt.close()

        corr.to_csv(out_dir / "corr_top_numeric.csv", index=True)

    # Save a small describe table
    desc = df.select_dtypes(include=[np.number]).describe().T
    desc.to_csv(out_dir / "numeric_describe.csv", index=True)

    print(f"[EDA] Saved to: {out_dir}")
