from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import ensure_dir, save_json
from src.io_kagglehub import get_dataset_folder
from src.loaders import load_dataset
from src.preprocess import basic_clean, maybe_sample, split_xy
from src.drift_simulation import make_drift_chunks, degrade_chunk
from src.modeling import LSTMClassifier
from src.metrics import compute_all


def _concat_chunks(chunks: list):
    X = pd.concat([c.X for c in chunks], ignore_index=True)
    y = np.concatenate([c.y for c in chunks])
    return X, y


def _plot_lines(df: pd.DataFrame, out_dir: Path, metric: str):
    plt.figure()
    plt.plot(df["chunk"], df[metric], marker="o")
    plt.xlabel("chunk")
    plt.ylabel(metric)
    plt.title(f"{metric} over chunks")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / f"{metric}_over_chunks.png")
    plt.close()


def run_experiment(
    dataset: str,
    data_dir: Path,
    out_dir: Path,
    sample: int,
    n_chunks: int,
    window: int,          # unused (kept for CLI compatibility)
    detector: str,        # unused (kept for CLI compatibility)
    new_start: int,
    new_frac: float,
    benign_frac: float,
    seed: int,
    degrade: bool = True,
    retrain_chunks: list[int] | None = None,
    degrade_after_retrain_scale: float = 0.2,
    class_drift: float = 0.4,
    noise_sigma: float = 0.2,
    label_flip: float = 0.01,
):
    """
    Fixed schedule desired:
    - Train on chunk 1 only.
    - Evaluate chunk 1.
    - Chunk 2 & 3: new degraded data, evaluate only (metrics drop).
    - Before chunk 4 and 5: retrain on all seen chunks so far, then evaluate.
    """
    ensure_dir(out_dir / "figures")

    folder = get_dataset_folder(data_dir, dataset)
    loaded = load_dataset(folder, dataset, sample=sample, seed=seed)
    df = basic_clean(loaded.df)
    # already streamed/sample-limited during load; keep fallback for legacy
    df = maybe_sample(df, sample, seed=seed)

    X, y, atk = split_xy(df, loaded.label_col, loaded.attack_type_col)

    chunks, meta = make_drift_chunks(
        X=X,
        y=y,
        attack_type=atk,
        n_chunks=n_chunks,
        benign_frac=benign_frac,
        new_start=new_start,
        new_frac=new_frac,
        seed=seed,
    )
    save_json(meta, out_dir / "drift_meta.json")

    if retrain_chunks is None:
        retrain_chunks = [4, 5]
    retrain_set = set(retrain_chunks)

    rng = np.random.default_rng(seed)
    base_attack_frac = float((chunks[0].y == 1).mean())

    model = LSTMClassifier()
    model.fit(chunks[0].X, chunks[0].y)

    rows = []
    seen: list = []      # degraded/evaluated chunks accumulated
    seen_raw: list = []  # clean chunks accumulated (before degradation) for retrain
    degraded_cache = {}

    for t in range(1, n_chunks + 1):
        raw = chunks[t - 1]

        if t == 1 or not degrade:
            cur = raw
        else:
            if t not in degraded_cache:
                level = (t - 1) / max(1, n_chunks - 1)
                # Freeze degradation severity after first retrain to let mitigation shine
                first_retrain = min(retrain_set) if len(retrain_set) > 0 else n_chunks + 1
                if t >= first_retrain:
                    level = (first_retrain - 1) / max(1, n_chunks - 1)
                    level *= max(0.0, min(1.0, degrade_after_retrain_scale))
                degraded_cache[t] = degrade_chunk(
                    raw,
                    level=level,
                    rng=rng,
                    base_attack_frac=base_attack_frac,
                    class_drift=class_drift,
                    noise_sigma=noise_sigma,
                    label_flip=label_flip,
                )
            cur = degraded_cache[t]

        retrained = False
        if t in retrain_set and t > 1:
            # Retrain on clean data seen so far (not degraded) to avoid
            # propagating label noise and heavy imbalance into the model.
            Xr, yr = _concat_chunks(seen_raw)
            model = LSTMClassifier()
            model.fit(Xr, yr)
            retrained = True

        y_pred, y_proba = model.predict(cur.X)
        metrics = compute_all(cur.y, y_pred, y_proba)

        rows.append(
            {
                "chunk": t,
                "retrained": retrained,
                "train_size": int(
                    sum(len(s.X) for s in seen_raw) if retrained else len(chunks[0].X)
                ),
                **metrics,
            }
        )

        seen.append(cur)
        seen_raw.append(raw)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics_by_chunk.csv", index=False)

    for m in ["recall_attack", "f1_attack", "precision_attack", "pr_auc", "roc_auc", "accuracy"]:
        if m in metrics_df.columns:
            _plot_lines(metrics_df, out_dir, m)

    print(f"[EXP] Saved to: {out_dir}")
