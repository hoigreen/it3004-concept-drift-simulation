from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils import ensure_dir, save_json
from src.io_kagglehub import get_dataset_folder
from src.loaders import load_dataset
from src.preprocess import basic_clean, maybe_sample, split_xy
from src.drift_simulation import make_drift_chunks
from src.modeling import build_lr_pipeline, fit, infer
from src.metrics import compute_all
from src.drift_detection import ks_test_drift, make_river_detector


def _concat_chunks(chunks, start: int, end: int):
    X = pd.concat([chunks[i].X for i in range(start, end)], ignore_index=True)
    y = np.concatenate([chunks[i].y for i in range(start, end)])
    return X, y


def _plot_lines(df: pd.DataFrame, out_dir: Path, metric: str):
    plt.figure()
    plt.plot(df["chunk"], df[f"{metric}_baseline"], marker="o")
    plt.plot(df["chunk"], df[f"{metric}_mitigated"], marker="o")
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
    window: int,
    detector: str,
    new_start: int,
    new_frac: float,
    benign_frac: float,
    seed: int,
):
    ensure_dir(out_dir / "figures")

    folder = get_dataset_folder(data_dir, dataset)
    loaded = load_dataset(folder, dataset)
    df = basic_clean(loaded.df)
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

    # Train baseline on chunk 1 only
    base_model = build_lr_pipeline(chunks[0].X)
    fit(base_model, chunks[0].X, chunks[0].y)

    # Mitigated model starts the same
    mit_model = build_lr_pipeline(chunks[0].X)
    fit(mit_model, chunks[0].X, chunks[0].y)

    river_det = None
    if detector != "ks":
        river_det = make_river_detector(detector)

    rows = []
    last_retrain_chunk = 1

    for t in range(2, n_chunks + 1):
        cur = chunks[t - 1]

        # Baseline metrics
        yb_pred, yb_proba = infer(base_model, cur.X)
        mb = compute_all(cur.y, yb_pred, yb_proba)

        # Mitigated metrics (pre-retrain)
        ym_pred, ym_proba = infer(mit_model, cur.X)
        mm = compute_all(cur.y, ym_pred, ym_proba)

        drift = False
        drift_info = ""

        if detector == "ks":
            # KS only on numeric columns
            num_cols = [
                c for c in cur.X.columns if pd.api.types.is_numeric_dtype(cur.X[c])]
            if len(num_cols) > 0:
                start = max(0, (t - 1) - window)  # chunks index
                X_ref, _ = _concat_chunks(chunks, start, t - 1)
                X_ref_num = X_ref[num_cols].to_numpy()
                X_cur_num = cur.X[num_cols].to_numpy()
                drift, ratio, drifted = ks_test_drift(
                    X_ref_num, X_cur_num, alpha=0.05, ratio_thr=0.30)
                drift_info = f"ks_ratio={ratio:.3f}, drifted_features={drifted}/{len(num_cols)}"
            else:
                drift = False
                drift_info = "ks_skipped(no_numeric_cols)"
        else:
            # river detector on error stream of mitigated model
            for yt, yp in zip(cur.y, ym_pred):
                river_det.update(int(yt != yp))
                if getattr(river_det, "drift_detected", False):
                    drift = True
                    break
            drift_info = f"river={detector}, drift_detected={drift}"

        retrained = False
        if drift:
            # retrain using sliding window of previous chunks (before current)
            start = max(0, (t - 1) - window)
            Xw, yw = _concat_chunks(chunks, start, t - 1)

            mit_model = build_lr_pipeline(Xw)
            fit(mit_model, Xw, yw)

            retrained = True
            last_retrain_chunk = t - 1

            # re-eval after retrain on current chunk (to show recovery)
            ym_pred2, ym_proba2 = infer(mit_model, cur.X)
            mm = compute_all(cur.y, ym_pred2, ym_proba2)

        row = {
            "chunk": t,
            "drift": drift,
            "drift_info": drift_info,
            "retrained": retrained,
            "last_retrain_chunk": last_retrain_chunk,
            **{f"{k}_baseline": v for k, v in mb.items()},
            **{f"{k}_mitigated": v for k, v in mm.items()},
        }
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics_by_chunk.csv", index=False)

    # plots
    for m in ["recall_attack", "f1_attack", "precision_attack", "pr_auc", "roc_auc", "accuracy"]:
        if f"{m}_baseline" in metrics_df.columns:
            _plot_lines(metrics_df, out_dir, m)

    print(f"[EXP] Saved to: {out_dir}")
