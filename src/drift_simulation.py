from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Chunk:
    X: pd.DataFrame
    y: np.ndarray
    attack_type: np.ndarray


def make_drift_chunks(
    X: pd.DataFrame,
    y: np.ndarray,
    attack_type: np.ndarray,
    n_chunks: int,
    benign_frac: float,
    new_start: int,
    new_frac: float,
    seed: int = 42,
):
    """
    Build a simple non-stationary stream by varying the benign/attack ratio
    over chunks.

    - Each chunk has approximately the same size.
    - `benign_frac` controls the benign share in early chunks.
    - `new_start` controls **when** drift starts (1-based chunk index).
      If 0, drift starts around the middle chunk.
    - `new_frac` is reinterpreted as a **drift strength** in [0, 1]:
      larger values make later chunks more attack-heavy (lower benign ratio).

    No special "Type-C" notion of new-vs-known attack categories is used; we
    only change the overall class balance over time for a simpler drift model.
    """
    rng = np.random.default_rng(seed)

    df = X.copy()
    df["_y"] = y
    df["_atk"] = attack_type

    benign = df[df["_y"] == 0]
    attacks = df[df["_y"] == 1]

    total = len(df)
    chunk_size = max(1, total // n_chunks)

    # Decide drift start chunk (1-based)
    if new_start is None or new_start <= 0:
        # "auto mid": start after the first half
        start_chunk = max(1, (n_chunks // 2) + 1)
    else:
        start_chunk = new_start
    start_chunk = max(1, min(n_chunks, start_chunk))

    def sample_df(d: pd.DataFrame, n: int) -> pd.DataFrame:
        if n <= 0:
            return d.iloc[:0]
        n = min(n, len(d))
        idx = rng.choice(len(d), size=n, replace=False)
        return d.iloc[idx]

    chunks: list[Chunk] = []
    for i in range(1, n_chunks + 1):
        # Normalised position in drift window [0, 1].
        # Before start_chunk => no drift (t = 0).
        if i < start_chunk:
            t = 0.0
        else:
            denom = max(1, n_chunks - start_chunk)
            t = (i - start_chunk) / denom

        # Benign fraction decreases over time depending on drift strength.
        drift_strength = max(0.0, min(1.0, new_frac))
        benign_frac_i = benign_frac * (1.0 - drift_strength * t)

        n_benign = int(chunk_size * benign_frac_i)
        n_attack = max(1, chunk_size - n_benign)

        c = pd.concat(
            [
                sample_df(benign, n_benign),
                sample_df(attacks, n_attack),
            ],
            ignore_index=True,
        )

        c = c.sample(frac=1, random_state=int(
            rng.integers(0, 1_000_000))).reset_index(drop=True)

        Xc = c.drop(columns=["_y", "_atk"])
        yc = c["_y"].astype(int).to_numpy()
        atkc = c["_atk"].astype(str).to_numpy()

        chunks.append(Chunk(X=Xc, y=yc, attack_type=atkc))

    meta = {
        "n_chunks": n_chunks,
        "chunk_size_approx": chunk_size,
        "drift_start_chunk": start_chunk,
        "benign_frac_start": benign_frac,
        "benign_frac_end": max(
            0.0,
            benign_frac
            * (1.0 - (max(0.0, min(1.0, new_frac)) if start_chunk < n_chunks else 0.0)),
        ),
        "drift_strength": max(0.0, min(1.0, new_frac)),
    }
    return chunks, meta


# ---------------------------------------------------------------------------
# Additional degradation helpers (class imbalance, noise, label flips)
# ---------------------------------------------------------------------------


def degrade_chunk(
    chunk: Chunk,
    level: float,
    rng: np.random.Generator,
    base_attack_frac: float,
    class_drift: float = 0.5,
    noise_sigma: float = 0.5,
    label_flip: float = 0.02,
):
    """
    Return a *new* Chunk with progressive degradation applied.

    Args:
        level: in [0,1], controls severity; increases with later chunks.
        base_attack_frac: baseline attack share in the (test) set; we shrink it
            over time to simulate growing benign dominance.
        class_drift: scaling for how fast attack fraction decays.
        noise_sigma: std of Gaussian noise injected into numeric columns.
        label_flip: fraction of labels flipped (scaled by level).
    """

    X = chunk.X.copy()
    y = chunk.y.copy()

    # ----- class imbalance -----
    attack_frac_t = max(0.01, base_attack_frac * (1.0 - class_drift * level))

    idx_attack = np.where(y == 1)[0]
    idx_benign = np.where(y == 0)[0]
    n_total = len(y)
    n_attack = max(1, int(n_total * attack_frac_t))
    n_benign = max(1, n_total - n_attack)

    def _sample(idxs, n):
        if len(idxs) == 0:
            return np.array([], dtype=int)
        replace = len(idxs) < n
        return rng.choice(idxs, size=n, replace=replace)

    sel_idx = np.concatenate([
        _sample(idx_attack, n_attack),
        _sample(idx_benign, n_benign)
    ])
    rng.shuffle(sel_idx)
    X = X.iloc[sel_idx].reset_index(drop=True)
    y = y[sel_idx]

    # ----- feature noise / concept drift -----
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if num_cols:
        scale = noise_sigma * level
        noise = rng.normal(loc=0.0, scale=scale, size=(len(X), len(num_cols)))
        X[num_cols] = X[num_cols].to_numpy() + noise

        # simple concept drift: shift first numeric feature mean over time
        drift_col = num_cols[0]
        X[drift_col] = X[drift_col] + level

    # ----- label noise -----
    flip_rate = label_flip * level
    if flip_rate > 0:
        mask = rng.random(len(y)) < flip_rate
        y[mask] = 1 - y[mask]

    return Chunk(X=X, y=y, attack_type=np.array(chunk.attack_type)[sel_idx])
