from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency guard
    pq = None


@dataclass
class Loaded:
    df: pd.DataFrame
    label_col: str          # binary label 0/1
    attack_type_col: str    # string category used for drift


def _largest_csv(folder: Path) -> Path:
    csvs = list(folder.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {folder}")
    return sorted(csvs, key=lambda p: p.stat().st_size, reverse=True)[0]


def _downsample_concat(buf: pd.DataFrame | None, chunk: pd.DataFrame, sample: int, rng: np.random.Generator):
    """Concatenate chunk into buf and downsample to at most `sample` rows."""
    if buf is None:
        buf = chunk
    else:
        buf = pd.concat([buf, chunk], ignore_index=True)

    if sample and sample > 0 and len(buf) > sample:
        idx = rng.choice(len(buf), size=sample, replace=False)
        buf = buf.iloc[idx].reset_index(drop=True)
    return buf


def _reservoir_csv(
    files: list[Path],
    sample: int,
    seed: int,
    read_kwargs: dict,
    chunk_size: int = 100_000,
):
    """Streaming reservoir sampler over one or many CSV files."""

    rng = np.random.default_rng(seed)
    if not sample or sample <= 0:
        dfs = [pd.read_csv(p, **read_kwargs) for p in files]
        return pd.concat(dfs, ignore_index=True)

    buf: pd.DataFrame | None = None
    for p in files:
        for chunk in pd.read_csv(p, chunksize=chunk_size, **read_kwargs):
            buf = _downsample_concat(buf, chunk, sample, rng)
    return buf if buf is not None else pd.DataFrame()


def _reservoir_parquet(files: list[Path], sample: int, seed: int):
    """Reservoir sample Parquet files by iterating row groups (memory-safe)."""

    if pq is None:
        raise ImportError("pyarrow is required to read Parquet files.")

    rng = np.random.default_rng(seed)
    if not sample or sample <= 0:
        dfs = [pd.read_parquet(p) for p in files]
        return pd.concat(dfs, ignore_index=True)

    buf: pd.DataFrame | None = None
    for p in files:
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)
            chunk = tbl.to_pandas()
            buf = _downsample_concat(buf, chunk, sample, rng)
    return buf if buf is not None else pd.DataFrame()


def _find_first_by_name_ci(folder: Path, names: list[str]) -> Path | None:
    """
    Find the first file in folder (recursive) whose name matches one of `names`,
    ignoring case. This is robust to Kaggle / OS case differences like
    KDDTrain+.txt vs KDDTrain+.TXT.
    """
    target = {n.lower() for n in names}
    for p in folder.rglob("*"):
        if p.is_file() and p.name.lower() in target:
            return p
    return None


def load_nslkdd(folder: Path, sample: int = 0, seed: int = 42) -> Loaded:
    # search recursively to be robust in kagglehub cache layout and filename case
    train = _find_first_by_name_ci(
        folder, ["KDDTrain+.TXT", "KDDTrain+_20Percent.TXT", "KDDTrain+.txt", "KDDTrain+_20Percent.txt"]
    )
    test = _find_first_by_name_ci(
        folder, ["KDDTest+.TXT", "KDDTest-21.TXT", "KDDTest+.txt", "KDDTest-21.txt"]
    )

    if train is None or test is None:
        raise FileNotFoundError(
            "NSL-KDD expects KDDTrain+.txt and KDDTest+.txt (or alternatives) in the dataset folder."
        )

    cols = [f"f{i}" for i in range(1, 42)] + ["attack_type", "difficulty"]
    df_train = pd.read_csv(train, header=None, names=cols)
    df_test = pd.read_csv(test, header=None, names=cols)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Optional downsample for quick runs
    if sample and sample > 0 and len(df) > sample:
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)

    df["label"] = (df["attack_type"] != "normal").astype(int)
    return Loaded(df=df, label_col="label", attack_type_col="attack_type")


def load_unsw(folder: Path, sample: int = 0, seed: int = 42) -> Loaded:
    """
    Load UNSW-NB15.

    The Kaggle dataset we use (`harshwardhanbhangale/unsw-complete-dataset`)
    ships the main CSVs (`UNSW-NB15_1.csv` ... `UNSW-NB15_4.csv`) **without**
    a header row, and a separate `NUSW-NB15_features.csv` that describes
    the column names (including `attack_cat` and `Label`).

    To be robust we:
    - Prefer using the feature description file to build explicit column names
      and read the header-less CSVs.
    - Fallback to the previous behaviour (largest CSV with inferred header)
      if the feature file is missing for some reason.
    """

    # Try to find feature description file to get canonical column names.
    feature_files = list(folder.rglob("*NB15_features*.csv"))
    df = None

    if feature_files:
        feat_path = feature_files[0]
        # Kaggle's feature description file may contain non-UTF8 characters,
        # so use a more permissive encoding.
        feat = pd.read_csv(feat_path, encoding="latin1")

        # Column that holds the actual feature names (e.g. 'Name')
        name_col_candidates = [c for c in feat.columns if c.strip().lower() == "name"]
        if name_col_candidates:
            name_col = name_col_candidates[0]
            col_names = feat[name_col].astype(str).tolist()

            # Read all UNSW-NB15_*.csv files (they are headerless)
            data_files = sorted(folder.rglob("UNSW-NB15_*.csv"))
            if not data_files:
                raise FileNotFoundError("UNSW: no UNSW-NB15_*.csv files found.")

            df = _reservoir_csv(
                data_files,
                sample=sample,
                seed=seed,
                read_kwargs={
                    "header": None,
                    "names": col_names,
                    "low_memory": False,
                },
            )

    # Fallback if we could not build from feature description for any reason.
    if df is None:
        p = _largest_csv(folder)
        df = _reservoir_csv([p], sample=sample, seed=seed, read_kwargs={"low_memory": False})

    # case-insensitive mapping
    lower = {c.lower(): c for c in df.columns}
    label_c = lower.get("label")
    atk_c = lower.get("attack_cat") or lower.get("attackcat")

    if label_c is None:
        raise ValueError("UNSW: missing 'label' column.")
    if atk_c is None:
        raise ValueError(
            "UNSW: missing 'attack_cat' column (needed for drift).")

    df[label_c] = df[label_c].astype(int)
    df[atk_c] = df[atk_c].fillna("Benign").astype(str)

    # standardize names
    if label_c != "label":
        df = df.rename(columns={label_c: "label"})
        label_c = "label"
    if atk_c != "attack_type":
        df = df.rename(columns={atk_c: "attack_type"})
        atk_c = "attack_type"

    return Loaded(df=df, label_col=label_c, attack_type_col=atk_c)


def load_cicids2018(folder: Path, sample: int = 0, seed: int = 42) -> Loaded:
    """
    Load CSE-CIC-IDS2018.

    The Kaggle dataset (`dhoogla/csecicids2018`) currently ships Parquet files
    (e.g. `*_TrafficForML_CICFlowMeter.parquet`). Older/other distributions
    may use CSVs. We support both:
    - Read all CSVs if present.
    - Additionally read all Parquet files if pyarrow/fastparquet is installed.
    """

    csvs = list(folder.rglob("*.csv"))
    pars = list(folder.rglob("*.parquet"))
    if not csvs and not pars:
        raise FileNotFoundError("CIC-IDS2018: no CSV or Parquet files found.")

    buf: pd.DataFrame | None = None

    # CSVs (if any)
    if csvs:
        try:
            df_csv = _reservoir_csv(csvs, sample=sample, seed=seed, read_kwargs={"low_memory": False})
            buf = _downsample_concat(buf, df_csv, sample, np.random.default_rng(seed))
        except Exception:
            pass

    # Parquet files (current Kaggle format)
    if pars:
        try:
            df_par = _reservoir_parquet(pars, sample=sample, seed=seed)
            buf = _downsample_concat(buf, df_par, sample, np.random.default_rng(seed + 1))
        except Exception:
            pass

    if buf is None or buf.empty:
        raise ValueError("CIC-IDS2018: cannot read any CSV/Parquet files.")

    df = buf

    # Label column typically 'Label'
    if "Label" not in df.columns:
        cand = [c for c in df.columns if c.lower() == "label"]
        if not cand:
            raise ValueError("CIC-IDS2018: missing Label column.")
        df = df.rename(columns={cand[0]: "Label"})

    df["label"] = (df["Label"].astype(str).str.lower() != "benign").astype(int)
    df["attack_type"] = df["Label"].astype(str)
    return Loaded(df=df, label_col="label", attack_type_col="attack_type")


def load_dataset(folder: Path, dataset: str, sample: int = 0, seed: int = 42) -> Loaded:
    if dataset == "nslkdd":
        return load_nslkdd(folder, sample=sample, seed=seed)
    if dataset == "unsw":
        return load_unsw(folder, sample=sample, seed=seed)
    if dataset == "cicids2018":
        return load_cicids2018(folder, sample=sample, seed=seed)
    raise ValueError("Unknown dataset")
