from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


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


def load_nslkdd(folder: Path) -> Loaded:
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

    df["label"] = (df["attack_type"] != "normal").astype(int)
    return Loaded(df=df, label_col="label", attack_type_col="attack_type")


def load_unsw(folder: Path) -> Loaded:
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

            dfs = []
            for p in data_files:
                dfs.append(
                    pd.read_csv(
                        p,
                        header=None,
                        names=col_names,
                        low_memory=False,
                    )
                )
            df = pd.concat(dfs, ignore_index=True)

    # Fallback if we could not build from feature description for any reason.
    if df is None:
        p = _largest_csv(folder)
        df = pd.read_csv(p, low_memory=False)

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


def load_cicids2018(folder: Path) -> Loaded:
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

    dfs: list[pd.DataFrame] = []

    # CSVs (if any)
    for p in csvs:
        try:
            dfs.append(pd.read_csv(p, low_memory=False))
        except Exception:
            continue

    # Parquet files (current Kaggle format)
    for p in pars:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception:
            continue

    if not dfs:
        raise ValueError("CIC-IDS2018: cannot read any CSV/Parquet files.")

    df = pd.concat(dfs, ignore_index=True)

    # Label column typically 'Label'
    if "Label" not in df.columns:
        cand = [c for c in df.columns if c.lower() == "label"]
        if not cand:
            raise ValueError("CIC-IDS2018: missing Label column.")
        df = df.rename(columns={cand[0]: "Label"})

    df["label"] = (df["Label"].astype(str).str.lower() != "benign").astype(int)
    df["attack_type"] = df["Label"].astype(str)
    return Loaded(df=df, label_col="label", attack_type_col="attack_type")


def load_dataset(folder: Path, dataset: str) -> Loaded:
    if dataset == "nslkdd":
        return load_nslkdd(folder)
    if dataset == "unsw":
        return load_unsw(folder)
    if dataset == "cicids2018":
        return load_cicids2018(folder)
    raise ValueError("Unknown dataset")
