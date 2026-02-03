from __future__ import annotations
import numpy as np
import pandas as pd


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    # drop all-empty columns
    df = df.dropna(axis=1, how="all")
    return df


def maybe_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if n and n > 0 and len(df) > n:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df


def split_xy(df: pd.DataFrame, label_col: str, attack_type_col: str):
    """
    Split dataframe into features X, binary label y and attack_type.

    To keep scikit-learn's encoders happy, we ensure that all non-numeric
    feature columns are cast to string, so there are no mixed int/str columns
    that would otherwise raise TypeError in OneHotEncoder.
    """
    y = df[label_col].astype(int).to_numpy()

    # Work on a copy for feature processing
    X = df.drop(columns=[label_col]).copy()

    # Ensure attack_type column exists and is string-typed, both in X and
    # in the separate attack_type array used for drift simulation.
    if attack_type_col in X.columns:
        X[attack_type_col] = X[attack_type_col].astype(str)
        attack_type = X[attack_type_col].to_numpy()
    else:
        attack_type = df[attack_type_col].astype(str).to_numpy()

    # Cast all non-numeric feature columns to string so that OneHotEncoder
    # sees uniform types within each column.
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str)

    # keep attack_type in X for chunk selection logic; model will treat it as categorical
    return X, y, attack_type
