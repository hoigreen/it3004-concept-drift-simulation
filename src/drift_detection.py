from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp


def ks_test_drift(X_ref_num: np.ndarray, X_cur_num: np.ndarray, alpha: float = 0.05, ratio_thr: float = 0.30):
    """
    Compare distribution feature-wise using 2-sample KS test.
    Return: (drift, drift_ratio, drifted_features_count)
    """
    n_features = X_ref_num.shape[1]
    drifted = 0

    for j in range(n_features):
        try:
            _, p = ks_2samp(X_ref_num[:, j], X_cur_num[:, j])
            if p < alpha:
                drifted += 1
        except Exception:
            continue

    ratio = drifted / max(1, n_features)
    return (ratio > ratio_thr), ratio, drifted


def make_river_detector(name: str):
    name = name.lower()
    if name == "adwin":
        from river.drift import ADWIN
        return ADWIN()
    if name == "ddm":
        from river.drift import DDM
        return DDM()
    if name == "eddm":
        from river.drift import EDDM
        return EDDM()
    raise ValueError("Detector must be one of: adwin, ddm, eddm")
