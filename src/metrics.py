from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


def compute_all(y_true, y_pred, y_proba):
    out = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["precision_attack"] = precision_score(
        y_true, y_pred, pos_label=1, zero_division=0)
    out["recall_attack"] = recall_score(
        y_true, y_pred, pos_label=1, zero_division=0)
    out["f1_attack"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    if y_proba is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
        out["pr_auc"] = average_precision_score(y_true, y_proba)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["tn"], out["fp"], out["fn"], out["tp"] = cm.ravel()
    return out
