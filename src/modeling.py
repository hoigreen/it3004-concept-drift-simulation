from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Optional torch import (kept here to fail fast with a clear message)
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:  # pragma: no cover - best-effort import guard
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


# ---------------------------- LSTM (PyTorch) ---------------------------- #


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a dense preprocessor (impute + scale + one-hot) suitable for torch.
    Returns a fitted ColumnTransformer.
    """

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden: int = 128, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.head(h_last).squeeze(-1)


def _to_tensor(pre: ColumnTransformer, X: pd.DataFrame):
    arr = pre.transform(X)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return arr.astype(np.float32)


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 256, shuffle: bool = True) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    # treat each feature vector as a length-1 sequence for LSTM compatibility
    X_tensor = X_tensor.unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@dataclass
class LSTMClassifier:
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 8
    batch_size: int = 256
    device: str | None = None

    def __post_init__(self):
        if torch is None:
            raise ImportError("PyTorch is required for LSTMClassifier. Please install 'torch'.")
        self.device_ = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pre = None
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.pre = _make_preprocessor(X)
        self.pre.fit(X)
        X_arr = _to_tensor(self.pre, X)

        input_size = X_arr.shape[1]
        self.model = _LSTMNet(
            input_size=input_size,
            hidden=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device_)

        # class weights to mitigate imbalance
        _, counts = np.unique(y, return_counts=True)
        if len(counts) == 2:
            weight_pos = counts[0] / max(counts[1], 1)
            class_weights = torch.tensor([1.0, weight_pos], device=self.device_)
        else:
            class_weights = None

        loader = _make_loader(X_arr, y, batch_size=self.batch_size, shuffle=True)
        self._train_loop(loader, class_weights)
        return self

    def _train_loop(self, loader: DataLoader, class_weights):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1] if class_weights is not None else None)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                optim.zero_grad()
                loss.backward()
                optim.step()

    def predict(self, X: pd.DataFrame):
        if self.model is None or self.pre is None:
            raise RuntimeError("Model not fitted")
        X_arr = _to_tensor(self.pre, X)
        loader = _make_loader(X_arr, np.zeros(len(X_arr)), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        preds = []
        probas = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device_)
                logits = self.model(xb)
                p = torch.sigmoid(logits)
                probas.append(p.cpu().numpy())
                preds.append((p > 0.5).float().cpu().numpy())
        y_proba = np.concatenate(probas, axis=0)
        y_pred = np.concatenate(preds, axis=0).astype(int)
        return y_pred, y_proba
