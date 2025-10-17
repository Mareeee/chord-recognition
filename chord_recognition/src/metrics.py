from __future__ import annotations
from typing import Dict
import numpy as np

def csr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0

def wcsr(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    if len(y_true)==0: return 0.0
    w = weights[:len(y_true)]
    correct = (y_true == y_pred).astype(float)
    return float((correct * w).sum() / (w.sum() + 1e-8))

def overlap_ratio(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return wcsr(y_true, y_pred, weights)

def beat_weights(boundaries: np.ndarray) -> np.ndarray:
    return boundaries[1:] - boundaries[:-1]

def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, boundaries: np.ndarray) -> Dict[str, float]:
    T = min(len(y_true), len(y_pred), len(boundaries) - 1)
    if T <= 0:
        return {"CSR": 0.0, "WCSR": 0.0, "Overlap": 0.0}

    y_true = y_true[:T]
    y_pred = y_pred[:T]
    boundaries = boundaries[:T + 1]

    w = beat_weights(boundaries)
    return {
        "CSR": csr(y_true, y_pred),
        "WCSR": wcsr(y_true, y_pred, w),
        "Overlap": overlap_ratio(y_true, y_pred, w),
    }
