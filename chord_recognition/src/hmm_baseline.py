from __future__ import annotations
from typing import List, Tuple
import numpy as np
from hmmlearn.hmm import GaussianHMM
import joblib
from pathlib import Path

from chord_vocab import IDX_TO_LABEL
from utils import MODELS_DIR

N_CLASSES = len(IDX_TO_LABEL)

def _align_sequences(X_list: List[np.ndarray], y_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    Xo, yo = [], []
    for X, y in zip(X_list, y_list):
        if X.ndim != 2 or y.ndim != 1:
            continue
        Bx = X.shape[1]
        By = len(y)
        m = min(Bx, By)
        if m <= 0:
            continue
        X = X[:, :m]
        y = y[:m]
        good = np.isfinite(X).all(axis=0)
        if not np.all(good):
            X = X[:, good]
            y = y[good]
        if X.shape[1] == 0:
            continue
        Xo.append(X)
        yo.append(y)
    return Xo, yo

def estimate_emissions(X_list: List[np.ndarray], y_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = _align_sequences(X_list, y_list)

    X = np.concatenate([x.T for x in X_list], axis=0)
    y = np.concatenate(y_list, axis=0)
    assert X.shape[0] == y.shape[0], "Feature/label length mismatch after alignment"

    F = X.shape[1]
    means = np.zeros((N_CLASSES, F), dtype=np.float32)
    covars = np.zeros((N_CLASSES, F), dtype=np.float32)

    for c in range(N_CLASSES):
        mask = (y == c)
        if mask.any():
            Xc = X[mask]
            means[c] = Xc.mean(axis=0)
            covars[c] = Xc.var(axis=0) + 1e-6
        else:
            means[c] = 0.0
            covars[c] = 1.0
    return means, covars

def estimate_transitions(y_seqs: List[np.ndarray], smoothing: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    _, y_seqs = _align_sequences([np.zeros((1,len(y))) for y in y_seqs], y_seqs)

    A = np.full((N_CLASSES, N_CLASSES), smoothing, dtype=np.float64)
    pi = np.full((N_CLASSES,), smoothing, dtype=np.float64)
    for y in y_seqs:
        if len(y)==0: 
            continue
        pi[y[0]] += 1.0
        for i in range(len(y)-1):
            A[y[i], y[i+1]] += 1.0

    A = A / A.sum(axis=1, keepdims=True)
    pi = pi / pi.sum()
    return A, pi

def train_hmm(train_X: List[np.ndarray], train_y: List[np.ndarray]) -> GaussianHMM:
    means, covars = estimate_emissions(train_X, train_y)
    A, pi = estimate_transitions(train_y)
    n_features = means.shape[1]
    model = GaussianHMM(n_components=N_CLASSES, covariance_type="diag", init_params="", params="")
    model.startprob_ = pi
    model.transmat_ = A
    model.means_ = means
    model.covars_ = covars
    return model

def decode_song(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    _, states = model.decode(X.T, algorithm="viterbi")
    return states

def save_model(model: GaussianHMM, name: str = "hmm_baseline.pkl") -> Path:
    path = MODELS_DIR / name
    joblib.dump(model, path)
    return path

def load_model(path: Path) -> GaussianHMM:
    return joblib.load(path)
