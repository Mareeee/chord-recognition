from __future__ import annotations
from typing import List
import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, X_list: List[np.ndarray], y_list: List[np.ndarray]):
        assert len(X_list) == len(y_list)
        self.X = X_list
        self.y = y_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        X = self.X[idx]
        y = self.y[idx]

        B = min(X.shape[1], len(y))
        X = X[:, :B]
        y = y[:B]
        return torch.from_numpy(X).float(), torch.from_numpy(y).long()

def collate_pad(batch):
    xs, ys = zip(*batch)
    lens = [x.shape[1] for x in xs]
    T = max(lens) if lens else 0
    F = xs[0].shape[0] if xs else 12
    B = len(xs)
    import torch
    Xb = torch.zeros((B, F, T), dtype=torch.float32)
    yb = torch.full((B, T), -100, dtype=torch.long)
    for i,(x,y) in enumerate(zip(xs,ys)):
        t = x.shape[1]
        Xb[i, :, :t] = x
        yb[i, :t] = y
    return Xb, yb, torch.tensor(lens, dtype=torch.long)
