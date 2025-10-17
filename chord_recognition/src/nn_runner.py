from __future__ import annotations
from typing import List, Dict
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from chord_vocab import IDX_TO_LABEL, id_to_chord_label
from utils import MODELS_DIR, PRED_DIR
from metrics import metrics_dict, beat_weights
from feature_extraction import load_audio, chroma_cqt, beat_sync, SR
from data_utils import BeatSample
from nn_data import SequenceDataset, collate_pad
from models.nn_models import SimpleCNN, SimpleLSTM, CNNLSTM

N_CLASSES = len(IDX_TO_LABEL)

def _class_weights_from_y(y_list, n_classes):
    counts = np.zeros(n_classes, dtype=np.float64)
    for y in y_list:
        if y is None or len(y)==0: 
            continue
        yy = np.asarray(y)
        for c in range(n_classes):
            counts[c] += (yy == c).sum()

    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(kind: str):
    kind = kind.lower()
    if kind == "cnn":
        return SimpleCNN(N_CLASSES)
    if kind == "lstm":
        return SimpleLSTM(N_CLASSES)
    if kind in ("cnn_lstm","cnn+lstm","cnnlstm"):
        return CNNLSTM(N_CLASSES)
    raise ValueError(f"Unknown model kind: {kind}")

def model_paths(kind: str) -> Path:
    fname = {
        "cnn": "cnn_baseline.pt",
        "lstm": "lstm_baseline.pt",
        "cnn_lstm": "cnn_lstm_baseline.pt",
        "cnn+lstm": "cnn_lstm_baseline.pt",
        "cnnlstm": "cnn_lstm_baseline.pt",
    }.get(kind.lower(), f"{kind}.pt")
    return MODELS_DIR / fname

def train(kind: str, X_list: List[np.ndarray], y_list: List[np.ndarray],
          epochs: int = 5, batch_size: int = 4, lr: float = 1e-3) -> Path:
    model = build_model(kind).to(device())
    ds = SequenceDataset(X_list, y_list)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)

    if lr is None:
        lr = 1e-3 if kind.lower()=="cnn" else 5e-4

    if kind.lower() == "cnn":
        def forward_batch(Xb, lengths):
            x = Xb.unsqueeze(1).to(device())
            return model(x)
    elif kind.lower() == "lstm":
        def forward_batch(Xb, lengths):
            x = Xb.transpose(1,2).to(device())
            return model(x, lengths.to(device()))
    else:
        def forward_batch(Xb, lengths):
            x = Xb.to(device())
            return model(x)

    from chord_vocab import IDX_TO_LABEL
    weights = _class_weights_from_y(y_list, n_classes=len(IDX_TO_LABEL)).to(device())
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    print("Class weights:", weights.detach().cpu().numpy())
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        n_frames = 0
        for Xb, yb, lengths in dl:
            logits = forward_batch(Xb, lengths)
            loss = criterion(logits, yb.to(device()))
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss.detach().cpu()) * int((lengths).sum())
            n_frames += int(lengths.sum())
        avg_loss = total_loss / max(n_frames,1)
        print(f"Epoch {epoch}/{epochs} - loss={avg_loss:.4f}")

    out_path = model_paths(kind)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "kind": kind}, out_path)
    print(f"Saved {kind} model to: {out_path}")
    return out_path

def eval_split(kind: str, model_path: Path, samples: List[BeatSample], indices: List[int]) -> Dict[str,float]:
    if not indices:
        return {}
    model = build_model(kind).to(device())
    ckpt = torch.load(model_path, map_location=device())
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    metrics_agg = {"CSR":0.0,"WCSR":0.0,"Overlap":0.0}
    tot_w = 0.0
    with torch.no_grad():
        for i in indices:
            s = samples[i]
            X = torch.from_numpy(s.X).float().unsqueeze(0)
            lengths = torch.tensor([X.shape[-1]], dtype=torch.long)
            if kind.lower() == "cnn":
                logits = model(X.unsqueeze(1).to(device()))
            elif kind.lower() == "lstm":
                logits = model(X.transpose(1,2).to(device()), lengths.to(device()))
            else:
                logits = model(X.to(device()))
            y_pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            T = min(len(s.y), len(y_pred))
            y_pred = y_pred[:T]
            y_true = s.y[:T]
            m = metrics_dict(y_true, y_pred, s.boundaries[:T+1])
            w = beat_weights(s.boundaries[:T+1]).sum()
            tot_w += w
            for k in metrics_agg:
                metrics_agg[k] += m[k]*w
    for k in metrics_agg:
        metrics_agg[k] = metrics_agg[k]/max(tot_w, 1e-8)
    return metrics_agg

def predict_on_pair(kind: str, model_path: Path, sample: BeatSample, out_name: str):
    model = build_model(kind).to(device())
    ckpt = torch.load(model_path, map_location=device())
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(sample.X).float().unsqueeze(0)
        lengths = torch.tensor([X.shape[-1]], dtype=torch.long)
        if kind.lower() == "cnn":
            logits = model(X.unsqueeze(1).to(device()))
        elif kind.lower() == "lstm":
            logits = model(X.transpose(1,2).to(device()), lengths.to(device()))
        else:
            logits = model(X.to(device()))
        y_pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    lines = []
    T = min(len(y_pred), len(sample.boundaries)-1)
    for b in range(T):
        t0, t1 = sample.boundaries[b], sample.boundaries[b+1]
        chord = id_to_chord_label(int(y_pred[b]))
        lines.append(f"{t0:.3f}\t{t1:.3f}\t{chord}")
    out_path = PRED_DIR / out_name
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path

def predict_on_path(kind: str, model_path: Path, audio_path: Path):
    model = build_model(kind).to(device())
    ckpt = torch.load(model_path, map_location=device())
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    y, sr = load_audio(str(audio_path), sr=SR)
    C = chroma_cqt(y, sr)
    X_sync, boundaries = beat_sync(C, y, sr)
    with torch.no_grad():
        X = torch.from_numpy(X_sync).float().unsqueeze(0)
        lengths = torch.tensor([X.shape[-1]], dtype=torch.long)
        if kind.lower() == "cnn":
            logits = model(X.unsqueeze(1).to(device()))
        elif kind.lower() == "lstm":
            logits = model(X.transpose(1,2).to(device()), lengths.to(device()))
        else:
            logits = model(X.to(device()))
        y_pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    lines = []
    T = min(len(y_pred), len(boundaries)-1)
    for b in range(T):
        t0, t1 = boundaries[b], boundaries[b+1]
        chord = id_to_chord_label(int(y_pred[b]))
        lines.append(f"{t0:.3f}\t{t1:.3f}\t{chord}")
    out_path = PRED_DIR / f"{audio_path.stem}_{kind}_inference.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
