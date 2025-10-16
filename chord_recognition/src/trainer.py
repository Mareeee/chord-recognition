import os
import platform
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.model import CNNChordNet


# -------------------- POMOĆNE: SPLIT BEZ CURENJA --------------------

def get_artist_key(mir_track, track_id: str) -> str:
    """
    Pokušaj da izvučeš identitet izvođača (artist/player) za GuitarSet.
    Radi robustno: pokušava više atributa; ako nema, pokuša da ga izvuče iz track_id.
    Ako baš ništa ne uspe, vrati prefiks track_id do prvog underscore-a.
    """
    for attr in ["artist", "player", "performer", "guitarist", "player_id"]:
        if hasattr(mir_track, attr):
            val = getattr(mir_track, attr)
            if val not in (None, "", "unknown"):
                return str(val)

    m = re.match(r"^([0-9]{2})[_-]", track_id)
    if m:
        return m.group(1)

    if "_" in track_id:
        return track_id.split("_", 1)[0]
    if "-" in track_id:
        return track_id.split("-", 1)[0]
    return track_id


def artist_based_split(dataset_instance, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Vraća liste indeksa za train/val/test tako da izvođači (artist_key)
    nemaju preklapanja između splitova.
    """
    rng = random.Random(seed)
    ds = dataset_instance
    md = ds.dataset

    artist_to_indices = defaultdict(list)
    for idx, tid in enumerate(ds.track_ids):
        try:
            mir_track = md.track(tid)
        except Exception:
            mir_track = None
        artist_key = get_artist_key(mir_track, tid)
        artist_to_indices[artist_key].append(idx)

    artists = list(artist_to_indices.keys())
    rng.shuffle(artists)

    n_art = len(artists)
    n_train = int(round(train_ratio * n_art))
    n_val   = int(round(val_ratio * n_art))
    # ostatak je test
    train_artists = set(artists[:n_train])
    val_artists   = set(artists[n_train:n_train + n_val])
    test_artists  = set(artists[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for a, idxs in artist_to_indices.items():
        if a in train_artists:
            train_idx.extend(idxs)
        elif a in val_artists:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    assert set(train_idx).isdisjoint(val_idx) and set(train_idx).isdisjoint(test_idx) and set(val_idx).isdisjoint(test_idx), \
        "Leakage detektovan: indeksi se preklapaju!"
    return train_idx, val_idx, test_idx, (train_artists, val_artists, test_artists)


# -------------------- EVALUACIJA --------------------

def evaluate_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100.0 * correct / max(1, total)
    model.train()
    return acc


def test_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing on Test Set", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / max(1, total)


# -------------------- GLAVNI POKRETAČ --------------------

def _loader_kwargs(device):
    """
    * Windows: num_workers=0 (spawn + pickle problemi -> tvoj crash)
    * Ostalo: skromno paralelno, ali bezbedno.
    * pin_memory samo kada postoji CUDA, inače False (skida warning).
    """
    on_windows = platform.system().lower().startswith("win")
    has_cuda = (device.type == "cuda")
    if on_windows:
        return dict(num_workers=0, pin_memory=False, persistent_workers=False)
    else:
        return dict(num_workers=2, pin_memory=has_cuda, persistent_workers=False)


def run_training_evaluation(
    dataset,
    num_classes,
    device,
    epochs=5,
    lr=1e-3,
    model_path="cnn_chordnet_baseline.pth",
    max_batches_per_epoch=100
):
    # NOVI AMP API (nema više FutureWarning):
    from torch.amp import autocast as amp_autocast
    from torch.amp import GradScaler as AmpGradScaler
    from torch.utils.data import random_split

    # 1) Split (jednostavan, stabilan seed)
    N = len(dataset)
    n_train = int(0.8 * N)
    n_val   = int(0.1 * N)
    n_test  = N - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )

    # 2) DataLoaders – platform-safe
    lkw = _loader_kwargs(device)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  drop_last=True,  **lkw)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, drop_last=False, **lkw)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, drop_last=False, **lkw)

    # 3) Class weights (da CNN ne „zaglavi“ na jednoj klasi)
    class_counts = np.zeros(num_classes, dtype=np.int64)
    with torch.no_grad():
        for xb, yb in train_loader:
            binc = np.bincount(yb.numpy(), minlength=num_classes)
            class_counts += binc
    class_counts = np.maximum(class_counts, 1)
    inv_sqrt = 1.0 / np.sqrt(class_counts.astype(np.float32))
    class_weights = torch.tensor(inv_sqrt, dtype=torch.float32, device=device)

    # 4) Model + AMP + OneCycle
    model = CNNChordNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(1, min(max_batches_per_epoch, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    # novi scaler API (bez FutureWarning)
    scaler = AmpGradScaler(device.type, enabled=(device.type == "cuda"))

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0

        # ---- TQDM PROGRESS BAR ZA TRAIN ----
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {ep}/{epochs} [train]", leave=True)
        for b, (xb, yb) in enumerate(train_loader, start=1):
            if b > max_batches_per_epoch:
                break
            xb = xb.to(device, non_blocking=False)
            yb = yb.to(device, non_blocking=False)

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * xb.size(0)
            seen += xb.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)
        pbar.close()

        train_loss = total_loss / max(1, seen)

        # ---- TQDM PROGRESS BAR ZA VAL ----
        model.eval()
        val_loss, vseen = 0.0, 0
        pbar_v = tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [val]  ", leave=False)
        with torch.no_grad(), amp_autocast(device_type=device.type, enabled=(device.type == "cuda")):
            for xb, yb in pbar_v:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                vseen += xb.size(0)
                pbar_v.set_postfix(loss=f"{loss.item():.4f}")
        pbar_v.close()
        val_loss /= max(1, vseen)

        print(f"Epoch {ep}/{epochs} - train {train_loss:.4f}  |  val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_path)
            print("  Saved best model.")

    # (Po potrebi: test acc; CSR/WCSR radiš opcijom 5 iz menija)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return {"best_val_loss": best_val}
