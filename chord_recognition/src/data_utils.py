from __future__ import annotations
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import jams
from tqdm import tqdm

from utils import find_pairs, CACHE_DIR
from feature_extraction import load_audio, chroma_cqt, beat_sync, SR, augmented_waves
from chord_vocab import chord_label_to_id

@dataclass
class BeatSample:
    X: np.ndarray
    y: np.ndarray
    boundaries: np.ndarray
    path_wav: Path
    path_jams: Path

def _get_artist_key(jam: jams.JAMS, fallback_stem: str) -> str:
    artist = None
    try:
        artist = jam.file_metadata.artist
    except Exception:
        artist = None
    if not artist:
        try:
            artist = jam.sandbox.get('artist') or jam.sandbox.get('performer')
        except Exception:
            artist = None
    return (artist or fallback_stem.split("_")[0]).strip()

def _load_chord_intervals(jam: jams.JAMS) -> List[Tuple[float,float,str]]:
    anns = jam.annotations.search(namespace="chord")
    if not anns:
        return []
    chord_ann = anns[0]
    triples = []
    for obs in chord_ann:
        start = float(obs.time)
        dur = float(obs.duration)
        label = str(obs.value).strip()
        end = start + dur
        triples.append((start,end,label))
    return triples

def _label_at(t: float, intervals: List[Tuple[float,float,str]]) -> str:
    for s,e,lbl in intervals:
        if s <= t < e:
            return lbl
    return "N"

def _labels_for_boundaries(boundaries: np.ndarray, intervals: List[Tuple[float,float,str]]) -> np.ndarray:
    mids = (boundaries[:-1] + boundaries[1:]) / 2.0
    labels = [chord_label_to_id(_label_at(float(t), intervals)) for t in mids]
    return np.asarray(labels, dtype=np.int64)

def _features_and_labels_from_wave(y: np.ndarray, sr: int, intervals) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = chroma_cqt(y, sr)
    X_sync, boundaries = beat_sync(C, y, sr)
    y_ids = _labels_for_boundaries(boundaries, intervals)
    return X_sync, y_ids, boundaries

def extract_pair(jams_path: Path, wav_path: Path, cache: bool = True) -> BeatSample:
    cache_path = CACHE_DIR / f"{jams_path.stem}.npz"
    if cache and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return BeatSample(
            X=data["X"], y=data["y"], boundaries=data["boundaries"],
            path_wav=wav_path, path_jams=jams_path
        )
    jam = jams.load(str(jams_path))
    y, sr = load_audio(str(wav_path), sr=SR)
    intervals = _load_chord_intervals(jam)
    X_sync, y_ids, boundaries = _features_and_labels_from_wave(y, sr, intervals)
    if cache:
        np.savez_compressed(cache_path, X=X_sync, y=y_ids, boundaries=boundaries)
    return BeatSample(X=X_sync, y=y_ids, boundaries=boundaries, path_wav=wav_path, path_jams=jams_path)

def extract_pair_with_augment(jams_path: Path, wav_path: Path) -> List[BeatSample]:
    jam = jams.load(str(jams_path))
    y, sr = load_audio(str(wav_path), sr=SR)
    intervals = _load_chord_intervals(jam)
    samples: List[BeatSample] = []

    X_sync, y_ids, boundaries = _features_and_labels_from_wave(y, sr, intervals)
    samples.append(BeatSample(X_sync, y_ids, boundaries, wav_path, jams_path))

    for y_aug in augmented_waves(y, sr, pitch_steps=(-2,2), stretch_rates=(0.9,1.1), noise_snr_db=25.0):
        Xa, ya, ba = _features_and_labels_from_wave(y_aug, sr, intervals)
        samples.append(BeatSample(Xa, ya, ba, wav_path, jams_path))

    return samples

def load_dataset() -> Tuple[List[BeatSample], Dict[str, List[int]]]:
    pairs = find_pairs()
    if not pairs:
        return [], {}
    samples: List[BeatSample] = []
    artist_to_indices: Dict[str, List[int]] = {}
    for i, (jp, wp) in enumerate(tqdm(pairs, desc="Collecting pairs...")):
        jam = jams.load(str(jp))
        artist = _get_artist_key(jam, fallback_stem=jp.stem)
        sample = extract_pair(jp, wp, cache=True)
        samples.append(sample)
        artist_to_indices.setdefault(artist, []).append(i)
    return samples, artist_to_indices

def split_by_artist(artist_to_indices: Dict[str, List[int]], ratios=(0.7, 0.15, 0.15)) -> Dict[str, List[int]]:
    artists = sorted(artist_to_indices.keys())
    n_total = sum(len(v) for v in artist_to_indices.values())
    target_train = int(ratios[0]*n_total)
    target_val = int(ratios[1]*n_total)
    train_idx, val_idx, test_idx = [], [], []
    for a in artists:
        idxs = artist_to_indices[a]
        if len(train_idx) < target_train:
            train_idx.extend(idxs)
        elif len(val_idx) < target_val:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    train_idx = sorted(set(train_idx))
    val_idx = sorted(set(val_idx) - set(train_idx))
    test_idx = sorted(set(test_idx) - set(train_idx) - set(val_idx))
    return {"train": train_idx, "val": val_idx, "test": test_idx}
