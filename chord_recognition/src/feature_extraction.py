from __future__ import annotations
from typing import Tuple, Iterable, List
import numpy as np
import librosa
rng = np.random.default_rng(0)

HOP_LENGTH = 512
SR = 44100

def load_audio(path: str, sr: int = SR) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y, sr

def chroma_cqt(y: np.ndarray, sr: int = SR, hop_length: int = HOP_LENGTH) -> np.ndarray:
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=12)
    C = C / (C.sum(axis=0, keepdims=True) + 1e-8)
    return C

def beat_sync(S: np.ndarray, y: np.ndarray, sr: int = SR, hop_length: int = HOP_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    if beats.size == 0:
        frames = np.arange(S.shape[1])
        win = int((0.5*sr) // hop_length)
        if win <= 1: 
            return S, librosa.frames_to_time(np.arange(S.shape[1]+1), sr=sr, hop_length=hop_length)
        boundaries = np.concatenate([[0], np.arange(win, frames[-1]+1, win)])
        boundaries = np.clip(boundaries, 0, frames[-1])
        boundaries = np.unique(boundaries).astype(int)
        S_sync = []
        for i in range(len(boundaries)-1):
            seg = S[:, boundaries[i]:boundaries[i+1]]
            S_sync.append(seg.mean(axis=1) if seg.size else S[:, boundaries[i]])
        S_sync = np.stack(S_sync, axis=1) if S_sync else S
        times = librosa.frames_to_time(boundaries, sr=sr, hop_length=hop_length)
        total_time = librosa.get_duration(y=y, sr=sr)
        times = np.concatenate([times, [total_time]])
        return S_sync, times
    S_sync = librosa.util.sync(S, beats, aggregate=np.mean)
    beat_frames = np.concatenate([[0], beats, [S.shape[1]]])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    return S_sync, beat_times

def add_noise(y: np.ndarray, snr_db: float = 25.0) -> np.ndarray:
    if np.allclose(y, 0):
        return y.copy()
    sig_power = np.mean(y**2)
    snr_linear = 10**(snr_db/10)
    noise_power = sig_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape)
    out = y + noise
    out = out / max(np.max(np.abs(out)), 1e-8)
    return out

def pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    if y.size < 2:
        return y
    out = librosa.effects.time_stretch(y, rate=rate)
    if np.max(np.abs(out)) > 0:
        out = out / np.max(np.abs(out))
    return out

def augmented_waves(y: np.ndarray, sr: int, pitch_steps: Iterable[float]=(-2,2), stretch_rates: Iterable[float]=(0.9,1.1), noise_snr_db: float=25.0) -> List[np.ndarray]:
    outs = []
    for ps in pitch_steps:
        outs.append(pitch_shift(y, sr, ps))
    for rt in stretch_rates:
        outs.append(time_stretch(y, rt))
    outs.append(add_noise(y, snr_db=noise_snr_db))
    return outs
