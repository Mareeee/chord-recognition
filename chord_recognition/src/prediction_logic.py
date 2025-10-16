import torch
import librosa
import numpy as np

# Konstante (deljene sa predictor/evaluator)
SR = 22050
HOP = 512


def _compute_segments_bounds(T_frames: int, max_frames: int = 128, overlap: float = 0.5):
    """
    Fiksni prozori u FREJMOVIMA sa zadatim overlap-om.
    Vraća listu (start_f, end_f) i step (u frame-ovima).
    """
    step = int(max_frames * (1.0 - overlap))
    step = max(1, min(step, max_frames))

    bounds = []
    start = 0
    while start < T_frames:
        end = min(start + max_frames, T_frames)
        bounds.append((start, end))
        if end == T_frames:
            break
        start += step

    if not bounds:
        bounds = [(0, min(max_frames, T_frames))]
    return bounds, step


def _beat_segment_bounds(y, sr=SR, hop_length=HOP, min_frames=48, pad_to=128):
    """
    Napravi granice segmenata po beatovima; spajaj prekratke delove da ne bi
    padovao sve u nedogled. Vraća bounds u FREJMOVIMA i 'nominalni step' (median dužina).
    """
    # detekcija beatova
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, units="frames")
    if beats is None or len(beats) < 2:
        # nema pouzdanih beatova -> fallback na fiksne prozore (caller će to obraditi)
        return None, None

    # pretvori beat markere u intervale
    bounds = []
    b = beats.tolist()
    for i in range(len(b) - 1):
        s = int(b[i]); e = int(b[i + 1])
        if e > s:
            bounds.append([s, e])

    if not bounds:
        return None, None

    # spoji premale intervale
    merged = []
    cur_s, cur_e = bounds[0]
    for s, e in bounds[1:]:
        if (cur_e - cur_s) < min_frames:
            # spoji sa sledećim
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # nominalni 'step' = medijana dužine segmenata
    lengths = [e - s for (s, e) in merged]
    nominal_step = int(np.median(lengths)) if lengths else None

    # paduj na pad_to (128) desno u feature ekstrakciji; ovde samo bounds vraćamo
    return merged, nominal_step


def extract_features_for_prediction(
    audio_path,
    sr: int = SR,
    hop_length: int = HOP,
    max_frames: int = 128,
    overlap: float = 0.5,
    return_meta: bool = False,
    seg_mode: str = "beat_or_fixed",
    min_beat_frames: int = 48,
):
    """
    Učita audio, izračuna hromagram (12 x T), podeli ga na segmente:
      - ako seg_mode == "beat_or_fixed": pokuša beat-based; ako nema beatova, padne na fixed
      - ako seg_mode == "fixed": koristi ravnomerne prozore
    Vraća tenzor oblika (num_segments, 1, 12, 128) i meta info.
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return (torch.tensor([]), None) if return_meta else torch.tensor([])

    # Hromagram + NORMALIZACIJA PO PITCH-KLASAMA (axis=1) – isto kao na treningu
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)  # (12, T)
    chroma = librosa.util.normalize(chroma, axis=1)

    T = chroma.shape[1]
    if T <= 0:
        return (torch.tensor([]), None) if return_meta else torch.tensor([])

    # --- odaberi granice ---
    bounds = None
    step = None

    if seg_mode == "beat_or_fixed":
        bnd, nominal_step = _beat_segment_bounds(
            y, sr=sr, hop_length=hop_length, min_frames=min_beat_frames, pad_to=max_frames
        )
        if bnd is not None:
            bounds = bnd
            step = nominal_step

    if bounds is None:
        bounds, step = _compute_segments_bounds(T, max_frames=max_frames, overlap=overlap)

    # --- iseckaj i paduj na (12, 128) ---
    segments = []
    for (s, e) in bounds:
        s = int(max(0, s)); e = int(min(T, e))
        if e <= s:
            continue
        seg = chroma[:, s:e]  # (12, <=128)
        if seg.shape[1] < max_frames:
            pad = max_frames - seg.shape[1]
            seg = np.pad(seg, ((0, 0), (0, pad)), mode="constant")
        else:
            seg = seg[:, :max_frames]
        segments.append(
            torch.tensor(seg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,12,128)
        )

    feats = torch.cat(segments, dim=0) if segments else torch.tensor([])

    if return_meta:
        meta = {"T_frames": T, "step": step, "bounds": [(int(s), int(e)) for (s, e) in bounds]}
        return feats, meta
    return feats
