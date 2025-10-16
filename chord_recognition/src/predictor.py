import os
import numpy as np

from src.prediction_logic import extract_features_for_prediction, SR, HOP
from src.hmm_utils import (
    extract_raw_cnn_log_probabilities,
    viterbi_decoding,
    HMM_TRANSITION_PATH,
)


# ---------- TUNE HERE (default predikcije) ----------
DEFAULT_STICKY_ALPHA = 0.0
DEFAULT_SMOOTH_K     = 1      # 1 = bez klizne mode (3 je blago glačanje)
DEFAULT_MIN_RUN      = 1      # 1 = bez spajanja kratkih run-ova
DEFAULT_SEG_MODE     = "beat_or_fixed"  # "beat_or_fixed" | "fixed"
DEFAULT_OVERLAP      = 0.5    # koristi se samo za fixed mod
# ---------------------------------------------------


def _majority_smooth(indices, k=DEFAULT_SMOOTH_K):
    """
    Klizna moda (k neparan). k=1 => bez efekta.
    """
    if len(indices) == 0 or k <= 1:
        return list(indices)

    k = max(1, int(k))
    if k % 2 == 0:
        k += 1

    half = k // 2
    out = []
    for i in range(len(indices)):
        s = max(0, i - half)
        e = min(len(indices), i + half + 1)
        window = indices[s:e]
        vals, counts = np.unique(window, return_counts=True)
        out.append(int(vals[np.argmax(counts)]))
    return out


def _merge_short_runs(indices, min_run=DEFAULT_MIN_RUN):
    """
    Ako je run dužine < min_run, spoji ga sa susedom (levo/desno).
    min_run=1 => nema spajanja.
    """
    if len(indices) == 0 or min_run <= 1:
        return list(indices)

    runs = []
    cur = indices[0]
    ln = 1
    for x in indices[1:]:
        if x == cur:
            ln += 1
        else:
            runs.append([cur, ln])
            cur = x
            ln = 1
    runs.append([cur, ln])

    i = 0
    while i < len(runs):
        val, length = runs[i]
        if length < min_run and len(runs) > 1:
            if i == 0:
                runs[i + 1][1] += length
                runs.pop(i)
                continue
            elif i == len(runs) - 1:
                runs[i - 1][1] += length
                runs.pop(i)
                i -= 1
                continue
            else:
                left_len = runs[i - 1][1]
                right_len = runs[i + 1][1]
                if right_len >= left_len:
                    runs[i + 1][1] += length
                    runs.pop(i)
                    continue
                else:
                    runs[i - 1][1] += length
                    runs.pop(i)
                    i -= 1
                    continue
        i += 1

    out = []
    for v, length in runs:
        out.extend([int(v)] * int(length))
    return out


def _centers_and_effective_durations(meta):
    """
    Izračunaj (centar, efektivno trajanje) za svaki segment bez dupliranja.
    Za beat-mode, step je varijabilan i dolazi iz meta['bounds'].
    """
    bounds = meta.get("bounds", [])
    if not bounds:
        return np.array([]), np.array([])

    centers = []
    durs = []
    for (s_f, e_f) in bounds:
        s_sec = (int(s_f) * HOP) / float(SR)
        e_sec = (int(e_f) * HOP) / float(SR)
        centers.append((s_sec + e_sec) * 0.5)
        durs.append(max(0.0, e_sec - s_sec))

    return np.asarray(centers, dtype=float), np.asarray(durs, dtype=float)


def predict_frames(
    audio_path,
    model_path,
    dataset_instance,
    device,
    use_hmm=False,
    sticky_alpha=DEFAULT_STICKY_ALPHA,
    seg_mode=DEFAULT_SEG_MODE,
    overlap=DEFAULT_OVERLAP,
    smooth_k=DEFAULT_SMOOTH_K,
    min_run=DEFAULT_MIN_RUN,
):
    """
    Segment-level predikcija:
      - CNN log-probovi po segmentu
      - opciono HMM Viterbi (sa 'stickiness' pojačanjem dijagonale)
      - opcioni smoothing (klizna moda + merge kratkih run-ova)
    Vraća:
      frame_times_sec: centri segmenata (u s)
      est_indices:     niz klasa po segmentu
      chord_labels:    mapiranje indeksa → label
    """
    chord_labels = dataset_instance.chord_labels

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_path}' not found. Please run training first (Option 1).")

    # CNN emissions + meta granice
    log_probs, meta = extract_raw_cnn_log_probabilities(
        audio_path, model_path, dataset_instance, device, overlap=overlap, seg_mode=seg_mode
    )
    if log_probs.size == 0 or meta is None:
        return np.array([]), np.array([]), chord_labels

    raw_idx = np.argmax(log_probs, axis=1).tolist()

    if use_hmm and os.path.exists(HMM_TRANSITION_PATH):
        try:
            T = np.load(HMM_TRANSITION_PATH)
            hmm_idx = viterbi_decoding(log_probs, T, sticky_alpha=sticky_alpha)
            idx_seq = hmm_idx
        except Exception as e:
            print(f"[WARN] HMM decoding failed ({e}). Falling back to raw CNN.")
            idx_seq = raw_idx
    else:
        idx_seq = raw_idx

    # opcioni smoothing
    idx_seq = _majority_smooth(idx_seq, k=smooth_k)
    idx_seq = _merge_short_runs(idx_seq, min_run=min_run)

    # centri i trajanja (tačno, bez dupliranja)
    centers, _ = _centers_and_effective_durations(meta)

    # uskladi dužine ako je potrebno
    if len(idx_seq) != len(centers):
        common = min(len(idx_seq), len(centers))
        idx_seq = idx_seq[:common]
        centers = centers[:common]

    print(f"[predict] segments={len(idx_seq)}, classes={len(chord_labels)}")
    return centers, np.asarray(idx_seq, dtype=np.int64), chord_labels


def predict_song(
    audio_path,
    model_path,
    dataset_instance,
    device,
    use_hmm=False,
    sticky_alpha=DEFAULT_STICKY_ALPHA,
    seg_mode=DEFAULT_SEG_MODE,
    overlap=DEFAULT_OVERLAP,
    smooth_k=DEFAULT_SMOOTH_K,
    min_run=DEFAULT_MIN_RUN,
):
    """
    CLI-friendly: štampa sažetu sekvencu akorda i tačno trajanje.
    """
    centers, est_idx, chord_labels = predict_frames(
        audio_path,
        model_path,
        dataset_instance,
        device,
        use_hmm=use_hmm,
        sticky_alpha=sticky_alpha,
        seg_mode=seg_mode,
        overlap=overlap,
        smooth_k=smooth_k,
        min_run=min_run,
    )

    if len(est_idx) == 0:
        print("Error: Could not extract features from audio file.")
        return

    # meta samo da uzmemo durations (već su precizne po bounds)
    feats, meta = extract_features_for_prediction(
        audio_path, max_frames=128, overlap=overlap, return_meta=True, seg_mode=seg_mode
    )
    if meta is None or not meta.get("bounds"):
        # fallback: konzervativno pretpostavi trajanja po proseku
        seg_durs = np.full(len(est_idx), (128 * HOP) / float(SR), dtype=float)
    else:
        _, seg_durs = _centers_and_effective_durations(meta)
        if len(seg_durs) != len(est_idx):
            common = min(len(seg_durs), len(est_idx))
            seg_durs = seg_durs[:common]
            est_idx = est_idx[:common]

    print("\n--- PREDICTED CHORD SEQUENCE ---")
    current = int(est_idx[0])
    acc_dur = float(seg_durs[0]) if len(seg_durs) else 0.0

    for i in range(1, len(est_idx)):
        cls = int(est_idx[i])
        if cls == current:
            acc_dur += float(seg_durs[i])
        else:
            print(f"Chord: {chord_labels[current]}  ({acc_dur:.2f}s)")
            current = cls
            acc_dur = float(seg_durs[i])

    print(f"Chord: {chord_labels[current]}  ({acc_dur:.2f}s)")
    print(
        "\nNote:",
        "Output smoothed using HMM Viterbi decoding with stickiness."
        if use_hmm
        else "Output based on CNN with optional light smoothing.",
    )
