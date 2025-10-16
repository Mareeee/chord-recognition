import numpy as np
import librosa

from torch.utils.data import Subset

from src.metrics import compute_framewise_metrics
from src.predictor import predict_frames, SR, HOP
from src.trainer import artist_based_split  # isti split/seed kao u traineru


def _indices_to_intervals_and_labels(frame_times, idx_seq, chord_labels):
    """
    Spoji uzastopne iste indekse u intervale [start, end) + chord string labele.
    """
    if len(idx_seq) == 0:
        return np.zeros((0, 2), dtype=float), []

    intervals = []
    labels = []

    current = int(idx_seq[0])
    start_t = float(frame_times[0]) if len(frame_times) else 0.0

    for i in range(1, len(idx_seq)):
        if int(idx_seq[i]) != current:
            end_t = float(frame_times[i])
            intervals.append([start_t, end_t])
            labels.append(chord_labels[current])
            current = int(idx_seq[i])
            start_t = float(frame_times[i])

    # zatvori poslednji interval
    end_t = float(frame_times[-1]) + (HOP / SR)
    intervals.append([start_t, end_t])
    labels.append(chord_labels[current])

    return np.asarray(intervals, dtype=float), list(labels)


def _weighted_accuracy_intervals(ref_intervals, ref_labels, est_intervals, est_labels):
    """
    Naša implementacija "weighted accuracy":
    - Ulazi: ref/est intervali oblika (N,2) u sekundama i paralelne liste string labela.
    - Izlaz: udeo vremena tokom kojeg je ref_label == est_label (u [0,1]).
    Radi sweep-line spajanje intervala bez oslanjanja na mir_eval.
    """
    ref_intervals = np.asarray(ref_intervals, dtype=float).reshape(-1, 2)
    est_intervals = np.asarray(est_intervals, dtype=float).reshape(-1, 2)
    ref_labels = list(ref_labels)
    est_labels = list(est_labels)

    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        return 0.0

    # Očisti eventualne inverzne/degenerisane intervale
    def _clean(iv, lb):
        out_i, out_l = [], []
        for (s, e), l in zip(iv, lb):
            s = float(s); e = float(e)
            if np.isfinite(s) and np.isfinite(e) and e > s:
                out_i.append([s, e])
                out_l.append(l)
        if not out_i:
            return np.zeros((0, 2), dtype=float), []
        return np.asarray(out_i, dtype=float), out_l

    ref_intervals, ref_labels = _clean(ref_intervals, ref_labels)
    est_intervals, est_labels = _clean(est_intervals, est_labels)

    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        return 0.0

    i = j = 0
    match_time = 0.0
    total_time = 0.0

    while i < len(ref_intervals) and j < len(est_intervals):
        rs, re = ref_intervals[i]
        es, ee = est_intervals[j]

        # presek
        s = max(rs, es)
        e = min(re, ee)

        if e > s:  # imaju preklapanje
            dur = e - s
            total_time += dur
            if ref_labels[i] == est_labels[j]:
                match_time += dur

        # pomeraj pointer sa kraćim krajem
        if re <= ee:
            i += 1
        else:
            j += 1

    if total_time <= 0:
        return 0.0
    return float(match_time / total_time)


def evaluate_on_test(dataset_instance, model_path, device, use_hmm=False, seed=42):
    """
    Evaluacija na TEST splitu (artist-based, isti seed kao u traineru).
    Vraća i štampa prosečne CSR/WCSR/Overlap (frame-wise)
    i naš interval-based weighted accuracy (bez zavisnosti od mir_eval).
    """
    # 1) Rekonstruiši isti split
    train_idx, val_idx, test_idx, _ = artist_based_split(dataset_instance, seed=seed)
    test_ds = Subset(dataset_instance, test_idx)

    frame_sec = HOP / float(SR)

    all_csr, all_wcsr, all_overlap = [], [], []
    all_weighted = []

    original_ids = dataset_instance.track_ids
    chord_vocab = dataset_instance.chord_labels

    print(f"Evaluating {len(test_idx)} tracks in TEST split...")

    for local_i in range(len(test_ds)):
        orig_index = test_idx[local_i]
        tid = original_ids[orig_index]

        # do audio puta i anotacija
        track = dataset_instance.dataset.track(tid)
        audio_path = track.audio_mix_path

        # 2) predikcija frame-by-frame
        try:
            frame_times, est_idx, _ = predict_frames(
                audio_path, model_path, dataset_instance, device, use_hmm=use_hmm
            )
        except Exception as e:
            print(f"[WARN] Skip {tid}: prediction failed ({e})")
            continue

        total_frames = len(frame_times)
        if total_frames == 0:
            print(f"[WARN] Skip {tid}: zero frames.")
            continue

        # 3) ref frame-by-frame iz intervala anotacija
        ref_idx = np.zeros(total_frames, dtype=np.int64)  # default 'N'
        chord_data = track.leadsheet_chords

        if chord_data and len(chord_data.labels) > 0:
            intervals = chord_data.intervals  # sekunde
            labels = chord_data.labels

            for (start, end), lab in zip(intervals, labels):
                if end <= 0:
                    continue
                start_f = max(0, librosa.time_to_frames(start, sr=SR, hop_length=HOP))
                end_f = max(start_f + 1, librosa.time_to_frames(end, sr=SR, hop_length=HOP))
                if start_f >= total_frames:
                    continue
                end_f = min(end_f, total_frames)
                cls_idx = dataset_instance._chord_to_index(lab)
                ref_idx[start_f:end_f] = cls_idx

        # 4) frame-wise metrike
        csr, wcsr, overlap = compute_framewise_metrics(ref_idx, est_idx, frame_sec)
        all_csr.append(csr)
        all_wcsr.append(wcsr)
        all_overlap.append(overlap)

        # 5) interval-based WA (naša implementacija)
        if chord_data and len(chord_data.labels) > 0:
            # ref normalizovane labele u naš vokabular
            gt_intervals_sec = chord_data.intervals
            gt_labels_raw = chord_data.labels
            norm_ref_labels = []
            for lab in gt_labels_raw:
                idx = dataset_instance._chord_to_index(lab)
                norm_ref_labels.append(chord_vocab[idx])

            # est: iz frame predikcija (spoji u intervale)
            est_intervals, est_labels = _indices_to_intervals_and_labels(
                frame_times, est_idx, chord_vocab
            )

            wa = _weighted_accuracy_intervals(
                ref_intervals=gt_intervals_sec,
                ref_labels=norm_ref_labels,
                est_intervals=est_intervals,
                est_labels=est_labels,
            )
            all_weighted.append(wa)

    # 6) agregat
    if len(all_csr) == 0:
        print("No test tracks evaluated.")
        return None

    mean_csr = float(np.mean(all_csr))
    mean_wcsr = float(np.mean(all_wcsr))
    mean_overlap = float(np.mean(all_overlap))

    print("\n--- TEST METRICS ---")
    print("Frame-wise:")
    print(f"- CSR (unweighted accuracy):  {mean_csr*100:.2f}%")
    print(f"- WCSR (time-weighted acc.):  {mean_wcsr*100:.2f}%")
    print(f"- Overlap ratio:              {mean_overlap*100:.2f}%")

    if len(all_weighted) > 0:
        mean_wa = float(np.mean(all_weighted))
        print("Interval-based (custom):")
        print(f"- Weighted Accuracy:          {mean_wa*100:.2f}%")
    else:
        print("\nInterval-based: no GT intervals on the evaluated tracks (or all skipped).")

    return {
        "CSR": mean_csr,
        "WCSR": mean_wcsr,
        "Overlap": mean_overlap,
        "interval_weighted_accuracy": float(np.mean(all_weighted)) if len(all_weighted) > 0 else None,
        "N_tracks": len(all_csr),
    }
