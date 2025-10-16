import csv
import json
import numpy as np
import librosa
import inspect
from pathlib import Path

from torch.utils.data import Subset

from src.predictor import predict_frames, SR, HOP
from src.trainer import artist_based_split  # da bismo koristili isti split kao u trainer-u

# mir_eval je opciono (za interval-based weighted_accuracy)
try:
    import mir_eval
    HAVE_MIR_EVAL = True
except Exception:
    HAVE_MIR_EVAL = False


# --------- pomoćnici: parsiranje labela, mape root/quality, čuvanje CSV ---------

ROOTS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','N']  # 12 nota + 'N'
QUALS = ['maj','min','7','maj7','min7','N']                        # 5 kvaliteta + 'N'

def label_to_root_quality(label: str):
    """ 'C:maj' -> ('C','maj'); 'N' -> ('N','N') """
    if label == 'N' or label is None or label == '':
        return 'N', 'N'
    if ':' in label:
        r, q = label.split(':', 1)
        return r, q
    return label, 'maj'

def indices_to_roots_quals(indices, chord_labels):
    roots, quals = [], []
    for i in indices:
        idx = int(i)
        label = chord_labels[idx] if 0 <= idx < len(chord_labels) else 'N'
        r, q = label_to_root_quality(label)
        roots.append(r if r in ROOTS else 'N')
        quals.append(q if q in QUALS else 'N')
    return roots, quals

def save_matrix_csv(matrix, row_labels, col_labels, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([''] + list(col_labels))
        for rlab, row in zip(row_labels, matrix):
            writer.writerow([rlab] + list(map(int, row)))

def save_rows_csv(rows, fieldnames, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# --------- pomoćnik: predikcioni frame indeksi u intervale ---------

def _indices_to_intervals_and_labels(frame_times, idx_seq, chord_labels):
    """ Spaja uzastopne iste indekse u [start, end) sa chord string labelom. """
    if len(idx_seq) == 0:
        return np.zeros((0, 2), dtype=float), []

    intervals, labels = [], []
    current = int(idx_seq[0])
    start_t = float(frame_times[0]) if len(frame_times) else 0.0

    for i in range(1, len(idx_seq)):
        if int(idx_seq[i]) != current:
            end_t = float(frame_times[i])
            intervals.append([start_t, end_t])
            labels.append(chord_labels[current])
            current = int(idx_seq[i])
            start_t = float(frame_times[i])

    end_t = float(frame_times[-1]) + (HOP / SR)
    intervals.append([start_t, end_t])
    labels.append(chord_labels[current])

    return np.asarray(intervals, dtype=float), list(labels)


def _call_mir_eval_weighted_accuracy(ref_intervals, ref_labels, est_intervals, est_labels):
    """
    Robustno poziva mir_eval.chord.weighted_accuracy bez obzira na verziju.
    """
    if not HAVE_MIR_EVAL:
        return None
    try:
        fn = mir_eval.chord.weighted_accuracy

        ref_intervals = np.asarray(ref_intervals, dtype=float).reshape(-1, 2)
        est_intervals = np.asarray(est_intervals, dtype=float).reshape(-1, 2)
        ref_labels = list(ref_labels)
        est_labels = list(est_labels)

        try:
            return float(fn(ref_intervals, ref_labels, est_intervals, est_labels))  # 4-arg
        except TypeError:
            return float(fn((ref_intervals, ref_labels), (est_intervals, est_labels)))  # 2-arg
    except Exception as e:
        print(f"[WARN] mir_eval failed: {e}")
        return None


# --------- glavni reporter ---------

def evaluate_and_report(dataset_instance, model_path, device, out_dir="reports", use_hmm=True, seed=42):
    """
    Radi evaluaciju na TEST splitu (isti artist-based split), i snima:
      - per_track.csv + per_track.json
      - summary.json
      - confusion_root.csv  (13x13, uključuje 'N' kao root)
      - confusion_quality.csv (6x6, uključuje 'N' kao kvalitet)
    Vraća dict sa putevima do fajlova i agregatnim metrikama.
    """
    # 1) rekonstruiši test split
    train_idx, val_idx, test_idx, _ = artist_based_split(dataset_instance, seed=seed)
    test_ds = Subset(dataset_instance, test_idx)

    frame_sec = HOP / float(SR)
    chord_labels = dataset_instance.chord_labels

    # confusion akumulatori
    root_index = {r: i for i, r in enumerate(ROOTS)}
    qual_index = {q: i for i, q in enumerate(QUALS)}
    root_conf = np.zeros((len(ROOTS), len(ROOTS)), dtype=np.int64)
    qual_conf = np.zeros((len(QUALS), len(QUALS)), dtype=np.int64)

    per_track_rows = []
    mir_scores = []

    original_ids = dataset_instance.track_ids
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[reporter] Evaluating and reporting {len(test_idx)} TEST tracks...")

    for local_i in range(len(test_ds)):
        orig_index = test_idx[local_i]
        tid = original_ids[orig_index]
        track = dataset_instance.dataset.track(tid)
        audio_path = track.audio_mix_path

        # predikcija frame-by-frame
        try:
            frame_times, est_idx, _ = predict_frames(
                audio_path, model_path, dataset_instance, device, use_hmm=use_hmm
            )
        except Exception as e:
            print(f"[WARN] Skip {tid}: prediction failed ({e})")
            continue

        T = len(frame_times)
        if T == 0:
            print(f"[WARN] Skip {tid}: zero frames.")
            continue

        # ref frame-wise iz intervala
        ref_idx = np.zeros(T, dtype=np.int64)
        chord_data = track.leadsheet_chords
        if chord_data and len(chord_data.labels) > 0:
            intervals = chord_data.intervals
            labels = chord_data.labels
            for (start, end), lab in zip(intervals, labels):
                if end <= 0:
                    continue
                s = max(0, librosa.time_to_frames(start, sr=SR, hop_length=HOP))
                e = max(s + 1, librosa.time_to_frames(end, sr=SR, hop_length=HOP))
                if s >= T:
                    continue
                e = min(e, T)
                cls = dataset_instance._chord_to_index(lab)
                ref_idx[s:e] = cls

        # frame-wise metrike
        correct = (ref_idx == est_idx)
        csr = float(np.sum(correct)) / float(T)
        wcsr = csr
        overlap = csr

        # dodatne info
        pct_N_ref = float(np.sum(ref_idx == 0)) / float(T)
        pct_N_est = float(np.sum(est_idx == 0)) / float(T)
        duration_sec = float(T * frame_sec)

        # mir_eval interval-based
        mir_weighted = None
        if HAVE_MIR_EVAL and chord_data and len(chord_data.labels) > 0:
            gt_intervals_sec = chord_data.intervals
            gt_labels_raw = chord_data.labels
            norm_ref_labels = []
            for lab in gt_labels_raw:
                idx = dataset_instance._chord_to_index(lab)
                norm_ref_labels.append(chord_labels[idx])

            est_intervals, est_lab = _indices_to_intervals_and_labels(frame_times, est_idx, chord_labels)
            mir_weighted = _call_mir_eval_weighted_accuracy(
                ref_intervals=gt_intervals_sec,
                ref_labels=norm_ref_labels,
                est_intervals=est_intervals,
                est_labels=est_lab,
            )
            if mir_weighted is not None:
                mir_scores.append(mir_weighted)

        # upiši per-track red
        per_track_rows.append({
            "track_id": tid,
            "frames": T,
            "duration_sec": round(duration_sec, 3),
            "CSR": round(csr, 6),
            "WCSR": round(wcsr, 6),
            "Overlap": round(overlap, 6),
            "mir_eval_weighted_accuracy": (round(float(mir_weighted), 6) if mir_weighted is not None else None),
            "pct_N_ref": round(pct_N_ref, 6),
            "pct_N_est": round(pct_N_est, 6),
        })

        # update confusion (po root i po kvalitetu)
        ref_roots, ref_quals = indices_to_roots_quals(ref_idx, chord_labels)
        est_roots, est_quals = indices_to_roots_quals(est_idx, chord_labels)
        for rr, er in zip(ref_roots, est_roots):
            root_conf[root_index[rr], root_index[er]] += 1
        for rq, eq in zip(ref_quals, est_quals):
            qual_conf[qual_index[rq], qual_index[eq]] += 1

    # agregati
    if len(per_track_rows) == 0:
        print("[reporter] No test tracks evaluated.")
        return None

    mean_csr = float(np.mean([r["CSR"] for r in per_track_rows]))
    mean_wcsr = float(np.mean([r["WCSR"] for r in per_track_rows]))
    mean_overlap = float(np.mean([r["Overlap"] for r in per_track_rows]))
    mean_mir = (float(np.mean(mir_scores)) if len(mir_scores) > 0 else None)

    # snimi per-track CSV/JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    per_track_csv = out_dir / "per_track.csv"
    per_track_json = out_dir / "per_track.json"
    save_rows_csv(
        per_track_rows,
        fieldnames=["track_id", "frames", "duration_sec", "CSR", "WCSR", "Overlap", "mir_eval_weighted_accuracy", "pct_N_ref", "pct_N_est"],
        path=per_track_csv,
    )
    with per_track_json.open('w', encoding='utf-8') as f:
        json.dump(per_track_rows, f, ensure_ascii=False, indent=2)

    # snimi confusion matrice
    root_csv = out_dir / "confusion_root.csv"
    qual_csv = out_dir / "confusion_quality.csv"
    save_matrix_csv(root_conf, ROOTS, ROOTS, root_csv)
    save_matrix_csv(qual_conf, QUALS, QUALS, qual_csv)

    # summary
    summary = {
        "N_tracks": len(per_track_rows),
        "CSR": mean_csr,
        "WCSR": mean_wcsr,
        "Overlap": mean_overlap,
        "mir_eval_weighted_accuracy": mean_mir,
        "frame_sec": frame_sec,
        "files": {
            "per_track_csv": str(per_track_csv),
            "per_track_json": str(per_track_json),
            "confusion_root_csv": str(root_csv),
            "confusion_quality_csv": str(qual_csv),
        },
    }
    summary_json = out_dir / "summary.json"
    with summary_json.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[reporter] === SUMMARY ===")
    print(f"Tracks: {summary['N_tracks']}")
    print(f"CSR:    {summary['CSR']*100:.2f}%")
    print(f"WCSR:   {summary['WCSR']*100:.2f}%")
    print(f"Overlap:{summary['Overlap']*100:.2f}%")
    if mean_mir is not None:
        print(f"mir_eval Weighted Accuracy: {mean_mir*100:.2f}%")
    else:
        if not HAVE_MIR_EVAL:
            print("mir_eval not installed; mir_eval metric skipped.")
        else:
            print("mir_eval available, but weighted_accuracy failed (see WARN lines).")

    return summary
