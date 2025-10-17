from __future__ import annotations
from pathlib import Path
from typing import List
from tqdm import tqdm
import json

from utils import find_pairs, OUTPUT_DIR, MODELS_DIR, PRED_DIR
from data_utils import load_dataset, split_by_artist, extract_pair, extract_pair_with_augment, BeatSample
from hmm_baseline import train_hmm, decode_song, save_model as save_hmm, load_model as load_hmm
from metrics import metrics_dict, beat_weights
from chord_vocab import id_to_chord_label
from feature_extraction import load_audio, chroma_cqt, beat_sync, SR

import sys, subprocess
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent

MODEL_PATH_HMM = MODELS_DIR / "hmm_baseline.pkl"

USE_AUG = True
MAX_TRAIN_SONGS = 0

def action_train_hmm():
    pairs = find_pairs()
    if not pairs:
        print("No data found.\n  Looked in: data/annotation and data/audio_mono-pickup_mix")
        return

    print(f"Found {len(pairs)} items. Building dataset...")
    samples, artist_map = load_dataset()
    if not samples:
        print("Dataset empty after loading. Check your file names.")
        return

    splits = split_by_artist(artist_map, ratios=(0.7, 0.15, 0.15))
    train_idx = splits["train"]; val_idx = splits["val"]; test_idx = splits["test"]
    print(f"Split by artist -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    idxs = train_idx[:MAX_TRAIN_SONGS] if MAX_TRAIN_SONGS else train_idx
    train_X, train_y = [], []
    print("Preparing training data..." + (" (NO augmentations)" if not USE_AUG else " (with augmentations)"))
    for i in tqdm(idxs, desc="Preparing"):
        jp, wp = pairs[i]
        if USE_AUG:
            for bs in extract_pair_with_augment(jp, wp):
                train_X.append(bs.X); train_y.append(bs.y)
        else:
            bs = extract_pair(jp, wp, cache=True)
            train_X.append(bs.X); train_y.append(bs.y)

    print(f"Total training sequences: {len(train_X)}")
    print("Training HMM...")
    model = train_hmm(train_X, train_y)
    path = save_hmm(model)
    print(f"Model saved to: {path}")

def _eval_split_hmm(model, samples: List[BeatSample], indices: List[int], name: str):
    if not indices:
        return None
    metrics_agg = {"CSR": 0.0, "WCSR": 0.0, "Overlap": 0.0}
    tot_w = 0.0
    for i in tqdm(indices, desc=f"Evaluating {name}"):
        s = samples[i]
        y_pred = decode_song(model, s.X)
        m = metrics_dict(s.y, y_pred, s.boundaries)
        w = beat_weights(s.boundaries).sum()
        tot_w += w
        for k in metrics_agg:
            metrics_agg[k] += m[k] * w
    for k in metrics_agg:
        metrics_agg[k] = metrics_agg[k] / max(tot_w, 1e-8)
    return metrics_agg

def action_evaluate_hmm():
    if not MODEL_PATH_HMM.exists():
        print("Model not found. Please run 'Train' first.")
        return

    print("Loading dataset and model...")
    samples, artist_map = load_dataset()
    if not samples:
        print("No data to evaluate.")
        return

    splits = split_by_artist(artist_map, ratios=(0.7, 0.15, 0.15))
    model = load_hmm(MODEL_PATH_HMM)

    res = {}
    for split in ["train", "val", "test"]:
        metrics = _eval_split_hmm(model, samples, splits[split], split)
        if metrics is not None:
            res[split] = metrics

    print("=== Evaluation (duration-weighted) ===")
    for split, m in res.items():
        print(f"{split:>6}: CSR={m['CSR']:.3f} | WCSR={m['WCSR']:.3f} | Overlap={m['Overlap']:.3f}")

    out = OUTPUT_DIR / "eval_hmm.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Saved: {out}")

def action_predict_paired_hmm():
    pairs = find_pairs()
    if not pairs:
        print("No data found for prediction.")
        return

    samples, artist_map = load_dataset()
    splits = split_by_artist(artist_map, ratios=(0.7, 0.15, 0.15)) if samples else {"test": [], "train": [], "val": []}
    target_idx = splits.get("test", [])
    idx = target_idx[0] if target_idx else 0

    if not MODEL_PATH_HMM.exists():
        print("Model not found. Please train first.")
        return

    model = load_hmm(MODEL_PATH_HMM)
    jams_path, wav_path = pairs[idx]
    sample = extract_pair(jams_path, wav_path, cache=True)
    y_pred = decode_song(model, sample.X)

    lines = []
    for b in range(len(y_pred)):
        t0, t1 = sample.boundaries[b], sample.boundaries[b+1]
        chord = id_to_chord_label(int(y_pred[b]))
        lines.append(f"{t0:.3f}\t{t1:.3f}\t{chord}")
    out_path = PRED_DIR / f"{wav_path.stem}_pred.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")

def action_predict_from_path_hmm():
    if not MODEL_PATH_HMM.exists():
        print("Model not found. Please train first.")
        return

    path_str = input("Enter path to audio file (e.g., C:\\path\\file.wav or .mp3): ").strip().strip('"')
    if not path_str:
        print("No path provided."); return
    p = Path(path_str)
    if not p.exists():
        print(f"File not found: {p}"); return

    y, sr = load_audio(str(p), sr=SR)
    C = chroma_cqt(y, sr)
    X_sync, boundaries = beat_sync(C, y, sr)

    model = load_hmm(MODEL_PATH_HMM)
    y_pred = decode_song(model, X_sync)

    lines = []
    for b in range(min(len(y_pred), len(boundaries) - 1)):
        t0, t1 = boundaries[b], boundaries[b+1]
        chord = id_to_chord_label(int(y_pred[b]))
        lines.append(f"{t0:.3f}\t{t1:.3f}\t{chord}")
    out_path = PRED_DIR / f"{p.stem}_inference_hmm.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")

def _run_nn_cli(args_list):
    cmd = [sys.executable, str(SRC_DIR / "nn_cli.py")] + args_list
    try:
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print(f"NN process exited with code {ret.returncode}")
    except Exception as e:
        print("Failed to run NN subprocess:", e)

def _submenu_nn(kind: str):
    kind_key = kind.lower()
    while True:
        print("\n" + "="*50)
        print(f"{kind.upper()} submenu")
        print("="*50)
        print("  1) Train")
        print("  2) Evaluate")
        print("  3) Predict (paired)")
        print("  4) Predict (audio path)")
        print("  0) Back\n")
        choice = input("Choose [1/2/3/4/0]: ").strip()

        if choice == "1":
            _run_nn_cli(["train", "--kind", kind_key])

        elif choice == "2":
            _run_nn_cli(["eval", "--kind", kind_key])

        elif choice == "3":
            _run_nn_cli(["predict_pair", "--kind", kind_key])

        elif choice == "4":
            path_str = input("Enter path to audio file: ").strip().strip('"')
            if not path_str:
                print("No path provided."); continue
            _run_nn_cli(["predict_path", "--kind", kind_key, "--path", path_str])

        elif choice == "0":
            break
        else:
            print("Invalid choice.")

def menu():
    while True:
        print("="*64)
        print("   CHORD RECOGNITION — Baseline + NN")
        print("="*64)
        print("Main menu:")
        print("  1) HMM submenu")
        print("  2) CNN submenu")
        print("  3) LSTM submenu")
        print("  4) CNN+LSTM submenu")
        print("  0) Exit\n")
        choice = input("Enter your choice [1/2/3/4/0]: ").strip()

        if choice == "1":
            while True:
                print("\n" + "="*50)
                print("HMM submenu")
                print("="*50)
                print("  1) Train")
                print("  2) Evaluate")
                print("  3) Predict (paired)")
                print("  4) Predict (audio path)")
                print("  0) Back\n")
                c2 = input("Choose [1/2/3/4/0]: ").strip()
                if c2 == "1":
                    action_train_hmm()
                elif c2 == "2":
                    action_evaluate_hmm()
                elif c2 == "3":
                    action_predict_paired_hmm()
                elif c2 == "4":
                    action_predict_from_path_hmm()
                elif c2 == "0":
                    break
                else:
                    print("Invalid choice.")

        elif choice == "2":
            _submenu_nn("cnn")
        elif choice == "3":
            _submenu_nn("lstm")
        elif choice == "4":
            _submenu_nn("cnn_lstm")
        elif choice == "0":
            print("Bye.")
            break
        else:
            print("Invalid choice.")
