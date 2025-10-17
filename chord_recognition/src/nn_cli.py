from __future__ import annotations

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning
)

import argparse
from pathlib import Path
from typing import Dict
import json

from utils import OUTPUT_DIR, find_pairs
from data_utils import load_dataset, split_by_artist, extract_pair

from nn_runner import (
    train as nn_train,
    eval_split as nn_eval_split,
    predict_on_pair as nn_predict_on_pair,
    predict_on_path as nn_predict_on_path,
    model_paths as nn_model_path,
)

USE_AUG = True
MAX_TRAIN_SONGS = 0

def _build_train_lists(train_idx, pairs):
    from tqdm import tqdm
    X_list, Y_list = [], []
    idxs = train_idx[:MAX_TRAIN_SONGS] if MAX_TRAIN_SONGS else train_idx
    for i in tqdm(idxs, desc="Preparing"):
        jp, wp = pairs[i]
        bs = extract_pair(jp, wp, cache=True)
        X_list.append(bs.X); Y_list.append(bs.y)
    return X_list, Y_list

def cmd_train(kind: str):
    pairs = find_pairs()
    if not pairs:
        print("No data found."); return 1
    samples, artist_map = load_dataset()
    splits = split_by_artist(artist_map, ratios=(0.7,0.15,0.15))
    X_list, Y_list = _build_train_lists(splits["train"], pairs)
    nn_train(kind, X_list, Y_list, epochs=15, batch_size=4, lr=1e-3)
    return 0

def cmd_eval(kind: str):
    pairs = find_pairs()
    if not pairs:
        print("No data found."); return 1
    samples, artist_map = load_dataset()
    splits = split_by_artist(artist_map, ratios=(0.7,0.15,0.15))
    mpath = nn_model_path(kind)
    if not mpath.exists():
        print(f"{kind} model not found. Train first."); return 2
    res: Dict[str,Dict[str,float]] = {}
    for split in ["train","val","test"]:
        m = nn_eval_split(kind, mpath, samples, splits[split])
        if m: res[split] = m
    print(f"=== {kind.upper()} Evaluation ===")
    for split, m in res.items():
        print(f"{split:>6}: CSR={m['CSR']:.3f} | WCSR={m['WCSR']:.3f} | Overlap={m['Overlap']:.3f}")
    out = OUTPUT_DIR / f"eval_{kind}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
    return 0

def cmd_predict_pair(kind: str):
    pairs = find_pairs()
    if not pairs:
        print("No data found."); return 1
    samples, artist_map = load_dataset()
    splits = split_by_artist(artist_map, ratios=(0.7,0.15,0.15))
    idx = (splits.get("test") or [0])[0] if (splits.get("test") and len(splits["test"])>0) else 0
    mpath = nn_model_path(kind)
    if not mpath.exists():
        print(f"{kind} model not found. Train first."); return 2
    jp, wp = pairs[idx]
    from data_utils import extract_pair
    sample = extract_pair(jp, wp, cache=True)
    out = nn_predict_on_pair(kind, mpath, sample, out_name=f"{wp.stem}_{kind}_pred.txt")
    print(f"Saved: {out}")
    return 0

def cmd_predict_path(kind: str, path: Path):
    mpath = nn_model_path(kind)
    if not mpath.exists():
        print(f"{kind} model not found. Train first."); return 2
    out = nn_predict_on_path(kind, mpath, path)
    print(f"Saved: {out}")
    return 0

def main():
    parser = argparse.ArgumentParser(prog="nn_cli", description="NN helper (separate process)")
    parser.add_argument("command", choices=["train","eval","predict_pair","predict_path"])
    parser.add_argument("--kind", required=True, choices=["cnn","lstm","cnn_lstm","cnn+lstm","cnnlstm"])
    parser.add_argument("--path", type=str, help="audio file path for predict_path")
    args = parser.parse_args()

    kind = args.kind.lower()
    if kind == "cnn+lstm": kind = "cnn_lstm"
    if args.command == "train":
        return cmd_train(kind)
    elif args.command == "eval":
        return cmd_eval(kind)
    elif args.command == "predict_pair":
        return cmd_predict_pair(kind)
    elif args.command == "predict_path":
        if not args.path: print("--path required"); return 3
        return cmd_predict_path(kind, Path(args.path))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
