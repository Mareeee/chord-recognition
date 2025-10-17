from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ANNO_DIR = DATA_DIR / "annotation"
AUDIO_DIR = DATA_DIR / "audio_mono-pickup_mix"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
MODELS_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "predictions"

for d in [OUTPUT_DIR, CACHE_DIR, MODELS_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def normalize_stem(stem: str) -> str:
    s = stem.strip()
    parts = s.split("_")
    SUFFIXES = {"mix", "mic", "pickup"}
    while parts and parts[-1].lower() in SUFFIXES:
        parts.pop()
    return "_".join(parts)


def find_pairs(anno_dir: Path = ANNO_DIR, audio_dir: Path = AUDIO_DIR) -> List[Tuple[Path, Path]]:
    if not anno_dir.exists() or not audio_dir.exists():
        return []
    audio_index = {}
    for p in audio_dir.glob("*.wav"):
        key = normalize_stem(p.stem)
        audio_index.setdefault(key, p)

    pairs = []
    for jams_path in anno_dir.glob("*.jams"):
        key = normalize_stem(jams_path.stem)
        wav = audio_index.get(key)
        if wav and wav.exists():
            pairs.append((jams_path, wav))
    return sorted(pairs)

def human_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
