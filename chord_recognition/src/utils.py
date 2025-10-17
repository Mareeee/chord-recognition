from __future__ import annotations
import json
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
ANNO_DIR     = DATA_DIR / "annotation"
AUDIO_DIR    = DATA_DIR / "audio_mono-pickup_mix"

OUTPUT_DIR   = PROJECT_ROOT / "outputs"
CACHE_DIR    = OUTPUT_DIR / "cache"
MODELS_DIR   = OUTPUT_DIR / "models"
PRED_DIR     = OUTPUT_DIR / "predictions"
DOWNLOADS_DIR= PROJECT_ROOT / "downloads"

for d in [OUTPUT_DIR, CACHE_DIR, MODELS_DIR, PRED_DIR, DOWNLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ZENODO_BASE = "https://zenodo.org/records/3371780/files"
ANNOTATION_ZIP = "annotation.zip"
AUDIO_PICKUP_ZIP = "audio_mono-pickup.zip"
AUDIO_MIC_ZIP    = "audio_mono-mic.zip"

def _zenodo_url(name: str) -> str:
    return f"{ZENODO_BASE}/{name}?download=1"

def human_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_stem(stem: str) -> str:
    s = stem.strip()
    parts = s.split("_")
    SUFFIXES = {"mix", "mic", "pickup"}
    while parts and parts[-1].lower() in SUFFIXES:
        parts.pop()
    return "_".join(parts)

def _download(url: str, out_path: Path) -> bool:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {url}")
        urlretrieve(url, out_path)
        print(f"[ok] saved: {out_path}")
        return True
    except Exception as e:
        print(f"[warn] could not download {url}: {e}")
        return False

def _extract_zip(zip_path: Path, dst_dir: Path):
    print(f"[extract] {zip_path.name} -> {dst_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)

def _collect_and_copy(src_root: Path, pattern: str, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in src_root.rglob(pattern):
        if p.is_file():
            target = dst_dir / p.name
            if target.exists():
                continue
            shutil.copy2(p, target)
            count += 1
    return count

def ensure_guitarset_downloaded() -> None:
    have_annos = ANNO_DIR.exists() and any(ANNO_DIR.glob("*.jams"))
    have_audio = AUDIO_DIR.exists() and any(AUDIO_DIR.glob("*.wav"))

    if have_annos and have_audio:
        return

    print("\n== GuitarSet auto-setup ==")
    temp_root = DOWNLOADS_DIR / "_tmp_extract"
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    if not have_annos:
        ann_zip = DOWNLOADS_DIR / ANNOTATION_ZIP
        if not ann_zip.exists():
            _download(_zenodo_url(ANNOTATION_ZIP), ann_zip)
        if ann_zip.exists():
            _extract_zip(ann_zip, temp_root)
            copied = _collect_and_copy(temp_root, "*.jams", ANNO_DIR)
            print(f"[guitarset] copied {human_int(copied)} JAMS -> {ANNO_DIR}")
        else:
            print("[error] cannot fetch annotation.zip — check internet connection / firewall.")

    if not have_audio:
        wav_copied = 0

        au_zip = DOWNLOADS_DIR / AUDIO_PICKUP_ZIP
        if not au_zip.exists():
            ok = _download(_zenodo_url(AUDIO_PICKUP_ZIP), au_zip)
            if not ok:
                mic_zip = DOWNLOADS_DIR / AUDIO_MIC_ZIP
                if not mic_zip.exists():
                    _download(_zenodo_url(AUDIO_MIC_ZIP), mic_zip)
                if mic_zip.exists():
                    _extract_zip(mic_zip, temp_root)
                    wav_copied = _collect_and_copy(temp_root, "*.wav", AUDIO_DIR)
                else:
                    print("[error] cannot fetch audio_mono-mic.zip either.")
            else:
                _extract_zip(au_zip, temp_root)
                wav_copied = _collect_and_copy(temp_root, "*.wav", AUDIO_DIR)
        else:
            _extract_zip(au_zip, temp_root)
            wav_copied = _collect_and_copy(temp_root, "*.wav", AUDIO_DIR)

        print(f"[guitarset] copied {human_int(wav_copied)} WAV -> {AUDIO_DIR}")

    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    print("== GuitarSet setup: DONE ==\n")

def find_pairs(anno_dir: Path = ANNO_DIR, audio_dir: Path = AUDIO_DIR) -> List[Tuple[Path, Path]]:
    need_dl = (not anno_dir.exists() or not any(anno_dir.glob("*.jams"))
               or not audio_dir.exists() or not any(audio_dir.glob("*.wav")))
    if need_dl:
        ensure_guitarset_downloaded()

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
