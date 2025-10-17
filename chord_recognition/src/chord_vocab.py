from __future__ import annotations
import re
from typing import Dict, List

PITCHES = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
ENHARMONIC = {
    "B#": "C", "Cb": "B",
    "E#": "F", "Fb": "E",
    "Db": "C#", "D#": "Eb",
    "Gb": "F#", "G#": "Ab",
    "A#": "Bb",
}
QUALITIES = ["maj", "min"]
N_SYMBOL = "N"

IDX_TO_LABEL: List[str] = []
for q in QUALITIES:
    for p in PITCHES:
        suffix = "" if q=="maj" else "m"
        IDX_TO_LABEL.append(p + suffix)
IDX_TO_LABEL.append(N_SYMBOL)
LABEL_TO_IDX: Dict[str,int] = {lab:i for i,lab in enumerate(IDX_TO_LABEL)}

ROOT_RE = re.compile(r"^([A-Ga-g])([#b]?)(.*)$")

def normalize_root(root: str) -> str:
    root = root.strip().upper()
    if len(root) >= 2 and root[:2] in ENHARMONIC:
        return ENHARMONIC[root[:2]]
    if root in ENHARMONIC:
        return ENHARMONIC[root]
    if root in PITCHES:
        return root
    if len(root)==1 and root in "ABCDEFG":
        return root
    return None

def reduce_quality(rest: str) -> str:
    rest = rest.strip().lower()
    if rest.startswith(("min","minor",":min",":m","m","-")):
        return "min"
    if "min" in rest or ":m" in rest:
        return "min"
    if "dim" in rest:
        return "min"
    return "maj"

def chord_label_to_id(label: str) -> int:
    if not label or label.upper() in ("N","NO CHORD","X"):
        return LABEL_TO_IDX[N_SYMBOL]
    m = ROOT_RE.match(label.strip())
    if not m:
        return LABEL_TO_IDX[N_SYMBOL]
    letter, accidental, rest = m.groups()
    root = normalize_root(letter.upper()+accidental)
    if root is None:
        return LABEL_TO_IDX[N_SYMBOL]
    q = reduce_quality(rest or "")
    triad = root + ("" if q=="maj" else "m")
    return LABEL_TO_IDX[triad]

def id_to_chord_label(idx: int) -> str:
    return IDX_TO_LABEL[idx]
