# Chord Recognition

*Automatic chord recognition from mono guitar audio using a simple, student‑level pipeline.*

This project demonstrates a complete chord-recognition workflow on the **GuitarSet** dataset: loading and pairing audio with annotations, extracting **chroma CQT** features, **beat-synchronizing** them, and training/evaluating several baseline models (HMM and lightweight neural models).

---

## 1) What this project does

- **Input:** a `.wav`/`.mp3` guitar recording.
- **Output:** a time‑segmented sequence of chord labels (e.g., `t0 – t1 : C:maj`, `t1 – t2 : G:min`, …) from a compact **triad vocabulary** (major/minor + `N` = no chord).
- **Why it works:** chords most often change near beats; **chroma** features summarize harmonic content; simple sequence models are sufficient for a solid baseline.

**High‑level pipeline** (no code):
1. Load audio (mono) and normalize; resample to a fixed sample rate.
2. Extract **chroma CQT** (12 bins) over time.
3. Perform **beat tracking** and aggregate chroma between beat boundaries (beat‑sync).
4. Map JAMS chord annotations to the project’s chord vocabulary.
5. Train models (HMM/CNN/LSTM/CNN+LSTM) and evaluate with duration‑aware metrics.
6. Predict chords for a random song and format results as `[start, end, chord]` segments.

---

## 2) Dataset and folder layout

- **Dataset:** GuitarSet (Zenodo 3371780) with annotations in **JAMS** and audio in mono pickup/mic tracks.
- **Auto‑download:** if `data/annotation/*.jams` or `data/audio_mono-pickup_mix/*.wav` are missing, the project will **automatically download and unpack** the necessary archives into the expected folders (via `utils.py`). Pickup is preferred; mic is used as a fallback.
- **Expected structure:**
  - `data/annotation/*.jams`
  - `data/audio_mono-pickup_mix/*.wav`
- **Pairing rule:** files are matched by **normalized stem**; suffixes like `_mix`, `_mic`, `_pickup` are ignored so that annotations and audio align correctly.
- **Split:** **artist‑level** train/val/test split to avoid leaking performer style between splits.

> Acknowledgement: GuitarSet authors and Zenodo. Data used for **educational purposes**.

---

## 3) Features and preprocessing

- **Chroma CQT (12×T)**: captures pitch‑class energy; robust to timbre changes.
- **Per‑frame L1 normalization**: stabilizes scale for sequence models.
- **Beat synchronization (12×B)**: feature aggregation between beat boundaries; fallback to uniform segmentation if beat tracking fails.
- **Label mapping**: JAMS → triad vocabulary (major/minor) + `N`.

---

## 4) Models (baseline‑friendly)

- **HMM (Gaussian emissions)** — *baseline*  
  States = chord classes; emissions = Gaussian on 12‑D beat‑synced chroma; transitions learned from chord bigrams. Decoding via Viterbi.
- **CNN** — *local 2D patterns on chroma*  
  Learns short harmonic “fingerprints” on the chroma image; typically the most stable NN under short training.
- **LSTM (bi‑LSTM)** — *temporal dependencies*  
  Uses `pack_padded_sequence` to handle variable‑length sequences (ignores padding). Benefits from longer training/augmentations.
- **CNN+LSTM** — *hybrid*  
  CNN for local patterns → LSTM for longer‑range progressions.

---

## 5) Metrics and interpretation

- **CSR** — Chord Symbol Recall (per‑beat accuracy).  
- **WCSR** — duration‑weighted CSR (longer segments weigh more).  
- **Overlap** — duration‑weighted segment overlap.
- **How to read results:** for a shortened training run, an HMM around ~0.30 WCSR is a reasonable baseline. CNN often improves with more epochs/data. Focus on **relative** improvements and the **consistency** of predicted progressions.

---

## 6) Outputs

- `outputs/models/` — saved models.
- `outputs/predictions/` — text files with `[start, end, chord]` segments.
- `outputs/eval_*.json` — evaluation summaries for HMM and NN models.
- `outputs/cache/` — optional intermediate caches for faster iteration.

---

## 7) Installation (the only code you need here)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
python.exe src\main.py

```

**macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python.exe src\main.py

```

> If you are on Windows and see a PyTorch DLL warning, use the **CPU‑only** PyTorch wheel (already set in requirements for the student setup) and run NN training via the provided helper that spawns a clean subprocess.

---

## 8) How to use

- Launch the project and follow the **text menu** to: train, evaluate, and predict.  
- HMM works out of the box for quick baselines.  
- Neural models (CNN/LSTM/CNN+LSTM) are run through a small helper that starts them in a **separate process** for stability on Windows.

---

## 9) Limitations and next steps

- **Vocabulary** is compact (major/minor + `N`) for simplicity; can be extended (e.g., 7th chords) with re‑training.
- **Short training** → underfitting for NN models; improve with more epochs, data, and augmentations.
- **No heavy post‑processing**; could add smoothing with sequence models or heuristics.
- **Dataset bias** (guitar focus); cross‑instrument generalization would require broader data.

**Planned improvements**
- Longer training with **augmentations** (tempo/pitch‑shift).  
- Stronger class balancing or **focal loss** alternatives.  
- Richer label set and improved segment smoothing.  
- Lightweight UI for loading a custom track and visualizing predicted timeline.

---

## 10) Credits

- **Data:** GuitarSet (Zenodo 3371780).  
- **Libraries:** `librosa`, `jams`, `hmmlearn`, `scikit‑learn`, `PyTorch`, `numpy/scipy`, `matplotlib`.  
- **Use case:** Educational demo for chord recognition; not optimized for production.
