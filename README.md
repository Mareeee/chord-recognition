# Chord Recognition (Baseline HMM)

This project follows your spec: load JAMS + WAV from `data/annotation` and `data/audio_mono-pickup_mix`,
extract beat-synchronized chroma-CQT features, reduce chord labels to 25 classes (12 major, 12 minor, N), 
train a supervised HMM baseline, evaluate (CSR, WCSR, Overlap), and predict chords for an example.

## Layout
```
chord_app/
  data/
    annotation/*.jams
    audio_mono-pickup_mix/*.wav
  src/
    main.py
    utils.py
    feature_extraction.py
    data_utils.py
    chord_vocab.py
    hmm_baseline.py
    metrics.py
  outputs/
    models/
    predictions/
    cache/
  requirements.txt
```

## Quick start
1. Put GuitarSet files into `data/annotation` and `data/audio_mono-pickup_mix` (matching stems).
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `python src/main.py`
4. Choose an action: Train / Evaluate / Predict

> No paths are requested at runtime; the app assumes the `data/` layout from your spec.
