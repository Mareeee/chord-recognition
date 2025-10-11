import librosa
import numpy as np
import os

def extract_chromagram(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return chroma.T

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if fname.endswith(".wav") or fname.endswith(".mp3"):
            path = os.path.join(input_folder, fname)
            chroma = extract_chromagram(path)
            np.save(os.path.join(output_folder, fname + ".npy"), chroma)
            print(f"Processed {fname} -> shape {chroma.shape}")
