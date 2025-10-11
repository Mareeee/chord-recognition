import torch
import librosa
import numpy as np

def extract_features_for_prediction(audio_path, sr=22050, hop_length=512, max_frames=128):
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return torch.tensor([])

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma = chroma.T

    all_frames = []
    step = max_frames
    
    for start in range(0, chroma.shape[0], step):
        segment = chroma[start:start + max_frames, :]
        
        if segment.shape[0] < max_frames:
            padding = max_frames - segment.shape[0]
            segment = np.pad(segment, ((0, padding), (0, 0)), mode='constant')
        
        x_tensor = torch.tensor(segment, dtype=torch.float32).T.unsqueeze(0).unsqueeze(0)
        all_frames.append(x_tensor)
        
    if all_frames:
        return torch.cat(all_frames, dim=0)
    else:
        return torch.tensor([])