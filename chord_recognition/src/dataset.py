import mirdata
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class GuitarSetDataset(Dataset):
    def __init__(self, data_dir, audio_folder="audio_mono-pickup_mix", sr=22050, duration=10.0, hop_length=512):
        self.data_dir = data_dir
        self.audio_folder = os.path.join(data_dir, audio_folder)
        self.dataset = mirdata.initialize('guitarset', data_home=data_dir)
        self.track_ids = self.dataset.track_ids
        self.sr = sr
        self.duration = duration
        self.hop_length = hop_length
        self.MAX_FRAMES = 128

        print(f"Found {len(self.track_ids)} tracks in GuitarSet.")
        self.chord_labels = self._generate_chord_labels()

    def _generate_chord_labels(self):
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        qualities = ['maj', 'min', '7', 'maj7', 'min7']
        labels = ['N']
        for n in notes:
            for q in qualities:
                labels.append(f"{n}:{q}")
        return labels

    def _chord_to_index(self, chord_name):
        chord_name = chord_name.split('/')[0] 
        if chord_name in self.chord_labels:
            return self.chord_labels.index(chord_name)
        else:
            return 0

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        track = self.dataset.track(track_id)

        audio_path = track.audio_mix_path 
        
        if not os.path.exists(audio_path):
             raise FileNotFoundError(f"Audio fajl {track_id} nije pronađen na očekivanoj putanji: {audio_path}. Proverite da li je folder 'audio_mono-pickup_mix' ispravno preuzet i da li je mirdata indeks ispravan.")

        y, sr = librosa.load(audio_path, sr=self.sr, mono=True, duration=self.duration)

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        chroma = chroma.T

        chord_data = track.leadsheet_chords 
        
        if chord_data and len(chord_data.labels) > 0:
            first_chord_symbol = chord_data.labels[0]
            label = self._chord_to_index(first_chord_symbol)
        else:
            label = 0 
        
        num_frames = min(chroma.shape[0], self.MAX_FRAMES)
        
        x_tensor = torch.tensor(chroma[:num_frames, :], dtype=torch.float32).T.unsqueeze(0)
        
        padding = self.MAX_FRAMES - x_tensor.shape[2]
        x_tensor = torch.nn.functional.pad(x_tensor, (0, padding))
        
        y_tensor = torch.tensor(label, dtype=torch.long)

        return x_tensor, y_tensor
    
    def get_all_chord_transitions(self):
        all_indices = []
        print("Extracting full chord sequences for HMM training...")
        
        for track_id in self.track_ids:
            track = self.dataset.track(track_id)
            chord_data = track.leadsheet_chords
            
            if chord_data and len(chord_data.labels) > 0:
                track_indices = []
                
                for chord_symbol in chord_data.labels:
                    index = self._chord_to_index(chord_symbol)
                    track_indices.append(index)
                
                if len(track_indices) > 1:
                    all_indices.extend(track_indices)
        
        return np.array(all_indices, dtype=np.int32)

    def get_all_chord_transitions_by_track(self):
        all_track_sequences = []
        print("Extracting full chord sequences (track by track) for HMM training...")
        
        for track_id in self.track_ids:
            track = self.dataset.track(track_id)
            chord_data = track.leadsheet_chords
            
            if chord_data and len(chord_data.labels) > 0:
                track_indices = []
                
                for chord_symbol in chord_data.labels:
                    index = self._chord_to_index(chord_symbol)
                    track_indices.append(index)
                
                if len(track_indices) > 1:
                    all_track_sequences.append(track_indices)
        
        return all_track_sequences
