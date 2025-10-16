import mirdata
import librosa
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import os
from collections import Counter

class GuitarSetDataset(Dataset):
    def __init__(
        self,
        data_dir,
        audio_folder="audio_mono-pickup_mix",
        sr=22050,
        duration=None,         
        hop_length=512,
        augment=False,
        beat_sync=True,        
        time_stretch_prob=0.0, 
        max_seconds=8
    ):
        self.data_dir = data_dir
        self.audio_folder = os.path.join(data_dir, audio_folder)
        self.dataset = mirdata.initialize('guitarset', data_home=data_dir)
        self.track_ids = self.dataset.track_ids
        self.sr = sr
        self.duration = duration
        self.hop_length = hop_length
        self.MAX_FRAMES = 128
        self.augment = augment
        self.beat_sync = beat_sync
        self.time_stretch_prob = float(time_stretch_prob)
        self.max_seconds = max_seconds

        print(f"Found {len(self.track_ids)} tracks in GuitarSet.")
        self.chord_labels = self._generate_chord_labels()

        # mapa za enharmoniku
        self._ENH = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}

    # ---------------------- VOKABULAR I NORMALIZACIJA ----------------------

    def _generate_chord_labels(self):
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        qualities = ['maj', 'min', '7', 'maj7', 'min7']
        labels = ['N']
        for n in notes:
            for q in qualities:
                labels.append(f"{n}:{q}")
        return labels

    def _normalize_root(self, root: str) -> str:
        return self._ENH.get(root, root)

    def _normalize_symbol(self, sym: str) -> str:
        """
        Ulaz GuitarSet npr: 'Db:maj7/3', 'G:sus4', 'N'
        Izlaz: normalizovani u vidu iz našeg vokabulara ili 'N'
        """
        if not sym or sym.upper() == 'N':
            return 'N'

        base = sym.split('/')[0]  # ukloni bas
        if ':' in base:
            root, qual = base.split(':', 1)
        else:
            root, qual = base, 'maj'

        root = self._normalize_root(root)
        q = qual.lower()

        if q in ['maj', 'min', '7', 'maj7', 'min7']:
            mapped_q = q
        elif q.startswith('min') or q in ['m', 'minor']:
            mapped_q = 'min'
        elif q.startswith('maj') or q in ['major', 'maj9', 'maj6']:
            mapped_q = 'maj'
        else:
            if 'dim' in q:
                mapped_q = 'min'
            elif 'aug' in q or 'sus' in q:
                mapped_q = 'maj'
            else:
                return 'N'

        cand = f"{root}:{mapped_q}"
        return cand if cand in self.chord_labels else 'N'

    def _chord_to_index(self, chord_name: str) -> int:
        normalized = self._normalize_symbol(chord_name)
        return self.chord_labels.index(normalized) if normalized in self.chord_labels else 0

    # ---------------------- AUGMENTACIJE ----------------------

    def _apply_augmentation(self, y):
        """
        Augmentacija bez zavisnosti od 'sklearn':
        - time-stretch (phase vocoder) -> ako zataji, preskoči
        - pitch-shift (resample trik)  -> bez eksternih zavisnosti
        - blagi šum
        """
        import numpy as np
        import librosa
        import random

        choice = random.choice(['none', 'pitch', 'noise'])

        # Opciono time-stretch (blag) – probaj, a ako pukne, preskoči
        if random.random() < self.time_stretch_prob:
            try:
                rate = random.uniform(0.9, 1.1)
                # Phase-vocoder varijanta bez sklearn:
                D = librosa.stft(y, n_fft=2048, hop_length=512)
                D_stretched = librosa.phase_vocoder(D, rate=rate, hop_length=512)
                y = librosa.istft(D_stretched, hop_length=512, length=int(len(y) / rate))
            except Exception as e:
                print(f"Time-stretch fallback failed: {e}")

        # Ograniči trajanje na ~15s (kao i ranije)
        max_len = int(self.sr * 15)
        if len(y) > max_len:
            y = y[:max_len]

        if choice == 'pitch':
            # Pitch-shift bez sklearn: resample trik (±1–2 polustepena)
            n_steps = random.choice([-2, -1, 1, 2])
            try:
                factor = 2.0 ** (n_steps / 12.0)
                y_fast = librosa.resample(y, orig_sr=self.sr, target_sr=int(self.sr * factor))
                # Vrati nazad na originalni SR da zadržiš dužinu signala
                y = librosa.resample(y_fast, orig_sr=int(self.sr * factor), target_sr=self.sr)
            except Exception as e:
                print(f"Pitch-shift fallback failed: {e}, vraćam originalan signal.")
        elif choice == 'noise':
            noise = np.random.randn(len(y)) * 0.005
            y = y + noise

        return y

    # ---------------------- PYTORCH HOOKOVI ----------------------

    def __len__(self):
        # Jedan sample po tracku po epohi (svaki put nasumičan prozor)
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        track = self.dataset.track(track_id)
        audio_path = track.audio_mix_path

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio fajl {track_id} nije pronađen: {audio_path}")

        y, sr = librosa.load(
            audio_path,
            sr=self.sr,
            mono=True,
            duration=self.duration
        )

        max_len = int(self.sr * self.max_seconds)
        if len(y) > max_len:
            y = y[:max_len]

        if self.augment:
            y = self._apply_augmentation(y)

        # --- FIČR: frame-wise chroma ---
        chroma_f = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=self.hop_length)  # (12, T_f)
        chroma_f = librosa.util.normalize(chroma_f, axis=1)
        T_f = chroma_f.shape[1]
        if T_f < 1:
            raise RuntimeError(f"Prazan hromagram za {track_id}")

        # --- GT frame-wise (služi i za beat agregaciju) ---
        labels_f = np.zeros(T_f, dtype=np.int64)  # default N
        chord_data = track.leadsheet_chords
        try:
            intervals = chord_data.intervals  # (N,2) sekunde
            labels = chord_data.labels        # list of strings
        except Exception:
            intervals, labels = None, None

        if intervals is not None and labels:
            for (start, end), lab in zip(intervals, labels):
                if end <= 0:
                    continue
                s = max(0, librosa.time_to_frames(start, sr=self.sr, hop_length=self.hop_length))
                e = max(s+1, librosa.time_to_frames(end,   sr=self.sr, hop_length=self.hop_length))
                if s >= T_f:
                    continue
                e = min(e, T_f)
                cls_idx = self._chord_to_index(lab)
                labels_f[s:e] = cls_idx

        # --- Pokušaj beat-sync ako je traženo ---
        beats_obj = getattr(track, "beats", None)
        have_beats = False
        beat_frames = None
        if self.beat_sync and beats_obj is not None:
            try:
                beat_times = beats_obj.times  # sekunde
                if beat_times is not None and len(beat_times) >= 2:
                    bf = librosa.time_to_frames(beat_times, sr=self.sr, hop_length=self.hop_length)
                    bf = bf[(bf >= 0) & (bf <= T_f)]
                    if len(bf) >= 2:
                        beat_frames = bf
                        have_beats = True
            except Exception:
                have_beats = False

        if have_beats:
            # --- Beat-sync FIČRE ---
            chroma_sync = librosa.util.sync(chroma_f, beat_frames, aggregate=np.median)  # (12, T_b)
            chroma_b = chroma_sync
            T_b = chroma_b.shape[1]

            # --- Beat-sync LABELI (moda frame-labela unutar svakog beat segmenta) ---
            labels_b = np.zeros(T_b, dtype=np.int64)
            for i in range(T_b):
                s = beat_frames[i] if i < len(beat_frames) else 0
                e = beat_frames[i+1] if i+1 < len(beat_frames) else T_f
                s = max(0, min(s, T_f - 1))
                e = max(s + 1, min(e, T_f))
                seg = labels_f[s:e]
                if seg.size == 0:
                    labels_b[i] = 0
                else:
                    counts = np.bincount(seg, minlength=len(self.chord_labels))
                    labels_b[i] = int(np.argmax(counts))

            # --- Prozor od 128 beat-koraka ---
            if T_b <= self.MAX_FRAMES:
                rep = int(np.ceil(self.MAX_FRAMES / T_b))
                chroma_b_expanded = np.repeat(chroma_b, repeats=rep, axis=1)[:, :self.MAX_FRAMES]  # (12,128)
                labels_b_expanded = np.repeat(labels_b, repeats=rep)[:self.MAX_FRAMES]             # (128,)
                x = torch.tensor(chroma_b_expanded, dtype=torch.float32).unsqueeze(0)              # (1,12,128)
                window_labels = labels_b_expanded
            else:
                start_b = random.randint(0, T_b - self.MAX_FRAMES)
                end_b = start_b + self.MAX_FRAMES
                x = torch.tensor(chroma_b[:, start_b:end_b], dtype=torch.float32).unsqueeze(0)     # (1,12,128)
                window_labels = labels_b[start_b:end_b]

            counts = np.bincount(window_labels, minlength=len(self.chord_labels))
            y_idx = int(np.argmax(counts))
            y_tensor = torch.tensor(y_idx, dtype=torch.long)
            return x, y_tensor

        # ---------------- Fallback: frame-wise (stari put) ----------------
        if T_f <= self.MAX_FRAMES:
            x = torch.tensor(chroma_f, dtype=torch.float32).unsqueeze(0)  # (1,12,T_f)
            pad = self.MAX_FRAMES - T_f
            x = torch.nn.functional.pad(x, (0, pad))
            window_labels = labels_f
        else:
            start = random.randint(0, T_f - self.MAX_FRAMES)
            end = start + self.MAX_FRAMES
            x = torch.tensor(chroma_f[:, start:end], dtype=torch.float32).unsqueeze(0)  # (1,12,128)
            window_labels = labels_f[start:end]

        if len(window_labels) == 0:
            y_idx = 0
        else:
            counts = np.bincount(window_labels, minlength=len(self.chord_labels))
            y_idx = int(np.argmax(counts))

        y_tensor = torch.tensor(y_idx, dtype=torch.long)
        return x, y_tensor

    # ---------------------- HMM POMOĆNE FUNKCIJE (OSTAVLJENE) ----------------------

    def get_all_chord_transitions(self):
        all_indices = []
        print("Extracting full chord sequences for HMM training...")

        for track_id in self.track_ids:
            track = self.dataset.track(track_id)
            chord_data = track.leadsheet_chords

            if chord_data and len(chord_data.labels) > 0:
                for chord_symbol in chord_data.labels:
                    index = self._chord_to_index(chord_symbol)
                    all_indices.append(index)

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
