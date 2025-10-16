import os
import numpy as np
import torch

from src.model import CNNChordNet
from src.prediction_logic import extract_features_for_prediction

HMM_TRANSITION_PATH = "hmm_transition_matrix.npy"
HMM_PRIOR_PATH = "hmm_prior_probabilities.npy"


def calculate_transition_matrix(dataset_instance, smoothing_factor=1e-6):
    """
    Računa matricu tranzicije i prior verovatnoće iz anotacija GuitarSet-a.
    Rezultate snima u .npy fajlove.
    """
    print("Calculating true transition probabilities (Manual Count)...")

    chord_labels = dataset_instance.chord_labels
    num_classes = len(chord_labels)

    transition_counts = np.full((num_classes, num_classes), smoothing_factor, dtype=np.float32)
    prior_counts = np.full(num_classes, smoothing_factor, dtype=np.float32)

    labels_sequence_tracks = dataset_instance.get_all_chord_transitions_by_track()
    if not labels_sequence_tracks:
        print("Error: Could not extract chord sequences. Matrica tranzicije nije kreirana.")
        return False

    for track_indices in labels_sequence_tracks:
        if not track_indices:
            continue
        prior_counts[track_indices[0]] += 1
        for i in range(len(track_indices) - 1):
            current_state = track_indices[i]
            next_state = track_indices[i + 1]
            transition_counts[current_state, next_state] += 1

    transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    prior_probs = prior_counts / prior_counts.sum()

    np.save(HMM_TRANSITION_PATH, transition_matrix)
    np.save(HMM_PRIOR_PATH, prior_probs)

    print(f"HMM transition matrix saved to {HMM_TRANSITION_PATH}")
    print(f"HMM prior probabilities saved to {HMM_PRIOR_PATH}")
    return True


def _apply_stickiness(T, alpha=0.95):
    """Pojačaj zadržavanje stanja:  T' = normalize( alpha*I + (1-alpha)*T )"""
    T = np.asarray(T, dtype=np.float64)
    n = T.shape[0]
    T_sticky = alpha * np.eye(n, dtype=np.float64) + (1.0 - alpha) * T
    T_sticky /= T_sticky.sum(axis=1, keepdims=True)
    return T_sticky.astype(np.float64)


def viterbi_decoding(log_emission_probs, transition_matrix, prior_probs=None, sticky_alpha=0.85):
    """
    Viterbi sa opcionalnim 'stickiness' pojačanjem dijagonale.
    log_emission_probs: (T, N)
    transition_matrix:  (N, N)
    prior_probs:        (N,)
    """
    T = np.asarray(log_emission_probs, dtype=np.float64)
    A = np.asarray(transition_matrix, dtype=np.float64)

    if prior_probs is None and os.path.exists(HMM_PRIOR_PATH):
        prior_probs = np.load(HMM_PRIOR_PATH)
    elif prior_probs is None:
        prior_probs = np.full(A.shape[0], 1.0 / A.shape[0], dtype=np.float64)

    # manje lepljivo po difoltu
    if sticky_alpha is not None and sticky_alpha > 0.0:
        A = _apply_stickiness(A, alpha=float(sticky_alpha))

    log_prior = np.log(np.asarray(prior_probs, dtype=np.float64) + 1e-12)
    log_A = np.log(A + 1e-12)

    T_len, N = T.shape
    if T_len == 0 or N == 0:
        return []

    DP = np.empty((T_len, N), dtype=np.float64)
    BP = np.empty((T_len, N), dtype=np.int32)

    DP[0, :] = log_prior + T[0, :]

    for t in range(1, T_len):
        prev = DP[t - 1, :].reshape(N, 1) + log_A  # (N, N)
        BP[t, :] = np.argmax(prev, axis=0)
        DP[t, :] = np.max(prev, axis=0) + T[t, :]

    path = np.empty(T_len, dtype=np.int32)
    path[-1] = int(np.argmax(DP[-1, :]))
    for t in range(T_len - 2, -1, -1):
        path[t] = BP[t + 1, path[t + 1]]
    return path.tolist()


def extract_raw_cnn_log_probabilities(
    audio_path, model_path, dataset_instance, device, overlap=0.5, seg_mode="beat_or_fixed", temperature=1.5
):
    """
    Izvlači log-probabilitete po segmentima iz treniranog CNN-a.
    Vraća (log_probs, meta) gde je meta dict sa segment granicama u FREJMOVIMA.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_path}' not found. Please run training first (Option 1).")

    num_classes = len(dataset_instance.chord_labels)
    model = CNNChordNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Manji prozor daje finiju “secku”
    inputs, meta = extract_features_for_prediction(
        audio_path,
        max_frames=96,              # <<— umesto 128 za finiju rezoluciju (~2.23s)
        overlap=overlap,
        return_meta=True,
        seg_mode=seg_mode,
    )
    if inputs is None or inputs.nelement() == 0:
        return np.zeros((0, num_classes), dtype=np.float32), meta

    inputs = inputs.to(device)
    with torch.no_grad():
        logits = model(inputs)

    T = float(temperature) if (temperature is not None and float(temperature) > 0) else 1.0
    logits = logits / T

    log_probs = torch.nn.functional.log_softmax(logits, dim=1).cpu().numpy()
    return log_probs, meta
