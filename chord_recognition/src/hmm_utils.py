import os
import numpy as np
import torch
from src.model import CNNChordNet
from src.prediction_logic import extract_features_for_prediction

HMM_TRANSITION_PATH = "hmm_transition_matrix.npy"
HMM_PRIOR_PATH = "hmm_prior_probabilities.npy" 

def calculate_transition_matrix(dataset_instance, smoothing_factor=1e-6):
    """
    Ručno izračunava matricu tranzicije i inicijalne verovatnoće
    brojanjem akordskih promena direktno iz dataset anotacija.
    """
    print("Calculating true transition probabilities (Manual Count)...")
    
    chord_labels = dataset_instance.chord_labels
    num_classes = len(chord_labels)
    
    # 1. Priprema matrica za brojanje (Dodajemo smoothing_factor da izbegnemo 0 verovatnoće)
    transition_counts = np.full((num_classes, num_classes), smoothing_factor, dtype=np.float32)
    prior_counts = np.full(num_classes, smoothing_factor, dtype=np.float32)

    # 2. Ekstrakcija sekvence akorda
    labels_sequence_tracks = dataset_instance.get_all_chord_transitions_by_track()
    
    if not labels_sequence_tracks:
        print("Error: Could not extract chord sequences. Matrica tranzicije nije kreirana.")
        return False
    
    # 3. Brojanje tranzicija
    for track_indices in labels_sequence_tracks:
        if not track_indices:
            continue
            
        # Brojanje inicijalnih stanja (prvi akord u pesmi)
        prior_counts[track_indices[0]] += 1
        
        # Brojanje tranzicija (akord A u akord B)
        for i in range(len(track_indices) - 1):
            current_state = track_indices[i]
            next_state = track_indices[i+1]
            transition_counts[current_state, next_state] += 1
            
    # 4. Normalizacija u verovatnoće
    
    # Normalizacija tranzicija po redovima (suma reda mora biti 1)
    transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    
    # Normalizacija inicijalnih verovatnoća
    prior_probs = prior_counts / prior_counts.sum()

    # 5. Čuvanje rezultata
    np.save(HMM_TRANSITION_PATH, transition_matrix)
    np.save(HMM_PRIOR_PATH, prior_probs)
    
    print(f"HMM transition matrix saved to {HMM_TRANSITION_PATH}")
    print(f"HMM prior probabilities saved to {HMM_PRIOR_PATH}")
    return True

def viterbi_decoding(log_emission_probs, transition_matrix, prior_probs=None):
    
    # Koristimo prior probs iz fajla
    if prior_probs is None and os.path.exists(HMM_PRIOR_PATH):
        prior_probs = np.load(HMM_PRIOR_PATH)
    elif prior_probs is None:
        num_states = transition_matrix.shape[0]
        prior_probs = np.full(num_states, 1.0 / num_states)
        
    num_states = transition_matrix.shape[0]
    num_observations = log_emission_probs.shape[0]
    
    # ... (Ostatak Viterbi koda koji smo već implementirali je isti)
    log_prior = np.log(prior_probs)
    log_transition = np.log(transition_matrix)

    T1 = np.zeros((num_observations, num_states))
    T2 = np.zeros((num_observations, num_states), dtype=int)

    T1[0, :] = log_prior + log_emission_probs[0, :]

    for t in range(1, num_observations):
        for j in range(num_states):
            
            scores = T1[t-1, :] + log_transition[:, j]
            
            T1[t, j] = np.max(scores) + log_emission_probs[t, j]
            T2[t, j] = np.argmax(scores)

    path = np.zeros(num_observations, dtype=int)
    path[-1] = np.argmax(T1[-1, :])
    
    for t in range(num_observations - 2, -1, -1):
        path[t] = T2[t+1, path[t+1]]

    return path.tolist()

def extract_raw_cnn_log_probabilities(audio_path, model_path, dataset_instance, device):
   
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_path}' not found. Please run training first (Option 1).")

    num_classes = len(dataset_instance.chord_labels)
    model = CNNChordNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    inputs = extract_features_for_prediction(audio_path, max_frames=128)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        
    log_probs = torch.nn.functional.log_softmax(outputs, dim=1).cpu().numpy()
    
    return log_probs