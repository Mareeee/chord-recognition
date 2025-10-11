import os
import numpy as np
from src.prediction_logic import extract_features_for_prediction
from src.hmm_utils import extract_raw_cnn_log_probabilities, viterbi_decoding, HMM_TRANSITION_PATH

def predict_song(audio_path, model_path, dataset_instance, device, use_hmm=False):
    
    if not os.path.exists(model_path):
        print(f"Error: Model '{model_path}' not found. Please run training first (Option 1).")
        return

    chord_labels = dataset_instance.chord_labels
    
    print(f"Processing audio file: {audio_path}...")
    
    try:
        log_probabilities = extract_raw_cnn_log_probabilities(audio_path, model_path, dataset_instance, device)
    except FileNotFoundError as e:
        print(e)
        return
    
    if log_probabilities.size == 0:
        print("Error: Could not extract features from audio file.")
        return

    predicted_indices = []
    
    if use_hmm and os.path.exists(HMM_TRANSITION_PATH):
        print("Applying HMM Viterbi decoding for sequence smoothing...")
        try:
            transition_matrix = np.load(HMM_TRANSITION_PATH)
            predicted_indices = viterbi_decoding(log_probabilities, transition_matrix)
        except Exception as e:
            print(f"Warning: Could not load or use HMM transition matrix: {e}. Falling back to RAW CNN.")
            predicted_indices = np.argmax(log_probabilities, axis=1).tolist()
    else:
        print("Predicting chords using raw CNN output (no smoothing)...")
        predicted_indices = np.argmax(log_probabilities, axis=1).tolist()

    predicted_chords = [chord_labels[idx] for idx in predicted_indices]

    if predicted_chords:
        current_chord = predicted_chords[0]
        count = 1
        
        print("\n--- PREDICTED CHORD SEQUENCE ---")
        
        for next_chord in predicted_chords[1:]:
            if next_chord == current_chord:
                count += 1
            else:
                time_approx = count * (128 * 512 / 22050)
                print(f"Chord: {current_chord} (Duration ~{time_approx:.1f}s)")
                current_chord = next_chord
                count = 1
        
        time_approx = count * (128 * 512 / 22050)
        print(f"Chord: {current_chord} (Duration ~{time_approx:.1f}s)")
        
        print("\nNote: Duration is an approximation based on 2.97s segments (128 frames at 512 hop length).")
        if use_hmm and os.path.exists(HMM_TRANSITION_PATH):
             print("Output smoothed using HMM Viterbi decoding.")
        else:
             print("Output based on independent CNN segment classification (can be noisy).")