from src.dataset import GuitarSetDataset
from src.model import CNNChordNet
from src.trainer import run_training_evaluation
from src.predictor import predict_song
from src.hmm_utils import calculate_transition_matrix, HMM_TRANSITION_PATH 
import torch
import mirdata
import sys
import os
import logging

try:
    logging.getLogger('mirdata').setLevel(logging.ERROR)
except:
    pass

if __name__ == "__main__":
    sys.tracebacklimit = 0 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    MODEL_PATH = "cnn_chordnet_baseline.pth"
    EPOCHS = 5
    
    print("Initializing GuitarSet data...")
    mirdata.initialize('guitarset', data_home="data/guitarset")
    dataset = GuitarSetDataset(data_dir="data/guitarset", audio_folder="audio_mono-pickup_mix") 
    NUM_CLASSES = len(dataset.chord_labels)

    while True:
        # Fiksiran meni (bez duplikata)
        print("\n==============================================")
        print(" OPERATION SELECTION:")
        print("==============================================")
        print("1. Train and Evaluate CNN Model (Saves to cnn_chordnet_baseline.pth)")
        print(f"2. Train HMM Transition Matrix (Saves to {HMM_TRANSITION_PATH})")
        print("3. Predict Chords for a New Audio File (with HMM Smoothing)")
        print("4. Predict Chords for a New Audio File (RAW CNN)")
        print("5. Exit")
        print("==============================================")
        
        choice = input("Enter operation number: ")

        if choice == '1':
            run_training_evaluation(dataset, NUM_CLASSES, device, EPOCHS, MODEL_PATH)
        
        elif choice == '2':
            calculate_transition_matrix(dataset)

        elif choice == '3':
            audio_file = input("Enter the full path to the audio file (e.g. C:\\Users\\...\\my_song.mp3): ")
            
            if os.path.exists(audio_file):
                predict_song(audio_file, MODEL_PATH, dataset, device, use_hmm=True) 
            else:
                print(f"Error: File '{audio_file}' does not exist or the path is incorrect.")

        elif choice == '4':
            audio_file = input("Enter the full path to the audio file (e.g. C:\\Users\\...\\my_song.mp3): ")
            
            if os.path.exists(audio_file):
                predict_song(audio_file, MODEL_PATH, dataset, device, use_hmm=False)
            else:
                print(f"Error: File '{audio_file}' does not exist or the path is incorrect.")

        elif choice == '5':
            print("Exiting program. Goodbye!")
            break
        
        else:
            print("Invalid input. Please choose 1, 2, 3, 4 or 5.")