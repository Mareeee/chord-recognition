from src.dataset import GuitarSetDataset
from src.trainer import run_training_evaluation
from src.predictor import predict_song
from src.hmm_utils import calculate_transition_matrix, HMM_TRANSITION_PATH
from src.evaluator import evaluate_on_test

import torch
import mirdata
import sys
import os
import logging
import platform


def silence_all_logging():
    """
    mirdata u GuitarSet-u koristi root logger (logging.warning).
    Globalno utišamo logging tokom rada aplikacije.
    Ako želiš da ponovo uključiš logove, pozovi logging.disable(logging.NOTSET).
    """
    try:
        logging.disable(logging.CRITICAL)
    except Exception:
        pass


if __name__ == "__main__":
    # nemoj da zatrpava traceback-ovima CLI
    sys.tracebacklimit = 0

    # UGASI sve logove (rešava spam poruku iz mirdata)
    silence_all_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    EPOCHS = 5
    BATCH_SIZE = 32
    LR = 1e-3
    MODEL_PATH = "cnn_chordnet_baseline.pth"

    print("Initializing GuitarSet data...")
    mirdata.initialize("guitarset", data_home="data/guitarset")
    dataset = GuitarSetDataset(
        data_dir="data/guitarset",
        audio_folder="audio_mono-pickup_mix",
        beat_sync=False,
        augment=True,
        time_stretch_prob=0.10,
        max_seconds=8,
    )

    NUM_CLASSES = len(dataset.chord_labels)

    while True:
        print("\n==============================================")
        print(" OPERATION SELECTION:")
        print("==============================================")
        print("1. Train and Evaluate CNN Model (Saves to cnn_chordnet_baseline.pth)")
        print(f"2. Train HMM Transition Matrix (Saves to {HMM_TRANSITION_PATH})")
        print("3. Predict Chords for a New Audio File (with HMM Smoothing)")
        print("4. Predict Chords for a New Audio File (RAW CNN)")
        print("5. Evaluate CSR/WCSR/Overlap on TEST split")
        print("6. Evaluate & Export Reports (CSV/JSON + Confusion)")
        print("7. Exit")
        print("==============================================")

        choice = input("Enter operation number: ").strip()

        if choice == "1":
            run_training_evaluation(dataset, NUM_CLASSES, device,
                                    epochs=EPOCHS, lr=LR, model_path=MODEL_PATH,
                                    max_batches_per_epoch=100)

        elif choice == "2":
            calculate_transition_matrix(dataset)

        elif choice == "3":
            audio_file = input("Enter the full path to the audio file (e.g. C:\\Users\\...\\my_song.mp3): ").strip()
            if os.path.exists(audio_file):
                predict_song(audio_file, MODEL_PATH, dataset, device, use_hmm=True)
            else:
                print(f"Error: File '{audio_file}' does not exist or the path is incorrect.")

        elif choice == "4":
            audio_file = input("Enter the full path to the audio file (e.g. C:\\Users\\...\\my_song.mp3): ").strip()
            if os.path.exists(audio_file):
                predict_song(audio_file, MODEL_PATH, dataset, device, use_hmm=False)
            else:
                print(f"Error: File '{audio_file}' does not exist or the path is incorrect.")

        elif choice == "5":
            evaluate_on_test(dataset, MODEL_PATH, device, use_hmm=True)

        elif choice == "6":
            from src.reporter import evaluate_and_report
            summary = evaluate_and_report(dataset, MODEL_PATH, device, out_dir="reports", use_hmm=True)
            if summary:
                print("\nReports saved to:")
                for k, v in summary["files"].items():
                    print(f" - {k}: {v}")

        elif choice == "7":
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid input. Please choose 1, 2, 3, 4, 5, 6 or 7.")
