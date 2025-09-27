"""
==========================================================================
 Room Impulse Response (RIR) Reconstruction using LSTM-based EDC Prediction
==========================================================================

Author: Imran Muhammad
Date: 2025-09-27

📌 DESCRIPTION
-------------
This script performs inference using a pre-trained LSTM model that predicts
Energy Decay Curves (EDCs) from room configuration features. It then reconstructs
the corresponding Room Impulse Responses (RIRs) using the Random Sign-Sticky method.

The model and scalers are trained separately. For inference, you have two options:

1️⃣ **Option 1 - Use Existing Dataset**  
   - Select a random example from a pre-generated dataset.  
   - Actual EDC will be loaded and compared to the predicted EDC.  
   - Plots will show both actual and predicted results.

2️⃣ **Option 2 - Use Custom Room Features**  
   - Manually enter or use default room dimensions, positions, and absorption.  
   - Only predicted EDC, RIR, and FFT will be generated (no ground truth).

📌 INPUT FEATURES FORMAT (16 features)
--------------------------------------
[L, W, H, src_x, src_y, src_z, rec_x, rec_y, rec_z, absorption_band1..7]

📌 OUTPUT
--------
- Predicted EDC curve (dB)
- Predicted RIR waveform
- FFT magnitude response
- Optional comparison with actual dataset (if using Option 1)

📌 USAGE
-------
$ python inference_edcModelPytorchLighteningV3.py

Make sure:
- `Models/` folder contains:
    - `best_model.ckpt`
    - `scaler_X_*.save`
    - `scaler_edc_*.save`
- Dataset CSV and EDC .npy files are in correct paths if using Option 1.

==========================================================================
"""

import os
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error

# ==========================================================
#  Model Definition (same as used in training)
# ==========================================================

class EDCModel(pl.LightningModule):
    def __init__(self, input_dim, target_length):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 2048)
        self.fc2 = nn.Linear(2048, target_length)

    def forward(self, x):
        x = x.to(self.device)
        _, (h_n, _) = self.lstm(x)
        x = torch.relu(self.fc1(h_n[-1]))
        x = self.dropout(x)
        return self.fc2(x)

# ==========================================================
#  Helper Function: Random Sign-Sticky RIR Reconstruction
# ==========================================================

def reconstruct_random_sign_sticky(edc, stickiness=0.90):
    diff_edc = -np.diff(edc, append=0)
    diff_edc = np.clip(diff_edc, 0, None)
    rir_mag = np.sqrt(diff_edc)
    signs = np.empty_like(rir_mag)
    last_sign = 1
    for i, mag in enumerate(rir_mag):
        if mag == 0:
            signs[i] = last_sign
        else:
            signs[i] = last_sign if np.random.rand() < stickiness else -last_sign
            last_sign = signs[i]
    return rir_mag * signs

# ==========================================================
#  MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    # ------------------------------
    # Configuration
    # ------------------------------
    FS = 48000
    target_length = FS * 2   # 2 seconds
    rooms_to_process = 200
    absCases = 30
    max_files_to_load = rooms_to_process * absCases
    input_dim = 16

    room_features_csv = "dataset/room_acoustic_largedataset/roomFeaturesDataset.csv"
    edc_folder = "dataset/room_acoustic_largedataset/EDC"
    checkpoint_path = "Models/best_model.ckpt"
    scaler_X_path = f"Models/scaler_X_{max_files_to_load}_{target_length}.save"
    scaler_y_path = f"Models/scaler_edc_{max_files_to_load}_{target_length}.save"

    # Create output folder
    os.makedirs("inference_results", exist_ok=True)

    print("\n==============================")
    print("  RIR Reconstruction Inference")
    print("==============================")
    print("Select mode:")
    print("1 - Use existing dataset example")
    print("2 - Use custom room features")
    choice = input("Enter choice (1/2): ").strip()

    # ==========================================================
    #  OPTION 1: Existing Dataset
    # ==========================================================
    if choice == "1":
        print("\nYou selected: Use existing dataset")

        df_features = pd.read_csv(room_features_csv)
        room_ids = df_features.iloc[:, 0].values
        features_only = df_features.drop(columns=[df_features.columns[0]]).values

        rng = np.random.default_rng()
        rand_idx = rng.integers(0, 6000)
        selected_features = features_only[rand_idx]
        selected_room_id = room_ids[rand_idx]

        print(f"Selected room ID: {selected_room_id}")

        edc_filename = f"{selected_room_id}.npy"
        edc_path = os.path.join(edc_folder, edc_filename)
        if not os.path.exists(edc_path):
            raise FileNotFoundError(f"EDC file not found: {edc_path}")

        actual_edc = np.load(edc_path).flatten()
        actual_edc = np.pad(actual_edc, (0, max(0, target_length - len(actual_edc))), mode='constant')[:target_length]

        # Load scalers
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)

        # Scale input features
        new_features_scaled = scaler_X.transform(selected_features.reshape(1, -1))
        new_features_scaled = new_features_scaled.reshape(1, 1, input_dim)

        # Load model
        model = EDCModel.load_from_checkpoint(checkpoint_path, input_dim=input_dim, target_length=target_length)
        model.eval()
        

        with torch.no_grad():
            X_tensor = torch.tensor(new_features_scaled, dtype=torch.float32)
            pred_scaled = model(X_tensor).cpu().numpy()

        pred_edc = scaler_y.inverse_transform(pred_scaled)[0]

        # MSE
        mse = mean_squared_error(actual_edc, pred_edc)
        print(f"EDC MSE = {mse:.6f}")

        # RIR reconstruction
        pred_rir = reconstruct_random_sign_sticky(pred_edc)
        actual_rir = reconstruct_random_sign_sticky(actual_edc)

        # Plot
        epsilon = 1e-18
        freqs = np.fft.rfftfreq(len(pred_rir), 1 / FS)
        pred_fft_db = 20 * np.log10(np.abs(np.fft.rfft(pred_rir)) / np.max(np.abs(np.fft.rfft(actual_rir))) + epsilon)
        actual_fft_db = 20 * np.log10(np.abs(np.fft.rfft(actual_rir)) / np.max(np.abs(np.fft.rfft(actual_rir))) + epsilon)

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        axs[0].plot(10 * np.log10(actual_edc + epsilon), label="Actual EDC")
        axs[0].plot(10 * np.log10(pred_edc + epsilon), label="Predicted EDC")
        axs[0].set_title("EDCs (dB)"); axs[0].legend(); axs[0].grid(True)

        axs[1].plot(actual_rir, label="Actual RIR")
        axs[1].plot(pred_rir, label="Predicted RIR")
        axs[1].set_title("RIR"); axs[1].legend(); axs[1].grid(True)

        axs[2].plot(freqs, actual_fft_db, label="Actual FFT")
        axs[2].plot(freqs, pred_fft_db, label="Predicted FFT")
        axs[2].set_xscale("log"); axs[2].set_title("FFT"); axs[2].legend(); axs[2].grid(True)

        plt.tight_layout()
        plt.savefig("inference_results/comparison_existing.png", dpi=300)
        plt.show()

    # ==========================================================
    #  OPTION 2: Custom Room Features
    # ==========================================================
    elif choice == "2":
        print("\nYou selected: Use custom room features")
        use_defaults = input("Use default example values? (y/n): ").strip().lower()

        if use_defaults == "y":
            length, width, height = 4.0, 5.0, 2.5
            src = [1.5, 2.0, 1.2]
            rec = [3.0, 3.5, 1.2]
            absorption = [0.01, 0.014, 0.1, 0.19, 0.2, 0.21, 0.023]
        else:
            length = float(input("Length (m): "))
            width = float(input("Width (m): "))
            height = float(input("Height (m): "))
            src = [float(input(f"Source {axis} (m): ")) for axis in ['X','Y','Z']]
            rec = [float(input(f"Receiver {axis} (m): ")) for axis in ['X','Y','Z']]
            absorption = [float(input(f"Absorption Band {i+1}: ")) for i in range(7)]

        selected_features = np.array([length, width, height] + src + rec + absorption)
        print(f"Custom feature vector: {selected_features}")

        # Predict only (no actual)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        new_features_scaled = scaler_X.transform(selected_features.reshape(1, -1)).reshape(1, 1, input_dim)
        model = EDCModel.load_from_checkpoint(checkpoint_path, input_dim=input_dim, target_length=target_length)
        model.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(new_features_scaled, dtype=torch.float32)
            pred_scaled = model(X_tensor).cpu().numpy()
        pred_edc = scaler_y.inverse_transform(pred_scaled)[0]

        pred_rir = reconstruct_random_sign_sticky(pred_edc)
        freqs = np.fft.rfftfreq(len(pred_rir), 1 / FS)
        epsilon = 1e-18
        pred_fft_db = 20 * np.log10(np.abs(np.fft.rfft(pred_rir)) / np.max(np.abs(np.fft.rfft(pred_rir))) + epsilon)

        # Plot predicted only
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        axs[0].plot(10 * np.log10(pred_edc + epsilon), label="Predicted EDC")
        axs[0].set_title("Predicted EDC (dB)"); axs[0].legend(); axs[0].grid(True)

        axs[1].plot(pred_rir, label="Predicted RIR")
        axs[1].set_title("Predicted RIR"); axs[1].legend(); axs[1].grid(True)

        axs[2].plot(freqs, pred_fft_db, label="Predicted FFT")
        axs[2].set_xscale("log"); axs[2].set_title("Predicted FFT"); axs[2].legend(); axs[2].grid(True)

        plt.tight_layout()
        plt.savefig("inference_results/predicted_only_custom.png", dpi=300)
        plt.show()

    else:
        print("❌ Invalid choice. Please run the script again.")
