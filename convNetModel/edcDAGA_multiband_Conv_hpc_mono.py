"""
==========================================================
Convnet-based model to predict EDC curves for 24 1/3 octave bands from room geometry + frequency input.
This code is using monotonic EDC generation to ensure physically plausible decay curves.
----------------------------------------------------------
Author: Imran
This Model is used for DAGA 2026
Last updated: 2026-03-11
==========================================================
"""

import os
import re
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


# ------------------------------
# Configuration
#today = datetime.today().strftime('%Y-%m-%d')
resuts_path = "Results_Conv_Mono"
os.makedirs(resuts_path, exist_ok=True)
# Paths
base_edc_path = f"EDC_1_3_Bands_500/" 
room_features_path = f"room_features/full_large_dataset.csv"
model_save_dir = f"{resuts_path}/Trained_Models_Conv_Mono"
os.makedirs(model_save_dir, exist_ok=True)
# Hyperparameters
target_length = 48000 * 2  # 2 seconds at 48kHz, adjust if needed
input_dim = 17             # 16 Geometry + 1 Frequency Feature
batch_size = 16
rooms_to_process = 300     # Adjust based on available rooms
maxEpochs = 200

#print(f"Configuration:\n- Target Length: {target_length}\n- Input Dim: {input_dim}\n- Batch Size: {batch_size}\n- Rooms to Process: {rooms_to_process}\n- Max Epochs: {maxEpochs}\n")

isScalingX = True
isScalingY = True       # EDC is already 0-1, but MinMaxScaler handles distribution

NOMINAL_CENTERS = [
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 
    1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 
    16000, 20000
]

# 1. Define the Logger with a custom name
# This will create a folder: lightning_logs/LogLoss_SSD_Test/
logger = TensorBoardLogger(
    save_dir="lightning_logs_Conv_Mono", 
    name="LogLoss_Conv_Mono"
)

def print_gpu_status():
    print("="*50)
    print("      DAGA 2026 - GPU     ")
    print("="*50)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            print(f"ID: {i} | Name: {prop.name}")
            print(f"      VRAM: {prop.total_memory / 1e9:.2f} GB")
            print(f"      Compute Capability: {prop.major}.{prop.minor}")
        print(f"\nACTIVE DEVICE: {torch.cuda.current_device()} ({torch.cuda.get_device_name()})")
    else:
        print("NO CUDA GPU DETECTED!")
    print("="*50 + "\n")

# ------------------------------
# 2. Natural Sorting Helper
# ------------------------------
def natural_key(string_):
    """
    Allows natural sorting (1, 2, 10 instead of 1, 10, 2)
    """
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string_)]

# ------------------------------
# Helper Functions
# ------------------------------
def extract_rir_case(fname):
    """
    Extracts RIR and Case numbers from strings like 'EDC_001_case0_edc.npy'
    or 'EDC_001_case0_edc' (from CSV).
    """
    nums = re.findall(r'\d+', fname)
    if len(nums) >= 2:
        # nums[0] is the room (001 -> 1), nums[1] is the case (0 -> 0)
        return int(nums[0]), int(nums[1])
    return None, None

class MultiBandEDCDataset(Dataset):
    def __init__(self, base_path, room_features_df, nominal_centers, rooms_limit=100):
        self.samples = []
        
        # 1. Scale Geometry Features (All cols except 'ID')
        self.scaler_X = MinMaxScaler()
        geom_cols = [c for c in room_features_df.columns if c != 'ID']
        geom_scaled = self.scaler_X.fit_transform(room_features_df[geom_cols].values)
        joblib.dump(self.scaler_X, f'{resuts_path}/scaler_X_geometry_Conv_Mono.save')

        # 2. Map ID string to scaled geometry
        geom_lookup = {}
        for idx, row in room_features_df.iterrows():
            r_id, c_id = extract_rir_case(row['ID'])
            if r_id is not None:
                geom_lookup[(r_id, c_id)] = geom_scaled[idx]

        # 3. Load Files from band folders
        for b_idx, fc in enumerate(nominal_centers):
            band_dir = os.path.join(base_path, f"band_{b_idx:02d}")
            if not os.path.exists(band_dir): continue
            
            files = sorted([f for f in os.listdir(band_dir) if f.endswith('.npy')], key=natural_key)
            freq_feat = np.log10(fc)
            
            for fname in files:
                r_num, c_num = extract_rir_case(fname)
                if r_num is None or r_num > rooms_limit: continue
                
                if (r_num, c_num) in geom_lookup:
                    combined_feat = np.append(geom_lookup[(r_num, c_num)], freq_feat)
                    self.samples.append({
                        'path': os.path.join(band_dir, fname), 
                        'features': combined_feat
                    })

    # --- ENSURE THESE ARE INDENTED ONCE UNDER THE CLASS ---
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        edc = np.load(s['path']).flatten()[:target_length]
        if len(edc) < target_length:
            edc = np.pad(edc, (0, target_length - len(edc)), mode='constant')
        x = torch.tensor(s['features'], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(edc, dtype=torch.float32)
        return x, y


# ------------------------------
# Loss Functions
# ------------------------------
class EDCLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()

    def forward(self, y_pred, y_true):
        edc_loss = F.mse_loss(y_pred, y_true)
        return edc_loss

class LogEDCLoss(nn.Module):
    def __init__(self, eps=1e-10, slope_weight=0.2):
        super().__init__()
        self.eps = eps
        self.slope_weight = slope_weight # Adjust this to balance level vs. slope accuracy

    def forward(self, y_pred, y_true):
        # 1. Convert to dB domain
        y_pred_db = 10 * torch.log10(y_pred + self.eps)
        y_true_db = 10 * torch.log10(y_true + self.eps)

        # 2. Standard MSE Loss (Level Error)
        mse_loss = F.mse_loss(y_pred_db, y_true_db)

        # 3. Slope Penalty (Derivative Error)
        # Calculate the difference between adjacent samples (the slope)
        # We use a stride to look at the slope over a slightly larger window (e.g., 10 samples)
        # to avoid noise and focus on the decay trend.
        stride = 50
        slope_pred = y_pred_db[:, stride:] - y_pred_db[:, :-stride]
        slope_true = y_true_db[:, stride:] - y_true_db[:, :-stride]
        
        slope_loss = F.mse_loss(slope_pred, slope_true)

        # Total Loss = Level Accuracy + Slope Accuracy
        return mse_loss + (self.slope_weight * slope_loss)


class EDCModel(pl.LightningModule):

    def __init__(self, input_dim, target_length, loss_type="linear"):
        super().__init__()
        self.save_hyperparameters()

        # -------------------------
        # Loss
        # -------------------------
        if loss_type == "log":
            self.criterion = LogEDCLoss()
        else:
            self.criterion = nn.MSELoss()

        # -------------------------
        # Feature Encoder (MLP)
        # -------------------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU()
        )

        # -------------------------
        # Latent → Temporal Grid
        # -------------------------
        self.fc_expand = nn.Linear(512, 256 * 64)

        # -------------------------
        # Conv1D Decoder
        # -------------------------
        self.decoder = nn.Sequential(

            nn.Conv1d(256, 128, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv1d(32, 1, kernel_size=7, padding=3)
        )

        self.dropout = nn.Dropout(0.3)
        self.target_length = target_length


    def forward(self, x):
        # remove seq dimension
        x = x.squeeze(1)
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.fc_expand(x)
        # reshape to temporal feature map
        x = x.view(x.shape[0], 256, 64)
        x = self.decoder(x)
        # upsample to full EDC resolution
        x = F.interpolate(
            x,
            size=self.target_length,
            mode="linear",
            align_corners=False
        )
        # --- MONOTONIC EDC GENERATION ---
        x = x.squeeze(1)                 # remove channel dim
        rate = F.softplus(x)
        rate = F.avg_pool1d(rate.unsqueeze(1), kernel_size=5, stride=1, padding=2 ).squeeze(1)
        # scale the rate for stability
        scale_factor = 5e-5
        rate_scaled = rate * scale_factor

        # integrate decay
        edc = torch.exp(-torch.cumsum(rate_scaled, dim=1))

        return edc


    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return loss


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }



class EDCPlotCallback(pl.Callback):
    def __init__(self, val_dataset, results_path, fs=48000):
        super().__init__()
        self.val_dataset = val_dataset
        self.results_path = results_path
        self.fs = fs

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only plot every 10 epochs
        if (trainer.current_epoch + 1) % 10 == 0:
            pl_module.eval()
            
            # Get the first 24 bands (Room 1)
            preds, targets = [], []
            device = pl_module.device
            
            with torch.no_grad():
                for i in range(24):
                    x, y = self.val_dataset[i]
                    out = pl_module(x.unsqueeze(0).to(device))
                    preds.append(out.squeeze().cpu().numpy())
                    targets.append(y.numpy())
            
            preds, targets = np.array(preds), np.array(targets)
            time_axis = np.linspace(0, targets.shape[1]/self.fs, targets.shape[1])
            
            # Create Plot
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            
            # Band Plot (Selected)
            indices = [0, 6, 12, 18, 23]
            for idx in indices:
                ax[0].plot(time_axis, 10*np.log10(targets[idx]+1e-10), '--', alpha=0.3)
                ax[0].plot(time_axis, 10*np.log10(preds[idx]+1e-10), label=f"Band {idx}")
            
            ax[0].set_title(f"Epoch {trainer.current_epoch}: Bands")
            ax[0].set_ylim([-60, 0]); ax[0].legend(); ax[0].grid(True)

            # Broadband Plot
            bb_t = np.sum(targets, axis=0); bb_t /= bb_t[0]
            bb_p = np.sum(preds, axis=0); bb_p /= bb_p[0]
            
            ax[1].plot(time_axis, 10*np.log10(bb_t+1e-10), 'k--', label="Target")
            ax[1].plot(time_axis, 10*np.log10(bb_p+1e-10), 'r', label="Pred")
            ax[1].set_title(f"Epoch {trainer.current_epoch}: Broadband")
            ax[1].set_ylim([-60, 0]); ax[1].legend(); ax[1].grid(True)

            plt.tight_layout()
            save_fn = os.path.join(self.results_path, f"EDC_Evolution_Epoch_{trainer.current_epoch}.png")
            plt.savefig(save_fn)
            plt.close()
            print(f"\n>>> Saved Evolution Plot: {save_fn}")

# ------------------------------
# 5. Execute & Save
# ------------------------------
if __name__ == "__main__":
    
    print_gpu_status()
    
    room_df = pd.read_csv(room_features_path)
    full_dataset = MultiBandEDCDataset(base_edc_path, room_df, NOMINAL_CENTERS, rooms_limit=rooms_to_process)
    
    selected_loss = "log"
    
    train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
    train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=11,persistent_workers=True )
    val_loader = DataLoader(torch.utils.data.Subset(full_dataset, val_idx), batch_size=batch_size, num_workers=11,persistent_workers=True )

    model = EDCModel(
        input_dim=17, 
        target_length=target_length, 
        loss_type=selected_loss
    )
    
    checkpoint = ModelCheckpoint(monitor="val_loss", dirpath=model_save_dir, filename="best_model-v1")
    
    # Define the Val Subset for the callback to use
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)
    
    # Initialize our new plotting callback
    plot_callback = EDCPlotCallback(val_subset, resuts_path, fs=48000)
    
    trainer = pl.Trainer(
        max_epochs=maxEpochs,
        callbacks=[checkpoint, plot_callback],
        accelerator='gpu',
        devices=1,
        precision=32,
        logger=logger,
        log_every_n_steps=10)
    
    checkpoint_path = f"{model_save_dir}/best_model-v1.ckpt"

    # 2. Check if it exists and resume
    if os.path.exists(checkpoint_path):
        print(f"RESUMING FROM CHECKPOINT: {checkpoint_path}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    else:
        print("No checkpoint found, starting fresh.")
        trainer.fit(model, train_loader, val_loader)
    
    print(f"Success! Model Traing completed")
    # Final Evaluation & Save Data
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            output = model(X)
            preds.append(output.cpu().numpy())
            targets.append(y.cpu().numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    # Save Results
    np.save(f"{resuts_path}/predicted_edcs.npy", preds)
    np.save(f"{resuts_path}/actual_edcs.npy", targets)

    # Metadata
    metadata = {
        "target_length": target_length,
        "input_dim": input_dim,
        "mae": float(mean_absolute_error(targets, preds)),
        "mse": float(mean_squared_error(targets, preds)),
        "model_path": str(checkpoint.best_model_path)
    }
    with open(os.path.join(resuts_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Success! Data natural sorted and results saved to {resuts_path}")

