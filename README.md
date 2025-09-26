# Room Impulse Response (RIR) Reconstruction using Deep Learning

## 📖 Introduction

This repository contains a deep learning framework for **reconstructing Room Impulse Responses (RIRs)** from Energy Decay Curves (EDCs) using a **PyTorch Lightning LSTM model**.

The workflow is as follows:

1. Input **room geometry**, **absorption coefficients**, and **source/receiver positions**.
2. The model predicts the **EDC** of the room.
3. The predicted EDC is converted back into a **time-domain RIR** using stochastic reconstruction (Random Sign / Random Sign Sticky).
4. The framework outputs:

   * **Predicted EDC vs. Ground Truth**
   * **Reconstructed RIRs**
   * **Frequency-domain magnitude response**
   * Error metrics such as **Mean Squared Error (MSE)**

This approach is efficient for **early-stage architectural design**, **auralization**, and **real-time acoustic parameter estimation**.

---

## ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/your-username/RIR-Reconstruction.git
cd RIR-Reconstruction
```

### Option 1 – Using pip

```bash
pip install -r requirements.txt
```

### Option 2 – Using conda

```bash
conda env create -f environment.yml
conda activate rir-reconstruction
```

---

## 📂 Repository Structure

```
RIR-Reconstruction/
│
├── dataset/                          # Input dataset (room configs, EDCs)
│   ├── room_acoustic_largedataset/
│   │   ├── full_large_dataset.csv    # Room features
│   │   ├── EDC/                      # Ground truth EDCs (.npy files)
│
├── Results/                          # Saved models, scalers, and inference outputs
│   ├── ICASSP/2025-09-13/...         # Example trained model checkpoint and scalers
│
├── inference_edcModelPytorchLighteningV3.py  # Main inference script
├── requirements.txt                  # Python dependencies (pinned versions)
├── environment.yml                   # Conda environment file
└── README.md
```

---

## ▶️ Running Inference

The main script is **`inference_edcModelPytorchLighteningV3.py`**.

### Step 1 – Prepare Files

Make sure you have:

* A **trained model checkpoint** (`.ckpt`)
* The **scalers** (`scaler_X_*.save` and `scaler_edc_*.save`)
* The **room features CSV** (`full_large_dataset.csv`)
* The **ground truth EDCs** (in `.npy` format inside the EDC folder)

### Step 2 – Run Inference

```bash
python inference_edcModelPytorchLighteningV3.py
```

By default, the script:

* Selects a **random room configuration** from the dataset.
* Loads the **trained model** and **scalers**.
* Predicts the EDC.
* Reconstructs the RIR using the **Random Sign Sticky method**.
* Saves plots, `.wav` files, and results inside `Results/.../inference_results`.

---

## 📊 Outputs

After running inference, you’ll get:

1. **Plots**

   * `comparison_plot.png` – Full view of EDCs, RIRs, FFT
   * `comparison_plot_Zoom.png` – Zoomed-in comparison

2. **Audio Files**

   * `predicted_rir.wav` – Predicted RIR reconstruction
   * `actual_rir.wav` – Ground truth RIR reconstruction

3. **Statistics**

   * Mean Squared Error (MSE) between predicted and actual EDCs

---

## 🎧 Listening Tests

The model’s reconstructed RIRs were further evaluated via **MUSHRA listening tests**, confirming perceptual plausibility of the synthesized responses. Results showed that the **Random Sign Sticky method** significantly outperformed the plain Random Sign approach in terms of subjective quality.

---

## 📌 Notes

* The provided dataset and checkpoints are examples; you can train your own using PyTorch Lightning.
* Default inference runs on **GPU (if available)**, otherwise CPU.
* Parameters like `stickiness` (default = 0.9) can be tuned to balance low-frequency coherence vs. randomness.

---

## 🛠️ Citation

If you use this work, please cite:

```
@inproceedings{YourCitation,
  title={Deep Learning-based RIR Reconstruction from EDCs},
  author={Your Name et al.},
  booktitle={ICASSP},
  year={2025}
}
```

---

✨ With this setup, anyone can easily reproduce your inference pipeline by just running **one Python file**.

---

Do you also want me to prepare a **Training Section** in the README (in case others want to retrain the model), or should we keep it **inference-only** for now?
