import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm
import re

# ------------------------------
# Configuration
# ------------------------------
RIR_FOLDER_PATH = "RIRs_equal_length/"
BASE_EDC_OUTPUT_PATH = "EDC_1_3_Bands_500/"
FS_TARGET = 48000 # Target sampling rate for processing
TARGET_LENGTH = FS_TARGET * 3  # 3 seconds 

ROOMS_TO_PROCESS = 500
EXCLUDE_CASES = [11, 25]  # The "culprit" cases identified

# Standard 1/3 octave centers starting at 100Hz (24 bands total)
NOMINAL_CENTERS = np.array([
    100, 125, 160, 200, 250, 315, 400, 500, 630, 
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 
    12500, 16000, 20000
])

# ------------------------------
# Helper Functions
# ------------------------------

def process_edc(energy_signal, target_len, normalize=True):
    """Calculates EDC via Schroeder integration with fixed length"""
    # Schroeder Integration: flip -> cumsum -> flip
    edc = np.flip(np.cumsum(np.flip(energy_signal)))
    
    if normalize and edc[0] > 0:
        edc = edc / edc[0]
        
    if len(edc) > target_len:
        edc = edc[:target_len]
    else:
        edc = np.pad(edc, (0, max(0, target_len - len(edc))), mode='constant')
    return edc.astype(np.float32)

def get_sort_key(filename):
    rir_match = re.search(r'RIR_(\d+)', filename)
    case_match = re.search(r'case(\d+)', filename)
    rir_num = int(rir_match.group(1)) if rir_match else 0
    case_num = int(case_match.group(1)) if case_match else 0
    return (rir_num, case_num)

# ------------------------------
# Main Process
# ------------------------------

# 1. Initialize Folders
for i in range(len(NOMINAL_CENTERS)):
    os.makedirs(os.path.join(BASE_EDC_OUTPUT_PATH, f"band_{i:02d}"), exist_ok=True)
os.makedirs(os.path.join(BASE_EDC_OUTPUT_PATH, "broadband"), exist_ok=True)

# 2. Filter and Select Rooms
all_wav_files = sorted([f for f in os.listdir(RIR_FOLDER_PATH) if f.endswith('.wav')], key=get_sort_key)
unique_room_ids = sorted(list(set([get_sort_key(f)[0] for f in all_wav_files])))
allowed_room_ids = set(unique_room_ids[:ROOMS_TO_PROCESS])

# 3. Filter files by Room ID AND exclude specific cases
files_to_process = []
for f in all_wav_files:
    r_num, c_num = get_sort_key(f)
    if r_num in allowed_room_ids and c_num not in EXCLUDE_CASES:
        files_to_process.append(f)

print(f"Targeting {ROOMS_TO_PROCESS} rooms.")
print(f"Excluding cases: {EXCLUDE_CASES}")
print(f"Total valid files to process: {len(files_to_process)}")
print("-" * 40)

# 4. Processing Loop
last_rir_num = -1
for filename in tqdm(files_to_process, desc="Processing RIRs"):
    try:
        rir_num, case_num = get_sort_key(filename)
        
        file_path = os.path.join(RIR_FOLDER_PATH, filename)
        out_name = filename.replace(".wav", ".npy").replace("RIR", "EDC")

        # Load Audio
        fs, rir = wavfile.read(file_path)
        rir = rir.astype(np.float32) / 32768.0 if rir.dtype == np.int16 else rir.astype(np.float32)
        if len(rir.shape) > 1: rir = np.mean(rir, axis=1)

        # A. BROADBAND EDC
        edc_bb = process_edc(rir**2, TARGET_LENGTH, normalize=True)
        np.save(os.path.join(BASE_EDC_OUTPUT_PATH, "broadband", out_name), edc_bb)

        # B. BAND EDCs (Analysis phase)
        for i, fc in enumerate(NOMINAL_CENTERS):
            f_low, f_high = fc / (2**(1/6)), fc * (2**(1/6))
            if f_high >= FS_TARGET / 2: continue

            # 2nd Order for temporal precision (step preservation)
            sos = signal.butter(2, [f_low, f_high], btype='bandpass', fs=FS_TARGET, output='sos')
            
            # Zero-phase filtfilt to prevent time-smearing
            filtered_rir = signal.sosfiltfilt(sos, rir)
            
            edc_band = process_edc(filtered_rir**2, TARGET_LENGTH, normalize=True)
            
            save_path = os.path.join(BASE_EDC_OUTPUT_PATH, f"band_{i:02d}", out_name)
            np.save(save_path, edc_band)

    except Exception as e:
        print(f"\n[ERROR] Failed on {filename}: {e}")

print(f"\nSUCCESS: Data stored in {BASE_EDC_OUTPUT_PATH}")