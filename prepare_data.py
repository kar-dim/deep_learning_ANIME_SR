# Dimitris Karatzas aivc25007

import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# PIL format is (width, height)
HR_SIZE = (1280, 720)   
LR_SIZE = (640,  360)


def cleanup_dataset(hr_dir, lr_dir):
    """Deletes images that do not match HR_SIZE, LR_SIZE exactly"""
    hr_paths = sorted(Path(hr_dir).glob("*.png"))
    lr_paths = sorted(Path(lr_dir).glob("*.png"))

    hr_deleted = lr_deleted = 0

    print(f"\nScanning HR folder: {hr_dir}")
    for p in tqdm(hr_paths):
        with Image.open(p) as img:
            if img.size != HR_SIZE:
                p.unlink()
                hr_deleted += 1

    print(f"Scanning LR folder: {lr_dir}")
    for p in tqdm(lr_paths):
        with Image.open(p) as img:
            if img.size != LR_SIZE:
                p.unlink()
                lr_deleted += 1

    print(f"\nDeleted {hr_deleted} HR and {lr_deleted} LR images (wrong size).")

def precompute_edsr_data(hr_dir, lr_dir, out_hr="hr_data.npy", out_lr="lr_data.npy"):
    """
    Converts paired HR/LR PNGs to uint8 .npy arrays for EDSR training
    skips any pairs that don't match the expected resolutions.
    """
    hr_paths = sorted(Path(hr_dir).glob("*.png"))
    lr_paths = sorted(Path(lr_dir).glob("*.png"))

    hr_map = {p.name: p for p in hr_paths}
    lr_map = {p.name: p for p in lr_paths}
    common = sorted(set(hr_map) & set(lr_map))

    print(f"\nFiltering and converting {len(common)} paired images to NumPy...")

    hr_clean, lr_clean = [], []
    skipped = 0

    for name in tqdm(common):
        with Image.open(hr_map[name]) as h_img, Image.open(lr_map[name]) as l_img:
            if h_img.size == HR_SIZE and l_img.size == LR_SIZE:
                hr_clean.append(np.array(h_img.convert("RGB"), dtype="uint8"))
                lr_clean.append(np.array(l_img.convert("RGB"), dtype="uint8"))
            else:
                skipped += 1

    print(f"Kept {len(hr_clean)} pairs, skipped {skipped} (wrong size).")
    print(f"Saving {out_hr} and {out_lr}...")
    np.save(out_hr, np.array(hr_clean))
    np.save(out_lr, np.array(lr_clean))
    print("Done!")

def precompute_srcnn_data(hr_npy="hr_data.npy", lr_npy="lr_data.npy", out_hr_y="y_hr_data.npy", out_lr_y="y_lr_data.npy"):
    """
    Precomputes bicubic upscaled LR Y channel and HR Y channel for SRCNN
    Reads .npy files via mmap more = 'r' (no full RAM load needed here)
    Output: uint8 (N, 720, 1280) arrays, 3.2 GB each
    SRCNN generator (used in training) becomes faster, uses simple numpy patch extraction
    """
    print("\nPRECOMPUTE: Reading source arrays via mmap...")
    hr_all = np.load(hr_npy, mmap_mode='r') # (N, 720, 1280, 3) uint8
    lr_all = np.load(lr_npy, mmap_mode='r') # (N, 360, 640, 3) uint8
    N = len(hr_all)
    H, W = 720, 1280

    y_hr_out = np.empty((N, H, W), dtype=np.uint8)
    y_lr_out = np.empty((N, H, W), dtype=np.uint8)

    print(f"PRECOMPUTE: Processing {N} pairs (bicubic upscale + YCbCr Y channel)...")
    for i in tqdm(range(N)):
        hr_img = Image.fromarray(hr_all[i])
        lr_img = Image.fromarray(lr_all[i])
        lr_up = lr_img.resize((W, H), Image.Resampling.BICUBIC)

        y_hr_out[i] = np.array(hr_img.convert("YCbCr"))[:, :, 0]
        y_lr_out[i] = np.array(lr_up.convert("YCbCr"))[:, :, 0]

    print(f"PRECOMPUTE: Saving {out_hr_y}  ({y_hr_out.nbytes / 1e9:.2f} GB)...")
    np.save(out_hr_y, y_hr_out)
    print(f"PRECOMPUTE: Saving {out_lr_y}  ({y_lr_out.nbytes / 1e9:.2f} GB)...")
    np.save(out_lr_y, y_lr_out)
    print("PRECOMPUTE: Done!")

if __name__ == "__main__":
    HR_DIR = "datasets/APISR_Dataset"
    LR_DIR = "datasets/APISR_Dataset_LR"

    cleanup_dataset(HR_DIR, LR_DIR)
    precompute_edsr_data(HR_DIR, LR_DIR)
    precompute_srcnn_data()
