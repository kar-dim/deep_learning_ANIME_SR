# Dimitris Karatzas aivc25007

import os
import json
import numpy as np

# MUST be set before importing keras or torch
os.environ["KERAS_BACKEND"] = "torch"

import torch

# GPU check, CPU is much slower for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nPyTorch is using: {device.upper()}")
if device == "cuda":
    torch.set_default_device('cuda')
    print("INFO: torch.set_default_device('cuda') set BEFORE keras import.")
else:
    print("WARNING: CUDA not available! Training will be very slow.")

import keras
from utils import psnr_metric
from models import build_srcnn, build_edsr

# Change this value to train a specific model
# Either "SRCNN", or "EDSR", or "EDSR_FULL"
MODEL_TYPE = "EDSR"     
EPOCHS = 150

# Settings per model
CONFIG = {
    "SRCNN": {"batch_size": 64, "patch_size": 33, "lr": 1e-3},
    "EDSR": {"batch_size": 32, "patch_size": 64, "lr": 1e-4},
    "EDSR_FULL": {"batch_size": 8, "patch_size": 64, "lr": 1e-4},
}


def random_flip(lr, hr):
    """Random horizontal and vertical flip, works on 2D (H,W) and 3D (H,W,C) arrays"""
    if np.random.rand() < 0.5:
        lr = lr[:, ::-1]
        hr = hr[:, ::-1]
    if np.random.rand() < 0.5:
        lr = lr[::-1]
        hr = hr[::-1]
    return lr, hr


# SRCNN DATA GENERATOR (reads precomputed Y-channel .npy files)
# Reads y_hr_data.npy and y_lr_data.npy: uint8 (N, 720, 1280), bicubic already done beforehand (prepare_data)
# Generator does only simple patch extraction + float32 cast
class SRDataGeneratorSRCNN(keras.utils.PyDataset):
    def __init__(self, y_hr_data, y_lr_data, img_indices,
                 patch_size=33, batch_size=64, shuffle=True, augment=True, **kwargs):
        super().__init__(**kwargs)
        self.y_hr = y_hr_data # (N, H, W) uint8
        self.y_lr = y_lr_data # (N, H, W) uint8 (bicubic upscaled)
        self.img_indices = np.array(img_indices)
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.order = np.arange(len(self.img_indices))
        self.on_epoch_end()

    def __len__(self):
        return len(self.order) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.order)

    def __getitem__(self, idx):
        p = self.patch_size
        batch_hr = np.empty((self.batch_size, p, p, 1), dtype=np.float32)
        batch_lr = np.empty((self.batch_size, p, p, 1), dtype=np.float32)

        for i, pos in enumerate(self.order[idx * self.batch_size : (idx + 1) * self.batch_size]):
            img_idx = self.img_indices[pos]
            hr_y = self.y_hr[img_idx] # (720, 1280) uint8
            lr_y = self.y_lr[img_idx] # (720, 1280) uint8

            h, w = hr_y.shape
            iy = np.random.randint(0, h - p)
            ix = np.random.randint(0, w - p)

            hr_p = hr_y[iy:iy+p, ix:ix+p]
            lr_p = lr_y[iy:iy+p, ix:ix+p]

            if self.augment:
                lr_p, hr_p = random_flip(lr_p, hr_p)

            batch_hr[i, :, :, 0] = hr_p
            batch_hr[i, :, :, 0] /= 255.0
            batch_lr[i, :, :, 0] = lr_p
            batch_lr[i, :, :, 0] /= 255.0

        return batch_lr, batch_hr


# EDSR DATA GENERATOR (RGB, LR patches, HR patches, again numpy only, no PIL here, it is done in prepare_data)
# DIFFERENCES vs SRCNN generator:
# 1) No bicubic upscale, EDSR takes the original LR directly
# 2) No Y channel conversion, EDSR works in RGB
class SRDataGeneratorEDSR(keras.utils.PyDataset):
    def __init__(self, hr_data, lr_data, img_indices, patch_size_lr=64, batch_size=32,
                 augment=True, **kwargs):
        super().__init__(**kwargs)
        self.hr_data = hr_data
        self.lr_data = lr_data
        self.img_indices = np.array(img_indices)
        self.patch_lr = patch_size_lr
        self.patch_hr = patch_size_lr * 2
        self.batch_size = batch_size
        self.augment = augment
        self.order = np.arange(len(self.img_indices))
        self.on_epoch_end()

    def __len__(self):
        return len(self.order) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.order)

    def __getitem__(self, idx):
        batch_pos = self.order[idx * self.batch_size : (idx + 1) * self.batch_size]

        plr = self.patch_lr
        phr = self.patch_hr
        batch_lr = np.empty((self.batch_size, plr, plr, 3), dtype=np.float32)
        batch_hr = np.empty((self.batch_size, phr, phr, 3), dtype=np.float32)

        for i, pos in enumerate(batch_pos):
            img_idx = self.img_indices[pos]
            lr_img = self.lr_data[img_idx] # (360, 640, 3) uint8
            hr_img = self.hr_data[img_idx] # (720, 1280, 3) uint8

            lr_h, lr_w = lr_img.shape[:2]
            iy = np.random.randint(0, lr_h - plr)
            ix = np.random.randint(0, lr_w - plr)

            lr_p = lr_img[iy : iy + plr, ix : ix + plr]
            hr_p = hr_img[iy*2 : iy*2 + phr, ix*2 : ix*2 + phr]

            if self.augment:
                lr_p, hr_p = random_flip(lr_p, hr_p)

            batch_lr[i] = lr_p
            batch_lr[i] /= 255.0
            batch_hr[i] = hr_p
            batch_hr[i] /= 255.0

        return batch_lr, batch_hr


# MAIN program
if __name__ == "__main__":
    cfg = CONFIG[MODEL_TYPE]
    batch_size = cfg["batch_size"]
    patch_size = cfg["patch_size"]
    lr_init = cfg["lr"]

    rng = np.random.default_rng(2918)

    if MODEL_TYPE == "SRCNN":
        # SRCNN reads precomputed Y channel data
        print(f"\nRAM LOAD: Loading y_hr_data.npy, y_lr_data.npy (close to 6.4 GB)...")
        y_hr_all = np.load("y_hr_data.npy") # (N, 720, 1280) uint8
        y_lr_all = np.load("y_lr_data.npy") # (N, 720, 1280) uint8
        print(f"Y_HR: {y_hr_all.shape}, Y_LR: {y_lr_all.shape}")
        n = len(y_hr_all)
        idx = rng.permutation(n)
        split = int(n * 0.9) # We use 90 % train and 10% validation split
        train_idx, val_idx = idx[:split], idx[split:]
        print(f"Train: {len(train_idx)} images, Validation: {len(val_idx)} images")

        model = build_srcnn()
        train_gen = SRDataGeneratorSRCNN(y_hr_all, y_lr_all, train_idx, patch_size=patch_size, batch_size=batch_size, augment=True, workers=4)
        val_gen = SRDataGeneratorSRCNN(y_hr_all, y_lr_all, val_idx, patch_size=patch_size, batch_size=batch_size, shuffle=False, augment=False, workers=2)
        save_path = "best_srcnn.keras"
        hist_path = "history_srcnn.json"

    elif MODEL_TYPE in ("EDSR", "EDSR_FULL"):
        # EDSR reads full RGB data
        print(f"\nRAM LOAD: Loading hr_data.npy, lr_data.npy (close to 12 GB)...")
        hr_all = np.load("hr_data.npy")
        lr_all = np.load("lr_data.npy")
        print(f"HR: {hr_all.shape}, LR: {lr_all.shape}")
        n = len(hr_all)
        idx = rng.permutation(n)
        split = int(n * 0.9) # We use 90 % train and 10% validation split
        train_idx, val_idx = idx[:split], idx[split:]
        print(f"Train: {len(train_idx)} images, Validation: {len(val_idx)} images")

        model = build_edsr(num_blocks=32, num_filters=256) if MODEL_TYPE == "EDSR_FULL" else build_edsr()
        train_gen = SRDataGeneratorEDSR(hr_all, lr_all, train_idx, patch_size_lr=patch_size, batch_size=batch_size, augment=True, workers=4)
        val_gen = SRDataGeneratorEDSR(hr_all, lr_all, val_idx, patch_size_lr=patch_size, batch_size=batch_size, augment=False, workers=2)
        save_path = "best_edsr_baseline.keras" if MODEL_TYPE == "EDSR" else "best_edsr_full.keras"
        hist_path = "history_edsr.json" if MODEL_TYPE == "EDSR" else "history_edsr_full.json"
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_init), loss="mae", metrics=[psnr_metric])
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor="val_psnr_metric", mode="max"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    ]

    print(f"\n-- TRAINING {MODEL_TYPE} --")
    if MODEL_TYPE == "SRCNN":
        print(f"Patch: {patch_size}x{patch_size} (Y channel, bicubic upscaled input)")
    else:
        print(f"Patch: {patch_size}x{patch_size} LR -> {patch_size*2}x{patch_size*2} HR (RGB)")
    print(f"Batch size: {batch_size}")
    print(f"LR init: {lr_init}")
    print(f"Epochs: {EPOCHS}")

    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
    with open(hist_path, "w") as f:
        json.dump(history.history, f)
    print(f"\n{MODEL_TYPE} training complete. History saved to {hist_path}")
