# Dimitris Karatzas aivc25007

import os
os.environ["KERAS_BACKEND"] = "torch"
from keras import ops
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def psnr_metric(y_true, y_pred):
    """PSNR calculation in dB between a given image and a ground truth"""
    mse = ops.mean(ops.square(y_true - y_pred))
    return 10.0 * ops.log(1.0 / (mse + 1e-10)) / ops.log(10.0)

def ssim_metric(y_true, y_pred):
    """SSIM (uses the standard 11x11 Gaussian window by skimage), evaluation only"""
    y_true_np = np.array(y_true).squeeze()
    y_pred_np = np.array(y_pred).squeeze()
    return ssim(y_true_np, y_pred_np, data_range=1.0)

def get_y_channel(pil_img):
    """Returns (y_arr [H,W,1] float32 normalized in [0,1], cb PIL, cr PIL)"""
    ycbcr = pil_img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y_arr = np.array(y).astype("float32") / 255.0
    return np.expand_dims(y_arr, axis=-1), cb, cr

def reconstruct_rgb(y_arr, cb, cr):
    """Reconstructs an RGB PIL image from Y array [H,W,1] float32 and Cb/Cr PIL channels"""
    y_uint8 = np.clip(y_arr[:, :, 0] * 255.0, 0, 255).astype("uint8")
    y_pil = Image.fromarray(y_uint8, mode="L")
    h, w = y_uint8.shape
    cb_r = cb.resize((w, h), Image.Resampling.BICUBIC)
    cr_r = cr.resize((w, h), Image.Resampling.BICUBIC)
    return Image.merge("YCbCr", (y_pil, cb_r, cr_r)).convert("RGB")

def downscale_image(pil_img, factor=2):
    """Downscales a PIL image by the given integer factor using BICUBIC"""
    w, h = pil_img.size
    return pil_img.resize((w // factor, h // factor), Image.Resampling.BICUBIC)