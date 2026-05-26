# Dimitris Karatzas aivc25007

import os

# Like in training, this must be set before importing keras or torch
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
keras.config.enable_unsafe_deserialization()

import numpy as np
import json
import shutil
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from tqdm import tqdm

from utils import psnr_metric, ssim_metric, get_y_channel, reconstruct_rgb
from models import build_srcnn, build_edsr, pixel_shuffle_output_shape

NUM_CROP_PLOTS = 20 # Number of images to generate detailed crop plots
VAL_HR_DIR = "datasets/validation_Dataset"

def tiled_sr_predict(model, lr_arr, tile_size=192, overlap=8, batch_size=3):
    """
    Tiled inference fallback (used when TensorRT is unavailable)
    Splits LR into overlapping tiles, predicts in tile batches, blends via weighted average
    tile_size: LR tile side (192 -> 384 HR output per tile, 3GB activations per tile for EDSR_FULL)
    overlap: LR pixels of overlap on each border to avoid stitching artifacts (8 is the better option for qualtiy and speed)
    batch_size: tiles per predict call, we keep it low to avoid VRAM overflow (3 -> 9GB, good for 12GB GPUs)
    """
    h, w = lr_arr.shape[:2]
    out = np.zeros((h * 2, w * 2, 3), dtype=np.float32)
    weight = np.zeros((h * 2, w * 2, 1), dtype=np.float32)
    step = tile_size - 2 * overlap

    ys = list(range(0, h - tile_size, step)) + [max(0, h - tile_size)]
    xs = list(range(0, w - tile_size, step)) + [max(0, w - tile_size)]

    # Collect all tile crops (LR coordinates + normalized pixel values)
    coords = []
    tiles = []
    for y in dict.fromkeys(ys): # dict.fromkeys preserves order (also removes duplicates)
        for x in dict.fromkeys(xs):
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            coords.append((y, x, y2, x2))
            tiles.append(lr_arr[y:y2, x:x2].astype(np.float32) / 255.0)

    # Predict in batches, accumulate into output via weighted average to blend overlapping regions
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i:i + batch_size]
        batch_coords = coords[i:i + batch_size]
        preds = model.predict(np.stack(batch_tiles, axis=0), verbose=0)
        for pred, (y, x, y2, x2) in zip(preds, batch_coords):
            oy1, ox1, oy2, ox2 = y * 2, x * 2, y2 * 2, x2 * 2  # HR coordinates (x2 upscale)
            out[oy1:oy2, ox1:ox2] += np.clip(pred, 0, 1)
            weight[oy1:oy2, ox1:ox2] += 1.0

    return out / np.maximum(weight, 1.0)


def bicubic_psnr_ssim(hr_img, lr_img):
    """Returns (PSNR, SSIM) of bicubic upscaled lr_img vs hr_img, computed on the Y channel"""
    bicubic_up = lr_img.resize(hr_img.size, Image.Resampling.BICUBIC)
    hr_y, _, _ = get_y_channel(hr_img)
    bc_y, _, _ = get_y_channel(bicubic_up)
    hr_t = np.expand_dims(hr_y, axis=0)
    bc_t = np.expand_dims(bc_y, axis=0)
    return float(psnr_metric(hr_t, bc_t)), float(ssim_metric(hr_t, bc_t))


def run_evaluation(model_type):
    if model_type == "SRCNN":
        model_path = "best_srcnn.keras"
        output_dir = "validation_results_SRCNN"
    elif model_type == "EDSR_FULL":
        model_path = "best_edsr_full.keras"
        output_dir = "validation_results_EDSR_FULL"
    else:
        model_path = "best_edsr_baseline.keras"
        output_dir = "validation_results_EDSR"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {model_type} model: {model_path}...")
    custom_objs = {"psnr_metric": psnr_metric, "ssim_metric": ssim_metric, "pixel_shuffle_output_shape": pixel_shuffle_output_shape}

    if model_type == "EDSR_FULL":
        # FP16 needed: EDSR_FULL (43M params) is too large for FP32 inference on 12GB VRAM
        precision_label = "FP16 (mixed_float16)"
        keras.mixed_precision.set_global_policy("mixed_float16") # FP16 globally before our model is built
        model = build_edsr(num_blocks=32, num_filters=256)
        model.load_weights(model_path)
        keras.mixed_precision.set_global_policy("float32") # Reset globally to not interfere with the rest of the code
    else:
        precision_label = "FP32"
        try:
            model = keras.models.load_model(model_path, compile=False, safe_mode=False, custom_objects=custom_objs)
        except Exception as e:
            print(f"Loading failed: {e}. Reconstructing architecture...")
            model = build_edsr() if model_type == "EDSR" else build_srcnn()
            model.load_weights(model_path)
    model.compile(optimizer="adam", loss="mae", metrics=[psnr_metric])
    torch.cuda.empty_cache()

    TensorRT_infer = None
    if model_type == "EDSR_FULL":
        import torch_tensorrt

        # we had to define this, else we had compiling errors
        class _KerasWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x)

        device = next(iter(model.parameters())).device
        engine_path = "edsr_full_TensorRT.pt2"
        t_compile = time.perf_counter()
        if os.path.exists(engine_path):
            print(f"Loading cached TensorRT engine from {engine_path}...")
            TensorRT_infer = torch_tensorrt.load(engine_path).module()
            print(f"Engine loaded in {time.perf_counter() - t_compile:.1f}s")
        else:
            wrapper = _KerasWrapper(model).eval().to(device)
            print("No TensorRT engine found, compiling...")
            TensorRT_infer = torch_tensorrt.compile(
                wrapper,
                inputs=[torch_tensorrt.Input(shape=(1, 360, 640, 3),)],
                enabled_precisions={torch.float16},
                use_explicit_typing=False,
            )
            torch_tensorrt.save(TensorRT_infer, engine_path, inputs=[torch.zeros(1, 360, 640, 3, dtype=torch.float32, device=device)])
            print(f"Engine compiled and saved to {engine_path} ({os.path.getsize(engine_path) // (1024*1024)} MB) in {time.perf_counter() - t_compile:.1f}s")
        with torch.no_grad():
            _ = TensorRT_infer(torch.zeros(1, 360, 640, 3, dtype=torch.float32, device=device))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    all_hr_files = sorted(list(Path(VAL_HR_DIR).glob("*.png")))
    plot_hr_files = all_hr_files[:NUM_CROP_PLOTS]
    results = []

    print(f"Evaluating {model_type} [{precision_label}] on ALL {len(all_hr_files)} images (plots for {len(plot_hr_files)})...")

    inference_times = []

    for img_idx, hr_path in enumerate(tqdm(all_hr_files)):
        hr_img = Image.open(hr_path).convert("RGB")
        hr_arr_rgb = np.array(hr_img).astype("float32") / 255.0

        lr_img = hr_img.resize((hr_img.size[0] // 2, hr_img.size[1] // 2), Image.Resampling.BICUBIC)
        bicubic_upscaled = lr_img.resize(hr_img.size, Image.Resampling.BICUBIC)
        bicubic_arr_rgb = np.array(bicubic_upscaled).astype("float32") / 255.0
        bc_psnr, bc_ssim = bicubic_psnr_ssim(hr_img, lr_img)

        if model_type == "SRCNN":
            y_input, cb, cr = get_y_channel(bicubic_upscaled)
            input_arr = np.expand_dims(y_input, axis=0)
            t0 = time.perf_counter()
            sr_y_arr = model.predict(input_arr, verbose=0)[0]
            if img_idx > 0:
                inference_times.append((time.perf_counter() - t0) * 1000)
            sr_rgb_img = reconstruct_rgb(sr_y_arr, cb, cr)
            target_y, _, _ = get_y_channel(hr_img)
            pred_y = sr_y_arr
        else:
            lr_np = np.array(lr_img)
            t0 = time.perf_counter()
            if model_type == "EDSR_FULL":
                if TensorRT_infer is not None:
                    device = next(iter(model.parameters())).device
                    lr_t = torch.from_numpy(lr_np.astype("float32") / 255.0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        sr_t = TensorRT_infer(lr_t)
                    sr_rgb_arr = np.clip(sr_t.float().cpu().numpy()[0], 0, 1)
                else:
                    sr_rgb_arr = tiled_sr_predict(model, lr_np)
            else:
                sr_rgb_arr = np.clip(model.predict(
                    np.expand_dims(lr_np.astype("float32") / 255.0, axis=0), verbose=0)[0], 0, 1)
            if img_idx > 0:
                inference_times.append((time.perf_counter() - t0) * 1000)
            sr_rgb_img = Image.fromarray((sr_rgb_arr * 255.0).astype("uint8"))
            target_y, _, _ = get_y_channel(hr_img)
            pred_y, _, _ = get_y_channel(sr_rgb_img)

        hr_tensor = np.expand_dims(target_y, axis=0)
        sr_tensor = np.expand_dims(pred_y, axis=0)
        psnr_val = float(psnr_metric(hr_tensor, sr_tensor))
        ssim_val = float(ssim_metric(hr_tensor, sr_tensor))
        results.append({
            "name": hr_path.stem,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "bc_psnr": bc_psnr,
            "bc_ssim": bc_ssim,
        })

        # Detailed crop plots, only for the first NUM_CROP_PLOTS images
        if hr_path not in plot_hr_files:
            continue
        diff_map = np.abs(target_y[:, :, 0] - pred_y[:, :, 0])
        w, h = hr_img.size
        zoom_size = 300
        sy, sx = h // 2 - zoom_size // 2, w // 2 - zoom_size // 2
        def get_crop(arr): return arr[sy:sy+zoom_size, sx:sx+zoom_size]

        plt.figure(figsize=(22, 6))
        plt.subplot(1, 4, 1); plt.title("Original (Zoom)", pad=10)
        plt.imshow(get_crop(hr_arr_rgb)); plt.axis("off")
        plt.subplot(1, 4, 2); plt.title("Bicubic (Zoom)", pad=10)
        plt.imshow(get_crop(bicubic_arr_rgb)); plt.axis("off")
        plt.subplot(1, 4, 3)
        plt.title(f"{model_type} AI (Zoom)\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}", pad=10)
        plt.imshow(get_crop(np.array(sr_rgb_img).astype("float32") / 255.0)); plt.axis("off")
        ax_err = plt.subplot(1, 4, 4)
        plt.title("Error Map (Residual)", pad=10)
        im = ax_err.imshow(get_crop(diff_map), cmap='hot', vmin=0, vmax=1.0)
        plt.colorbar(im, ax=ax_err, fraction=0.046, pad=0.04); plt.axis("off")
        plt.tight_layout(pad=3.0); plt.subplots_adjust(top=0.85)
        plt.savefig(f"{output_dir}/detailed_{hr_path.stem}.png", bbox_inches='tight', dpi=200)
        plt.close()

    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])
    avg_bc_psnr = np.mean([r['bc_psnr'] for r in results])
    avg_bc_ssim = np.mean([r['bc_ssim'] for r in results])
    avg_inf = np.mean(inference_times)

    print(f"\n{'='*60}")
    print(f"  Method               | Avg PSNR (dB) | Avg SSIM | Inf (ms)")
    print(f"{'='*60}")
    print(f"  Bicubic              | {avg_bc_psnr:>13.2f} | {avg_bc_ssim:>8.4f} |   N/A")
    print(f"  {model_type} [{precision_label}]")
    print(f"                       | {avg_psnr:>13.2f} | {avg_ssim:>8.4f} | {avg_inf:>7.1f}")
    print(f"{'='*60}")
    print(f"  Delta PSNR           | {avg_psnr - avg_bc_psnr:>+13.2f} dB")

    summary = {
        "model": model_type,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "avg_bc_psnr": avg_bc_psnr,
        "avg_bc_ssim": avg_bc_ssim,
        "avg_inference_ms": avg_inf,
        "per_image": results,
    }
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(f"{output_dir}/detailed_report.txt", "w") as f:
        f.write(f"Detailed {model_type} Report\n")
        f.write(f"Avg PSNR: {avg_psnr:.2f} dB\nAvg SSIM: {avg_ssim:.4f}\n")
        f.write(f"Avg Bicubic PSNR: {avg_bc_psnr:.2f} dB\nAvg Bicubic SSIM: {avg_bc_ssim:.4f}\n")
        f.write(f"Avg Inference Time: {avg_inf:.1f} ms\n")

    # Per image PSNR line chart sorted by difficulty (bicubic psnr)
    sorted_r = sorted(results, key=lambda r: r['bc_psnr'])
    sorted_bc = [r['bc_psnr'] for r in sorted_r]
    sorted_m = [r['psnr'] for r in sorted_r]
    x = np.arange(len(sorted_r))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, sorted_bc, color='#5b9bd5', linewidth=1.5, label=f'Bicubic (avg {avg_bc_psnr:.2f} dB)')
    ax.plot(x, sorted_m,  color='#ed7d31', linewidth=1.5, label=f'{model_type} (avg {avg_psnr:.2f} dB)')
    ax.fill_between(x, sorted_bc, sorted_m, alpha=0.15, color='#ed7d31', label=f'Gain (avg {avg_psnr - avg_bc_psnr:+.2f} dB)')
    ax.set_xlabel('Images sorted by Bicubic PSNR (hardest -> easiest)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'Per-Image PSNR: {model_type} vs Bicubic (sorted by difficulty)')
    ax.legend(); ax.yaxis.set_minor_locator(ticker.AutoMinorLocator()); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/psnr_per_image.png", bbox_inches='tight', dpi=200); plt.close()
    print(f"Saved: {output_dir}/psnr_per_image.png")

    # PSNR improvement histogram
    deltas = [r['psnr'] - r['bc_psnr'] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(deltas, bins=10, color='#70ad47', edgecolor='white', linewidth=0.8)
    ax.axvline(np.mean(deltas), color='#c00000', linestyle='--', label=f'Mean delta = {np.mean(deltas):+.2f} dB')
    ax.set_xlabel('PSNR improvement over Bicubic (dB)'); ax.set_ylabel('Count')
    ax.set_title(f'{model_type} - PSNR Gain Distribution over Bicubic')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/psnr_delta_histogram.png", bbox_inches='tight', dpi=200); plt.close()
    print(f"Saved: {output_dir}/psnr_delta_histogram.png")

    # SSIM scatter
    model_ssims = [r['ssim'] for r in results]
    bc_ssims = [r['bc_ssim'] for r in results]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(bc_ssims, model_ssims, color='#ed7d31', edgecolors='gray', s=60, zorder=3)
    lims = [min(bc_ssims + model_ssims) - 0.002, max(bc_ssims + model_ssims) + 0.002]
    ax.plot(lims, lims, 'k--', linewidth=1, label='y = x (no gain)')
    ax.set_xlabel('Bicubic SSIM'); ax.set_ylabel(f'{model_type} SSIM')
    ax.set_title('SSIM: Model vs Bicubic (per image)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ssim_scatter.png", bbox_inches='tight', dpi=200); plt.close()
    print(f"Saved: {output_dir}/ssim_scatter.png")

    return summary


def plot_comparison_figures():
    output_dir = "report_figures"
    srcnn_hist = "history_srcnn.json"
    edsr_hist = "history_edsr.json"
    edsr_full_hist = "history_edsr_full.json"
    srcnn_res_file = "validation_results_SRCNN/results.json"
    edsr_res_file = "validation_results_EDSR/results.json"
    edsr_full_res_file = "validation_results_EDSR_FULL/results.json"

    Path(output_dir).mkdir(exist_ok=True)

    def plot_training_curves(hist_path, model_name, out_name):
        h = json.load(open(hist_path))
        epochs = np.arange(1, len(h['loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{model_name} - Training History", fontsize=14, fontweight='bold')
        ax = axes[0]
        ax.plot(epochs, h['loss'], color='#5b9bd5', linewidth=1.5, label='Train Loss')
        ax.plot(epochs, h['val_loss'], color='#ed7d31', linewidth=1.5, label='Val Loss')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MAE Loss'); ax.set_title('Loss (MAE)')
        ax.legend(); ax.grid(alpha=0.3); ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax = axes[1]
        ax.plot(epochs, h['psnr_metric'], color='#5b9bd5', linewidth=1.5, label='Train PSNR')
        ax.plot(epochs, h['val_psnr_metric'], color='#ed7d31', linewidth=1.5, label='Val PSNR')
        best_ep = int(np.argmax(h['val_psnr_metric'])) + 1
        best_val = max(h['val_psnr_metric'])
        ax.axvline(best_ep, color='green', linestyle=':', linewidth=1.5, label=f'Best epoch {best_ep} ({best_val:.2f} dB)')
        ax.set_xlabel('Epoch'); ax.set_ylabel('PSNR (dB)'); ax.set_title('PSNR')
        ax.legend(); ax.grid(alpha=0.3); ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{out_name}", bbox_inches='tight', dpi=200); plt.close()
        print(f"Saved: {output_dir}/{out_name}")

    plot_training_curves(srcnn_hist, "SRCNN", "training_curves_SRCNN.png")
    plot_training_curves(edsr_hist, "EDSR-Baseline", "training_curves_EDSR.png")
    plot_training_curves(edsr_full_hist, "EDSR-Full", "training_curves_EDSR_FULL.png")

    # Overlay: SRCNN vs EDSR vs EDSR_FULL validation curves
    hs = json.load(open(srcnn_hist))
    he = json.load(open(edsr_hist))
    hf = json.load(open(edsr_full_hist))
    ep_s = np.arange(1, len(hs['val_psnr_metric']) + 1)
    ep_e = np.arange(1, len(he['val_psnr_metric']) + 1)
    ep_f = np.arange(1, len(hf['val_psnr_metric']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SRCNN vs EDSR vs EDSR-Full - Validation Curves", fontsize=14, fontweight='bold')
    ax = axes[0]
    ax.plot(ep_s, hs['val_loss'], color='#5b9bd5', linewidth=1.5, label='SRCNN')
    ax.plot(ep_e, he['val_loss'], color='#ed7d31', linewidth=1.5, label='EDSR-Baseline')
    ax.plot(ep_f, hf['val_loss'], color='#70ad47', linewidth=1.5, label='EDSR-Full')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val MAE Loss'); ax.set_title('Validation Loss')
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(ep_s, hs['val_psnr_metric'], color='#5b9bd5', linewidth=1.5, label='SRCNN')
    ax.plot(ep_e, he['val_psnr_metric'], color='#ed7d31', linewidth=1.5, label='EDSR-Baseline')
    ax.plot(ep_f, hf['val_psnr_metric'], color='#70ad47', linewidth=1.5, label='EDSR-Full')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val PSNR (dB)'); ax.set_title('Validation PSNR')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_training_curves.png", bbox_inches='tight', dpi=200); plt.close()
    print(f"Saved: {output_dir}/comparison_training_curves.png")

    # Model comparison table
    srcnn_res = json.load(open(srcnn_res_file))
    edsr_res = json.load(open(edsr_res_file))
    edsr_full_res = json.load(open(edsr_full_res_file))
    rows = [
        ["Bicubic (baseline)", f"{srcnn_res['avg_bc_psnr']:.2f}", f"{srcnn_res['avg_bc_ssim']:.4f}", "-", "-"],
        ["SRCNN", f"{srcnn_res['avg_psnr']:.2f}", f"{srcnn_res['avg_ssim']:.4f}", f"{srcnn_res['avg_inference_ms']:.1f}", "~57K"],
        ["EDSR-Baseline", f"{edsr_res['avg_psnr']:.2f}", f"{edsr_res['avg_ssim']:.4f}", f"{edsr_res['avg_inference_ms']:.1f}", "~1.2M"],
        ["EDSR-Full", f"{edsr_full_res['avg_psnr']:.2f}", f"{edsr_full_res['avg_ssim']:.4f}", f"{edsr_full_res['avg_inference_ms']:.1f}", "~38.4M"],
    ]
    col_labels = ["Method", "PSNR (dB)", "SSIM", "Inf. time (ms)", "Params"]
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.6 * len(rows)))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.2, 2.0)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2e4057')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for col_idx in [1, 2]:
        vals = []
        for row_idx in range(1, len(rows) + 1):
            try:
                vals.append(float(tbl[row_idx, col_idx].get_text().get_text()))
            except ValueError:
                vals.append(-np.inf)
        best_row = int(np.argmax(vals)) + 1
        tbl[best_row, col_idx].set_facecolor('#c6efce')
        tbl[best_row, col_idx].set_text_props(fontweight='bold')
    fig.suptitle("Model Comparison - Super Resolution (2x)", fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison_table.png", bbox_inches='tight', dpi=200); plt.close()
    print(f"Saved: {output_dir}/model_comparison_table.png")

    # Combined per image PSNR line chart: all models sorted by difficulty
    edsr_map = {r['name']: r for r in edsr_res['per_image']}
    edsr_full_map = {r['name']: r for r in edsr_full_res['per_image']}
    common_sorted = sorted(
        [r for r in srcnn_res['per_image'] if r['name'] in edsr_map and r['name'] in edsr_full_map],
        key=lambda r: r['bc_psnr']
    )
    x = np.arange(len(common_sorted))
    bc_vals = [r['bc_psnr'] for r in common_sorted]
    srcnn_vals = [r['psnr'] for r in common_sorted]
    edsr_vals = [edsr_map[r['name']]['psnr'] for r in common_sorted]
    full_vals = [edsr_full_map[r['name']]['psnr'] for r in common_sorted]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, bc_vals, color='#5b9bd5', linewidth=1.5, label=f'Bicubic       (avg {np.mean(bc_vals):.2f} dB)')
    ax.plot(x, srcnn_vals, color='#ed7d31', linewidth=1.5, label=f'SRCNN         (avg {np.mean(srcnn_vals):.2f} dB)')
    ax.plot(x, edsr_vals, color='#70ad47', linewidth=1.5, label=f'EDSR-Baseline (avg {np.mean(edsr_vals):.2f} dB)')
    ax.plot(x, full_vals, color='#c00000', linewidth=1.5, label=f'EDSR-Full     (avg {np.mean(full_vals):.2f} dB)')
    ax.fill_between(x, bc_vals, srcnn_vals, alpha=0.08, color='#ed7d31')
    ax.fill_between(x, srcnn_vals, edsr_vals, alpha=0.08, color='#70ad47')
    ax.fill_between(x, edsr_vals, full_vals, alpha=0.08, color='#c00000')
    ax.set_xlabel('Images sorted by Bicubic PSNR (hardest -> easiest)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Per-Image PSNR: Bicubic vs SRCNN vs EDSR-Baseline vs EDSR-Full (sorted by difficulty)')
    ax.legend(); ax.yaxis.set_minor_locator(ticker.AutoMinorLocator()); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/psnr_comparison_all_models.png", bbox_inches='tight', dpi=200); plt.close()
    print(f"Saved: {output_dir}/psnr_comparison_all_models.png")

    print(f"\nAll comparison figures saved to '{output_dir}/'")


# For evaluation to run, the trained models (and their history), plus the validation HR images must exist!  
REQUIRED_FILES = [
    "best_srcnn.keras",
    "best_edsr_baseline.keras",
    "best_edsr_full.keras",
    "history_srcnn.json",
    "history_edsr.json",
    "history_edsr_full.json",
    VAL_HR_DIR,
]

if __name__ == "__main__":
    missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
    if missing:
        print("ERROR: Missing required files, run training first:")
        for f in missing:
            print(f"  x {f}")
        exit(1)

    run_evaluation("SRCNN")
    run_evaluation("EDSR")
    run_evaluation("EDSR_FULL")
    plot_comparison_figures()
