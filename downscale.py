# Dimitris Karatzas aivc25007

import os
from pathlib import Path
from PIL import Image
from utils import downscale_image

def process_dataset(hr_dir_name, lr_dir_name, factor=2):
    """Bicubic downscale of all PNG/JPG images in hr_dir, saving results as lossless PNG in lr_dir."""
    hr_dir = Path(hr_dir_name)
    lr_dir = Path(lr_dir_name)
    lr_dir.mkdir(parents=True, exist_ok=True)

    # Filter for standard image formats (although my dataset is png ONLY because of lossless compression)
    files = [f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort()

    print(f"\nProcessing {hr_dir_name} -> {lr_dir_name} (factor: {factor}x)")
    
    count = 0
    for filename in files:
        hr_path = hr_dir / filename
        lr_path = lr_dir / filename
        
        try:
            with Image.open(hr_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                lr_img = downscale_image(img, factor=factor)
                # Save as PNG to maintain quality (it is lossless)
                lr_img.save(lr_path, "PNG")
            
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} images...")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Done. Processed {count} images in {lr_dir}")

if __name__ == "__main__":
    process_dataset('datasets/APISR_Dataset', 'datasets/APISR_Dataset_LR')
