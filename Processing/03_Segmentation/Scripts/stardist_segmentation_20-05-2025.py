import math
import tifffile
import os
import time
import numpy as np
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# Author: Cruz Osuna (cruzosuna2003@gmail.com)
# Last update: 13/05/2025

# ======== Configuration ========
INPUT_PATH = "/media/cruz/Spatial/t-CycIF_human_2025/02_Visualization/t-CycIF/Images_IC/"
OUTPUT_PATH = "/media/cruz/Spatial/t-CycIF_human_2025/03_Segmentation/Mask/"
TILE_SIZE = 256  # Optimal for not high end GPUs (8-15 GB of VRAM), adjust if needed
# ===============================

def get_model_choice():
    """Interactive model selection with validation"""
    models = {
        '1': '2D_versatile_fluo',
        '2': '2D_versatile_he',
        '3': '2D_paper_dsb2018'
    }
    while True:
        print("\nSelect segmentation model:")
        print("1) 2D_versatile_fluo (Fluorescence microscopy images)")
        print("2) 2D_versatile_he (H&E stained histology)")
        print("3) 2D_paper_dsb2018 (DSB 2018 challenge dataset)")
        choice = input("Enter choice (1-3) [default=1]: ").strip() or '1'
        if choice in models:
            return models[choice]
        print(f"Invalid choice '{choice}' - please select 1, 2, or 3")

def get_threshold(prompt, default, min_val=0.0, max_val=1.0):
    while True:
        try:
            value = input(f"{prompt} [default={default:.2f}]: ") or default
            value = float(value)
            if min_val <= value <= max_val:
                return value
            print(f"Value must be between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid number")

def main():
    start_time = time.time()
    
    # ---------- Get User Parameters ----------
    print("\n=== Segmentation Parameters ===")
    model_name = get_model_choice()
    prob_thresh = get_threshold("Probability threshold (0-1)", 0.50)
    overlap_thresh = get_threshold("Overlap threshold (0-1)", 0.30)
    
    # ---------- Input Validation & Sorting ----------
    valid_extensions = (".tif", ".tiff")
    image_files = [f for f in os.listdir(INPUT_PATH) 
                  if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        raise FileNotFoundError(f"No TIFF files found in {INPUT_PATH}")

    # Sort images by file size (smallest first)
    image_files.sort(key=lambda f: os.path.getsize(os.path.join(INPUT_PATH, f)))
    
    print(f"\nFound {len(image_files)} images to process (ordered by size):")
    for idx, f in enumerate(image_files, 1):
        size = os.path.getsize(os.path.join(INPUT_PATH, f))
        print(f"{idx:3d}. {f:<40} ({size/1024/1024:.1f} MB)")

    # ---------- Output Preparation ----------
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ---------- Model Initialization ----------
    model = StarDist2D.from_pretrained(model_name)
    print(f"\nLoaded {model_name} model with thresholds:")
    print(f" - Probability: {prob_thresh:.2f}")
    print(f" - Overlap: {overlap_thresh:.2f}")

    # ---------- Processing Loop ----------
    for image_file in image_files:
        try:
            image_id = os.path.splitext(image_file)[0]
            input_path = os.path.join(INPUT_PATH, image_file)
            output_path = os.path.join(OUTPUT_PATH, f"{image_id}.ome.tif")

            print(f"\nProcessing: {image_file}")
            
            # Load image and add channel dimension
            img = tifffile.imread(input_path, key=0)
            print(f"Image shape: {img.shape}")

            # ---------- Dynamic Tiling ----------
            n_tiles = (
                math.ceil(img.shape[0] / TILE_SIZE),
                math.ceil(img.shape[1] / TILE_SIZE)
            )
            print(f"Using tile grid: {n_tiles}")

            # ---------- Processing ----------
            # Normalize and add channel dimension
            img_normalized = normalize(img)[..., np.newaxis]
            
            labels, _ = model.predict_instances(
                img_normalized,
                n_tiles=n_tiles,
                prob_thresh=prob_thresh,
                nms_thresh=overlap_thresh
            )
            
            # ---------- Save Results ----------
            tifffile.imwrite(output_path, labels.astype("int32"))
            print(f"Saved mask: {output_path}")

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

    # ---------- Final Report ----------
    elapsed = time.time() - start_time
    print(f"\nProcessing complete! Time: {elapsed:.2f} seconds")

if __name__ == '__main__':
    if input("Start processing? [y/n] ").lower() == "y":
        main()
    else:
        print("Operation cancelled")