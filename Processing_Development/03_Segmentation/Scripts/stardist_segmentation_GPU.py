import math
import tifffile
import os
import time
import numpy as np
import gc
import psutil
import platform  # Added to detect operating system
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import tensorflow as tf

# Author: Cruz Osuna (cruzosuna2003@gmail.com)
# Last update: 01/06/2025

# ======== Configuration ========
# Use raw strings for Windows paths or standard Linux paths
if platform.system() == 'Windows':
    # Windows paths (use either format)
    INPUT_PATH = r"\\NAS_BMF_LAB\Projects\t-CycIF\t-CycIF_human_2025_2\02_Visualization\t-CycIF\Images_IC"
    # INPUT_PATH = "Z:\\Projects\\t-CycIF\\t-CycIF_human_2025_2\\02_Visualization\\t-CycIF\\Images_IC"  # If mapped to drive
else:
    # Linux/Mac path
    INPUT_PATH = "/NAS_BMF_LAB/Projects/t-CycIF/t-CycIF_human_2025_2/02_Visualization/t-CycIF/Images_IC"

if platform.system() == 'Windows':
    OUTPUT_PATH = r"\\NAS_BMF_LAB\Projects\t-CycIF\t-CycIF_human_2025_2\03_Segmentation\Mask"
else:
    OUTPUT_PATH = "/NAS_BMF_LAB/Projects/t-CycIF/t-CycIF_human_2025_2/03_Segmentation/Mask/"

TILE_SIZE = 256
# ===============================

def clean_memory():
    """Comprehensive memory cleanup routine"""
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()
    if 'psutil' in globals():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"Memory after cleanup: {mem_info.rss/1024**2:.1f} MB")

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
    
    # Validate paths
    print(f"\nUsing input path: {INPUT_PATH}")
    print(f"Using output path: {OUTPUT_PATH}")
    
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input path does not exist: {INPUT_PATH}")
    
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

    image_files.sort(key=lambda f: os.path.getsize(os.path.join(INPUT_PATH, f)))
    
    print(f"\nFound {len(image_files)} images to process (ordered by size):")
    for idx, f in enumerate(image_files, 1):
        size = os.path.getsize(os.path.join(INPUT_PATH, f))
        print(f"{idx:3d}. {f:<40} ({size/1024/1024:.1f} MB)")

    # ---------- Output Preparation ----------
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ---------- Model Initialization ----------
    print(f"\nLoading {model_name} model...")
    model = StarDist2D.from_pretrained(model_name)
    print(f"Model loaded with thresholds:")
    print(f" - Probability: {prob_thresh:.2f}")
    print(f" - Overlap: {overlap_thresh:.2f}")

    # ---------- Processing Loop ----------
    for image_file in image_files:
        try:
            # Initialize variables for cleanup
            img = img_normalized = labels = None
            process = psutil.Process(os.getpid())
            
            image_id = os.path.splitext(image_file)[0]
            input_path = os.path.join(INPUT_PATH, image_file)
            output_path = os.path.join(OUTPUT_PATH, f"{image_id}.ome.tif")

            print(f"\nProcessing: {image_file}")
            
            # Load image and add channel dimension
            img = tifffile.imread(input_path, key=0)
            print(f"Image shape: {img.shape}")
            
            # Monitor memory before processing
            mem_start = process.memory_info().rss
            print(f"Memory before processing: {mem_start/1024**2:.1f} MB")

            # ---------- Dynamic Tiling ----------
            n_tiles = (
                math.ceil(img.shape[0] / TILE_SIZE),
                math.ceil(img.shape[1] / TILE_SIZE),
                1  # Channel dimension
            )
            print(f"Using tile grid: {n_tiles}")

            # ---------- Processing ----------
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
        finally:
            # ---------- Memory Cleanup ----------
            print("Performing memory cleanup...")
            
            # 1. Delete large objects
            del img, img_normalized, labels
            
            # 2. Clear TensorFlow session
            tf.keras.backend.clear_session()
            
            # 3. Force garbage collection
            gc.collect()
            
            # 4. Verify memory release
            mem_end = process.memory_info().rss
            print(f"Memory after cleanup: {mem_end/1024**2:.1f} MB")
            print(f"Memory released: {(mem_start - mem_end)/1024**2:.1f} MB")

    # ---------- Final Cleanup & Report ----------
    del model
    clean_memory()
    
    elapsed = time.time() - start_time
    print(f"\nProcessing complete! Time: {elapsed/60:.2f} minutes")

if __name__ == '__main__':
    print(f"Running on: {platform.system()}")
    if input("Start processing? [y/n] ").lower() == "y":
        main()
    else:
        print("Operation cancelled")