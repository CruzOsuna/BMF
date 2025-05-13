import math
import tifffile
import os
import time
import tensorflow as tf
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# ======== Forced CPU Configuration ========
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
tf.config.set_visible_devices([], 'GPU')  # Ensure TF does not use GPU
# ===========================================

# Author: Cruz Osuna (cruzosuna2003@gmail.com)

# ======== Configuration ========
INPUT_PATH = "/media/cruz/Spatial/t-CycIF_human_2025/02_Visualization/t-CycIF/Images_IC/"
OUTPUT_PATH = "/media/cruz/Spatial/t-CycIF_human_2025/03_Segmentation/Mask/"
TILE_SIZE = 512  # Maximum possible size for CPU (adjust according to RAM)
# ===============================

def get_model_choice():
    """Interactive model selection"""
    models = {
        '1': '2D_versatile_fluo',
        '2': '2D_versatile_he',
        '3': '2D_paper_dsb2018'
    }
    while True:
        print("\nSelect the segmentation model:")
        print("1) 2D_versatile_fluo (Fluorescence microscopy)")
        print("2) 2D_versatile_he (H&E histology)")
        print("3) 2D_paper_dsb2018 (DSB 2018 dataset)")
        choice = input("Choice (1-3) [default=1]: ").strip() or '1'
        if choice in models:
            return models[choice]
        print(f"Invalid choice '{choice}' - choose 1, 2, or 3")

def get_threshold(prompt, default, min_val=0.0, max_val=1.0):
    while True:
        try:
            value = input(f"{prompt} [default={default:.2f}]: ") or default
            value = float(value)
            if min_val <= value <= max_val:
                return value
            print(f"The value must be between {min_val} and {max_val}")
        except ValueError:
            print("Enter a valid number")

def main():
    start_time = time.time()
    
    # ---------- TensorFlow CPU Configuration ----------
    tf.config.threading.set_intra_op_parallelism_threads(8)  # Adjust according to your CPU
    tf.config.threading.set_inter_op_parallelism_threads(8)
    
    # ---------- User parameters ----------
    print("\n=== Segmentation Parameters ===")
    model_name = get_model_choice()
    prob_thresh = get_threshold("Probability threshold (0-1)", 0.50)
    overlap_thresh = get_threshold("Overlap threshold (0-1)", 0.30)
    
    # ---------- File validation ----------
    valid_extensions = (".tif", ".tiff")
    image_files = [f for f in os.listdir(INPUT_PATH) 
                  if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        raise FileNotFoundError(f"No TIFF files found in {INPUT_PATH}")

    image_files.sort(key=lambda f: os.path.getsize(os.path.join(INPUT_PATH, f)))
    
    print(f"\nImages to process ({len(image_files)} sorted by size):")
    for idx, f in enumerate(image_files, 1):
        size = os.path.getsize(os.path.join(INPUT_PATH, f))
        print(f"{idx:3d}. {f:<40} ({size/1024/1024:.1f} MB)")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # ---------- Load model ----------
    model = StarDist2D.from_pretrained(model_name)
    print(f"\nModel {model_name} loaded with thresholds:")
    print(f" - Probability: {prob_thresh:.2f}")
    print(f" - Overlap: {overlap_thresh:.2f}")

    # ---------- Processing with memory cleanup ----------
    from csbdeep.utils.tf import keras_import
    keras = keras_import()
    
    for image_file in image_files:
        try:
            image_id = os.path.splitext(image_file)[0]
            input_path = os.path.join(INPUT_PATH, image_file)
            output_path = os.path.join(OUTPUT_PATH, f"{image_id}.ome.tif")

            print(f"\nProcessing: {image_file}")
            
            img = tifffile.imread(input_path, key=0)
            print(f"Image dimensions: {img.shape}")

            # Block processing
            labels = np.zeros_like(img, dtype="int32")
            for i in range(0, img.shape[0], TILE_SIZE):
                for j in range(0, img.shape[1], TILE_SIZE):
                    tile = img[i:i+TILE_SIZE, j:j+TILE_SIZE]
                    lbl, _ = model.predict_instances(
                        normalize(tile),
                        prob_thresh=prob_thresh,
                        nms_thresh=overlap_thresh
                    )
                    labels[i:i+TILE_SIZE, j:j+TILE_SIZE] = lbl
            
            tifffile.imwrite(output_path, labels.astype("int32"))
            print(f"Mask saved: {output_path}")

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
        finally:
            # Aggressive memory cleanup
            keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            time.sleep(1)  # Pause for GC

    elapsed = time.time() - start_time
    print(f"\nProcessing complete! Total time: {elapsed/60:.2f} minutes")

if __name__ == '__main__':
    if input("Start processing? [y/n] ").lower() == "y":
        main()
    else:
        print("Process canceled")
