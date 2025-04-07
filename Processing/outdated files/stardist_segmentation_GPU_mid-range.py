import math
import tifffile
import os
import time
import gc
from csbdeep.utils import normalize
from stardist.models import StarDist2D
import tensorflow as tf
from keras import backend as K

# Configure GPU memory management
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=7168)]  # 7GB limit
        )
    except RuntimeError as e:
        print("Error during GPU configuration:", e)

INPUT_PATH = "/media/cruz/Spatial/CycIF_human_2024/2_Visualization/t-CycIF/Images_illumination-corrected"
OUTPUT_PATH = "/media/cruz/Spatial/CycIF_human_2024/3_Segmentation/Mask_illumination-corrected/"
model_name = '2D_versatile_fluo'

def calculate_tiles(img_shape):
    """Dynamic tile sizing with even smaller tiles for very large images"""
    max_pixels = max(img_shape)
    if max_pixels > 50000:
        return (math.ceil(img_shape[0]/64), math.ceil(img_shape[1]/64))  # 64x64 tiles
    elif max_pixels > 30000:
        return (math.ceil(img_shape[0]/128), math.ceil(img_shape[1]/128))
    elif max_pixels > 15000:
        return (math.ceil(img_shape[0]/256), math.ceil(img_shape[1]/256))
    return (math.ceil(img_shape[0]/512), math.ceil(img_shape[1]/512))

if __name__ == '__main__':
    start_time = time.time()
    
    model = StarDist2D.from_pretrained(model_name)
    arr = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith(('.tif', '.tiff'))]
    
    if input(f"Process {len(arr)} images? [y/n]: ").lower() == 'y':
        for idx, image_file in enumerate(arr, 1):
            image_path = os.path.join(INPUT_PATH, image_file)
            output_path = os.path.join(OUTPUT_PATH, os.path.splitext(image_file)[0] + ".ome.tif")
            
            print(f"\nProcessing image {idx}/{len(arr)}: {image_file}")
            img = tifffile.imread(image_path, key=0)
            print(f"Dimensions: {img.shape} | Dtype: {img.dtype}")
            
            n_tiles = calculate_tiles(img.shape)
            print(f"Tile configuration: {n_tiles[0]}x{n_tiles[1]}")
            
            labels = None  # Initialize labels variable
            try:
                # Corrected prediction call without batch_size
                labels, _ = model.predict_instances(
                    normalize(img),
                    n_tiles=n_tiles,
                    verbose=0
                )
                tifffile.imwrite(output_path, labels.astype('int32'))
                print(f"Successfully saved: {output_path}")
            except Exception as e:
                print(f"Failed processing {image_file}: {str(e)}")
            finally:
                # Safe cleanup
                del img
                if labels is not None:
                    del labels
                K.clear_session()
                gc.collect()

    print(f"\nTotal execution time: {(time.time()-start_time)/60:.2f} minutes")