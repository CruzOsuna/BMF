import math
import tifffile
import os
import time
import multiprocessing
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# Configurations (unchanged)
INPUT_PATH = "/home/cruz/Escritorio/03_Segmentation/image"
OUTPUT_PATH = "/home/cruz/Escritorio/03_Segmentation/masks/"
MODEL_NAME = '2D_versatile_fluo'
NUM_WORKERS = 2  # Adjust based on your GPU VRAM (RTX 4060: start with 2-3)

def process_image(image_name):
    model = StarDist2D.from_pretrained(MODEL_NAME)
    sep = '.'
    imageid = image_name.split(sep, 1)[0]
    IMAGE_PATH = os.path.join(INPUT_PATH, image_name)
    
    img = tifffile.imread(IMAGE_PATH, key=0)
    tiles = int(math.sqrt(int(img.shape[0]*img.shape[1]/(4781712.046875))))
    
    labels, _ = model.predict_instances(normalize(img), n_tiles=(tiles, tiles))
    output_name = os.path.join(OUTPUT_PATH, f"{imageid}.ome.tif")
    tifffile.imwrite(output_name, labels.astype("int32"))
    return image_name

if __name__ == '__main__':
    start_time = time.time()
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    arr = os.listdir(INPUT_PATH)

    print(f"Found {len(arr)} images")
    a = input("Proceed with parallel processing? [y/n] ")
    
    if a.lower() == "y":
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            results = pool.imap_unordered(process_image, arr)
            for i, res in enumerate(results, 1):
                print(f"Completed {i}/{len(arr)}: {res}")
    
    print(f"Total time: {time.time()-start_time:.2f}s")