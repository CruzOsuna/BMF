import math
import tifffile
import os
import time
import logging
from pathlib import Path
from typing import Optional
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
MODEL_NAME = '2D_versatile_fluo'
INPUT_PATH = "/home/cruz/Escritorio/03_Segmentation/image"
OUTPUT_PATH = "/home/cruz/Escritorio/03_Segmentation/masks/"
SUPPORTED_EXTENSIONS = ('.tif', '.tiff')

def calculate_tiles(img_shape, tile_size: tuple) -> tuple:
    """Calculate optimal tile partitioning based on model's preferred tile size"""
    try:
        tiles_y = math.ceil(img_shape[0] / tile_size[0])
        tiles_x = math.ceil(img_shape[1] / tile_size[1])
        return (tiles_y, tiles_x)
    except Exception as e:
        logging.error(f"Error calculating tiles: {str(e)}")
        return (1, 1)  # Fallback to no tiling

def process_image(image_path: str, output_dir: str, model: StarDist2D) -> Optional[str]:
    """Process a single image and return output path if successful"""
    try:
        # Read image
        img = tifffile.imread(image_path, key=0)
        logging.info(f"Processing image: {image_path} (shape: {img.shape})")

        # Calculate optimal tiling
        tile_size = model.config.tile_size
        n_tiles = calculate_tiles(img.shape, tile_size)
        logging.debug(f"Using tiling configuration: {n_tiles}")

        # Process image
        labels, _ = model.predict_instances(normalize(img), n_tiles=n_tiles)
        labels = labels.astype("int32")

        # Save result
        output_path = Path(output_dir) / f"{Path(image_path).stem}.ome.tif"
        tifffile.imwrite(output_path, labels, ome=True)
        logging.info(f"Saved mask to: {output_path}")
        return str(output_path)

    except Exception as e:
        logging.error(f"Failed to process {image_path}: {str(e)}")
        return None

def main():
    # Verify paths
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Get valid image files
    image_files = [
        f for f in input_path.iterdir() 
        if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()
    ]

    if not image_files:
        logging.warning(f"No valid image files found in {input_path}")
        return

    # Load model
    try:
        model = StarDist2D.from_pretrained(MODEL_NAME)
        logging.info(f"Loaded model: {MODEL_NAME}")
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        return

    # User confirmation
    print(f"Found {len(image_files)} images to process")
    if input("Proceed with segmentation? [y/n]: ").lower() != 'y':
        logging.info("Processing cancelled by user")
        return

    # Process images
    start_time = time.time()
    success_count = 0

    for image_file in image_files:
        result = process_image(str(image_file), str(output_path), model)
        if result is not None:
            success_count += 1

    # Print summary
    elapsed_time = time.time() - start_time
    logging.info(
        f"Processed {success_count}/{len(image_files)} images successfully "
        f"in {elapsed_time:.2f} seconds"
    )

if __name__ == '__main__':
    main()