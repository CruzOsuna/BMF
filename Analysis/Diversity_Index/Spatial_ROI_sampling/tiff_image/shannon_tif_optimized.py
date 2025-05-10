import numpy as np
import pandas as pd
import tifffile
from scipy.interpolate import griddata
import os
import re
from shapely.geometry import Polygon
from skimage.draw import polygon

# ========================================================================
# CONFIGURATION
# ========================================================================
RADIUS_STEP = 50  
TIFF_ORIGINAL_PATH = "/media/cruz/Spatial/CycIF_human_2024/2_Visualization/t-CycIF/Images_IC/AGSCC_1.ome.tif"
X_RES = 1  # microns/pixel
Y_RES = 1
RESULTS_CSV = "/media/HDD_1/BMF/Spatial_sampling/Output/AGSCC_1/shannon_index_AGSCC_1.csv"

def create_sampling_area():
    POLYGON_FILE = '/media/HDD_1/BMF/Spatial_sampling/Shapes/AGSCC_1/AGSCC_1_Carcinoma-Stroma.txt'
    with open(POLYGON_FILE, 'r') as f:
        content = f.read()
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', content)
    coords = list(zip(map(float, numbers[::2]), map(float, numbers[1::2])))
    return Polygon(coords)

def generate_shannon_tiff(results_csv, step, output_dir, output_tiff_name):
    df = pd.read_csv(results_csv)
    sampling_area = create_sampling_area()
    points = df[['center_x', 'center_y']].values
    values = df[f'step_{step}'].values

    with tifffile.TiffFile(TIFF_ORIGINAL_PATH) as tif:
        width, height = tif.pages[0].shape[1], tif.pages[0].shape[0]

    # Rasterize polygon mask
    poly_coords = np.array(sampling_area.exterior.coords)
    poly_coords_pixels = (poly_coords / [X_RES, Y_RES]).astype(int)
    rr, cc = polygon(poly_coords_pixels[:, 1], poly_coords_pixels[:, 0], (height, width))
    mask = np.zeros((height, width), dtype=bool)
    mask[rr, cc] = True

    # Initialize output array with NaNs
    grid_z = np.full((height, width), np.nan, dtype=np.float32)

    # Process in chunks to save memory
    chunk_size = 500  # Adjust based on available RAM
    for y in range(0, height, chunk_size):
        y_end = min(y + chunk_size, height)
        for x in range(0, width, chunk_size):
            x_end = min(x + chunk_size, width)
            
            # Get chunk indices
            chunk_slice = (slice(y, y_end), slice(x, x_end))
            chunk_mask = mask[chunk_slice]
            
            if not np.any(chunk_mask):
                continue  # Skip empty chunks
            
            # Generate grid points for the chunk
            x_coords = np.linspace(x * X_RES, x_end * X_RES, x_end - x)
            y_coords = np.linspace(y * Y_RES, y_end * Y_RES, y_end - y)
            grid_x, grid_y = np.meshgrid(x_coords, y_coords)
            
            # Interpolate values for this chunk
            chunk_z = griddata(points, values, (grid_x, grid_y), method='linear')
            
            # Apply mask and store in output
            grid_z_chunk = grid_z[chunk_slice]
            grid_z_chunk[chunk_mask] = chunk_z[chunk_mask]
            grid_z[chunk_slice] = grid_z_chunk

    # Fill NaNs with minimum value
    min_val = np.nanmin(values)
    grid_z = np.nan_to_num(grid_z, nan=min_val)

    # Save as 32-bit float TIFF
    output_path = os.path.join(output_dir, output_tiff_name)
    tifffile.imwrite(output_path, grid_z.astype(np.float32), photometric='minisblack')
    print(f"TIFF saved at: {output_path}")

if __name__ == "__main__":
    generate_shannon_tiff(RESULTS_CSV, RADIUS_STEP, 
                         "/media/HDD_1/BMF/Spatial_sampling/Output/AGSCC_1/",
                         "shannon_map_AGSCC_1_step_50.tif")