import numpy as np
import pandas as pd
import tifffile
from scipy.interpolate import griddata
import os
import re
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

# ========================================================================
# CONFIGURATION
# ========================================================================
RADIUS_STEP = 50  
TIFF_ORIGINAL_PATH = "/media/HDD_1/BMF/FA/visualization/images/FAHNSCC_14.ome.tiff"
X_RES = 1  # microns/pixel
Y_RES = 1

# Specify the path to the .csv file with Shannon index data
RESULTS_CSV = "/media/HDD_1/BMF/Spatial_sampling/Output/FAHNSCC_14/simpson_index_FAHNSCC_14.csv"

def create_sampling_area():
    """Parse coordinates from a NumPy array formatted file."""
    POLYGON_FILE = '/media/HDD_1/BMF/Spatial_sampling/Shapes/FAHNSCC_14/FAHNSCC_14_partial_S1.txt'
    with open(POLYGON_FILE, 'r') as f:
        content = f.read()
    
    # Extract all numbers from the content, including negatives and decimals
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', content)
    numbers = list(map(float, numbers))
    
    # Convert to (x, y) pairs
    coords = list(zip(numbers[::2], numbers[1::2]))
    
    return Polygon(coords)

def generate_shannon_tiff(results_csv, step, output_dir, output_tiff_name):
    """Generate the TIFF with data validation."""
    df = pd.read_csv(results_csv)
    sampling_area = create_sampling_area()
    points = df[['center_x', 'center_y']].values
    values = df[f'step_{step}'].values

    print(f"Simpson values (min, max): {np.nanmin(values)}, {np.nanmax(values)}")

    with tifffile.TiffFile(TIFF_ORIGINAL_PATH) as tif:
        width, height = tif.pages[0].shape[1], tif.pages[0].shape[0]
    
    x_coords = np.linspace(0, width * X_RES, width)
    y_coords = np.linspace(0, height * Y_RES, height)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    grid_z = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=np.nan)
    roi_mask = np.array([sampling_area.contains(Point(x, y)) for x, y in zip(grid_x.ravel(), grid_y.ravel())])
    roi_mask = roi_mask.reshape(grid_x.shape)
    grid_z[~roi_mask] = np.nan

    grid_z_filled = np.nan_to_num(grid_z, nan=np.nanmin(values))

    output_path = os.path.join(output_dir, output_tiff_name)
    tifffile.imwrite(output_path, grid_z_filled.T)
    print(f"TIFF saved at: {output_path}")

    plt.imshow(grid_z_filled, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title("Simpson Map")
    plt.show()

# ========================================================================
# Execution
# ========================================================================
if __name__ == "__main__":
    SAMPLE_NAME = "FAHNSCC_14"
    OUTPUT_DIR = "/media/HDD_1/BMF/Spatial_sampling/Output/FAHNSCC_14/"
    OUTPUT_TIFF_NAME = f'Simpson_map_{SAMPLE_NAME}_step_{RADIUS_STEP}.tif'
    
    # Use the RESULTS_CSV variable defined above
    generate_shannon_tiff(RESULTS_CSV, RADIUS_STEP, OUTPUT_DIR, OUTPUT_TIFF_NAME)
