import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
import random
import math
import re
import time
from numba import njit
from scipy.spatial import KDTree  # Keep import for reference


# Author: Cruz Osuna
# Script optimized with numba, with a 67.37% performance improvement over the non-optimized version.

# ========================================================================
# CONFIGURABLE PARAMETERS
# ========================================================================
USE_LINE_BUFFER = False   # True: Generate buffer around line, False: Use polygon
NUM_POINTS = 1000         # Number of sampling points to generate
SIDE_DISTANCE = 100       # Buffer width in pixels (line mode only)
STEP_SIZE = 10            # Radius increment per step
MAX_STEPS = 100           # Maximum number of radial steps

# File paths (mutually exclusive based on mode)
LINE_FILE = '/home/cruz/Escritorio/BMF_t-CyCIF/Analysis/Diversity_Index/ROI_Sampling/Shape/Line_try1.txt'    # For line buffer mode
POLYGON_FILE = '/home/cruz/Escritorio/BMF_t-CyCIF/Analysis/Diversity_Index/ROI_Sampling/Shape/polygon_try.txt'  # For polygon mode

CELLS_FILE = '/home/cruz/Escritorio/BMF_t-CyCIF/Analysis/Diversity_Index/ROI_Sampling/FAHNSCC_14_phenotype_annotated.csv'          # Cell position and phenotype data
SAMPLE_NAME = 'FAHNSCC_14'                     # Target sample name in cell data
OUTPUT_FILE = 'shannon_results.csv'            # Diversity index output
POINTS_LAYER_FILE = 'sampling_points.csv'      # Sampling coordinates output

# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def timer(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Executed {func.__name__} in {end-start:.2f} seconds")
        return result
    return wrapper

def save_napari_points_layer(points, file_path):
    """Save sampling points with numeric labels for visualization"""
    df = pd.DataFrame(points, columns=['x', 'y'])
    df['label'] = df.index + 1
    df.to_csv(file_path, index=False)
    print(f"Saved {len(points)} sampling points to {file_path}")

def load_geometry(file_path):
    """Parse coordinate pairs from text file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    points = re.findall(r'\[\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*\]', content)
    return np.array([[float(x), float(y)] for x, y in points])

def create_sampling_area():
    """Create Shapely geometry based on selected mode"""
    if USE_LINE_BUFFER:
        line_points = load_geometry(LINE_FILE)
        line = LineString(line_points)
        return line.buffer(SIDE_DISTANCE)
    else:
        poly_points = load_geometry(POLYGON_FILE)
        poly = Polygon(poly_points)
        if not poly.is_valid:
            raise ValueError("Invalid polygon geometry")
        return poly

def generate_random_points(polygon, num_points):
    """Generate random points within polygon bounds"""
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []
    while len(points) < num_points:
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        if polygon.contains(Point(x, y)):
            points.append((x, y))
    return np.array(points)  # Convert to numpy array for Numba

# ========================================================================
# NUMBA-OPTIMIZED CALCULATIONS
# ========================================================================

@njit
def calculate_shannon(unique_ids, counts):
    """Calculate Shannon index from unique phenotype counts"""
    proportions = counts / counts.sum()
    return -np.sum(proportions * np.log(proportions))

@njit
def process_centroid(cx, cy, cell_coords, phenotypes, step_size, max_steps):
    """Calculate diversity indices for all radial steps at one centroid"""
    results = np.zeros(max_steps)
    
    # Pre-calculate distances from centroid to all cells
    dx = cell_coords[:,0] - cx
    dy = cell_coords[:,1] - cy
    distances = np.sqrt(dx**2 + dy**2)
    
    for step in range(max_steps):
        radius = step_size * (step + 1)
        mask = distances <= radius
        
        if np.any(mask):
            # Get unique phenotypes in radius using bincount
            selected = phenotypes[mask]
            counts = np.bincount(selected)
            unique_ids = np.where(counts > 0)[0]
            counts = counts[unique_ids]
            
            if len(unique_ids) == 0:
                results[step] = 0.0
            else:
                results[step] = calculate_shannon(unique_ids, counts)
        else:
            results[step] = 0.0
            
    return results

# ========================================================================
# MAIN PROCESSING
# ========================================================================

@timer
def main():
    """Main analysis workflow with timing"""
    # Create sampling geometry
    sampling_area = create_sampling_area()
    
    # Generate sampling points
    centroids = generate_random_points(sampling_area, NUM_POINTS)
    save_napari_points_layer(centroids, POINTS_LAYER_FILE)

    # Load and prepare cell data
    cells_df = pd.read_csv(CELLS_FILE)
    sample_cells = cells_df[cells_df['Sample'] == SAMPLE_NAME]
    
    # Convert to numpy arrays and factorize phenotypes
    cell_coords = sample_cells[['X_centroid', 'Y_centroid']].values.astype(np.float64)
    phenotypes, _ = pd.factorize(sample_cells['phenotype_key'])
    phenotypes = phenotypes.astype(np.int32)

    # Calculate diversity indices for all centroids
    results = []
    for cx, cy in centroids:
        shannon_values = process_centroid(cx, cy, cell_coords, phenotypes, STEP_SIZE, MAX_STEPS)
        result_row = {
            'sample': SAMPLE_NAME,
            'center_x': cx,
            'center_y': cy,
            **{f'step_{i+1}': val for i, val in enumerate(shannon_values)}
        }
        results.append(result_row)
    
    # Save final results
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print("Analysis completed successfully")

if __name__ == "__main__":
    main()