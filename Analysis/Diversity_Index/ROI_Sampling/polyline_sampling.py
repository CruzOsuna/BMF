import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
import random
from scipy.spatial import KDTree
import math
import re

# ========================================================================
# CONFIGURABLE PARAMETERS
# ========================================================================
USE_LINE_BUFFER = False   # True: Generate around line, False: Use polygon
NUM_POINTS = 100        
SIDE_DISTANCE = 100      # Buffer width (only for line mode)
STEP_SIZE = 10          
MAX_STEPS = 100          

# File paths (choose based on mode)
LINE_FILE = '/home/cruz/Escritorio/BMF_t-CyCIF/Analysis/Diversity_Index/ROI_Sampling/Shape/Line_try1.txt'  # For line mode
POLYGON_FILE = '/home/cruz/Escritorio/BMF_t-CyCIF/Analysis/Diversity_Index/ROI_Sampling/Shape/polygon_try.txt'      # For polygon mode

CELLS_FILE = '/home/cruz/Escritorio/BMF_t-CyCIF/Analysis/Diversity_Index/ROI_Sampling/FAHNSCC_14_phenotype_annotated.csv'
SAMPLE_NAME = 'FAHNSCC_14'  
OUTPUT_FILE = 'shannon_results.csv'
POINTS_LAYER_FILE = 'sampling_points.csv'
# ========================================================================

def save_napari_points_layer(points, file_path):
    """Save points layer with numeric labels"""
    df = pd.DataFrame(points, columns=['x', 'y'])
    df['label'] = df.index + 1
    df.to_csv(file_path, index=False)
    print(f"Saved {len(points)} sampling points to {file_path}")

def load_geometry(file_path):
    """Load coordinates from text file for line or polygon"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    points = re.findall(r'\[\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*\]', content)
    return np.array([[float(x), float(y)] for x, y in points])

def create_sampling_area():
    """Create geometry based on selected mode"""
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
    """Generate points within any Shapely polygon"""
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []
    while len(points) < num_points:
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        if polygon.contains(Point(x, y)):
            points.append((x, y))
    return points

# ========================================================================
# MAIN PROCESSING
# ========================================================================
def main():
    # Create sampling area
    sampling_area = create_sampling_area()
    
    # Generate and save points
    centroids = generate_random_points(sampling_area, NUM_POINTS)
    save_napari_points_layer(centroids, POINTS_LAYER_FILE)

    # Load cell data
    cells_df = pd.read_csv(CELLS_FILE)
    if 'Sample' not in cells_df.columns:
        raise ValueError("CSV missing 'Sample' column")
    
    sample_cells = cells_df[cells_df['Sample'] == SAMPLE_NAME]
    if sample_cells.empty:
        raise ValueError(f"Sample '{SAMPLE_NAME}' not found")

    # Prepare analysis
    cell_coords = sample_cells[['X_centroid', 'Y_centroid']].values
    phenotypes = sample_cells['phenotype_key'].values
    kdtree = KDTree(cell_coords)
    
    # Calculate Shannon indices
    results = []
    for cx, cy in centroids:
        row = {'sample': SAMPLE_NAME, 'center_x': cx, 'center_y': cy}
        for step in range(1, MAX_STEPS + 1):
            radius = STEP_SIZE * step
            indices = kdtree.query_ball_point([cx, cy], radius)
            
            if indices:
                counts = pd.Series(phenotypes[indices]).value_counts()
                proportions = counts / counts.sum()
                shannon = -sum(p * math.log(p) for p in proportions if p > 0)
            else:
                shannon = 0.0
            
            row[f'step_{step}'] = shannon
        results.append(row)
    
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()