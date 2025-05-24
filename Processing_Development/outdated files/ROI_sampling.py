import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
import time
import os
import re
import geopandas as gpd
from numba import njit, config
from scipy.spatial import KDTree
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

# ========================================================================
# CONFIGURATION
# ========================================================================

config.THREADING_LAYER = 'omp'  # OpenMP threading configuration

# Configuration parameters
USE_LINE_BUFFER = False
NUM_POINTS = 100000
SIDE_DISTANCE = 100
STEP_SIZE = 10
MAX_STEPS = 100

# Metric selection
METRICS = {
    1: "Shannon",
    2: "Ripley_K"
}
selected = int(input("Choose metric to calculate:\n1) Shannon Index\n2) Ripley K Function\nOption: "))
METRIC = METRICS[selected]
print(f"\nCalculating metric: {METRIC}\n")

# Input files
LINE_FILE = '/media/HDD_1/ROI_Sampling-benchmark/input/line_try-3.txt'
POLYGON_FILE = '/media/HDD_1/ROI_Sampling-benchmark/input/Square_3.txt'
CELLS_FILE = '/media/HDD_1/ROI_Sampling-benchmark/input/aggregated_data.csv'
SAMPLE_NAME = 'FAHNSCC_14'

# Output files
OUTPUT_DIR = '/media/HDD_1/ROI_Sampling-benchmark/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
POINTS_LAYER_FILE = f'sampling_points_{SAMPLE_NAME}.csv'

# ========================================================================
# SAMPLING AREA CREATION FUNCTION
# ========================================================================

def create_sampling_area():
    """Create sampling area from either line buffer or polygon file"""
    if USE_LINE_BUFFER:
        with open(LINE_FILE) as f:
            content = f.read()
        numbers = list(map(float, re.findall(r'\d+\.\d+', content)))
        coords = list(zip(numbers[::2], numbers[1::2]))
        line = LineString(coords)
        return line.buffer(SIDE_DISTANCE, cap_style=2)
    else:
        with open(POLYGON_FILE) as f:
            content = f.read()
        numbers = list(map(float, re.findall(r'\d+\.\d+', content)))
        coords = list(zip(numbers[::2], numbers[1::2]))
        return Polygon(coords)

# ========================================================================
# NAPARI POINTS LAYER SAVER
# ========================================================================

def save_napari_points_layer(centroids, filename):
    """Save centroids in Napari-compatible CSV format"""
    df = pd.DataFrame(centroids, columns=['center_x', 'center_y'])
    df['name'] = [f'centroid_{i}' for i in range(len(df))]
    df.to_csv(filename, index=False)

# ========================================================================
# VECTORIZED RANDOM POINT GENERATOR
# ========================================================================

def generate_random_points(polygon, num_points):
    """Generate random points within polygon using vectorized GeoPandas checks."""
    min_x, min_y, max_x, max_y = polygon.bounds
    points = np.zeros((num_points, 2), dtype=np.float32)
    generated = 0
    area_ratio = polygon.area / ((max_x - min_x) * (max_y - min_y))
    
    while generated < num_points:
        remaining = num_points - generated
        batch_size = max(int(remaining / area_ratio * 1.5), 1000)
        batch_size = min(batch_size, 1000000)
        
        x = np.random.uniform(min_x, max_x, batch_size)
        y = np.random.uniform(min_y, max_y, batch_size)
        
        points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y))
        mask = points_gdf.within(polygon)
        valid = np.column_stack((x[mask], y[mask]))
        
        if valid.size > 0:
            needed = min(remaining, valid.shape[0])
            points[generated:generated+needed] = valid[:needed]
            generated += needed
            
    return points

# ========================================================================
# OPTIMIZED CENTROID PROCESSING FUNCTION
# ========================================================================

@njit
def process_centroid(cx, cy, cell_coords, phenotypes, indices, step_size, max_steps, max_phenotype, metric):
    """Process centroid for selected metric"""
    subset_coords = cell_coords[indices]
    dx = subset_coords[:, 0] - cx
    dy = subset_coords[:, 1] - cy
    squared_distances = dx**2 + dy**2
    sorted_idx = np.argsort(squared_distances)
    sorted_sq_dist = squared_distances[sorted_idx]
    
    if metric == "Shannon":
        subset_pheno = phenotypes[indices]
        sorted_pheno = subset_pheno[sorted_idx]
        results = np.zeros(max_steps)
        cumulative_counts = np.zeros(max_phenotype + 1, dtype=np.int32)
        last_idx = 0
        
        for step in range(max_steps):
            current_radius = step_size * (step + 1)
            sq_radius = current_radius ** 2
            current_idx = np.searchsorted(sorted_sq_dist, sq_radius, side='right')
            new_pheno = sorted_pheno[last_idx:current_idx]
            
            counts = np.bincount(new_pheno, minlength=max_phenotype+1)
            cumulative_counts += counts
            last_idx = current_idx
            
            total = cumulative_counts.sum()
            if total == 0:
                results[step] = 0.0
                continue
                
            proportions = cumulative_counts[cumulative_counts > 0] / total
            results[step] = -np.sum(proportions * np.log(proportions))
            
        return results
    
    elif metric == "Ripley_K":
        results = np.zeros(max_steps)
        area = np.pi * (step_size * np.arange(1, max_steps+1))**2
        total_cells = len(subset_coords)
        lambda_val = total_cells / (np.max(area) if total_cells > 0 else 1.0
        
        for step in range(max_steps):
            current_radius = step_size * (step + 1)
            current_idx = np.searchsorted(sorted_sq_dist, current_radius**2, side='right')
            results[step] = current_idx / (lambda_val * area[step]) if lambda_val > 0 else 0
            
        return results

# ========================================================================
# SHARED MEMORY WRAPPER
# ========================================================================

def process_wrapper(args):
    """Multiprocessing wrapper with metric support"""
    (cx, cy, indices,
     shm_coords_name, coords_shape, coords_dtype,
     shm_pheno_name, pheno_shape, pheno_dtype,
     step_size, max_steps, max_phenotype, metric) = args
    
    shm_coords = shared_memory.SharedMemory(name=shm_coords_name)
    shm_pheno = shared_memory.SharedMemory(name=shm_pheno_name)
    
    cell_coords = np.ndarray(coords_shape, dtype=coords_dtype, buffer=shm_coords.buf)
    phenotypes = np.ndarray(pheno_shape, dtype=pheno_dtype, buffer=shm_pheno.buf)
    
    result = process_centroid(cx, cy, cell_coords, phenotypes, indices,
                            step_size, max_steps, max_phenotype, metric)
    
    shm_coords.close()
    shm_pheno.close()
    
    return result

# ========================================================================
# MAIN FUNCTION
# ========================================================================

def main():
    start_total = time.perf_counter()
    
    # Create sampling area
    print("Creating sampling area...")
    sampling_area = create_sampling_area()

    # Generate centroids
    print(f"Generating {NUM_POINTS} random points...")
    centroids = generate_random_points(sampling_area, NUM_POINTS)
    points_path = os.path.join(OUTPUT_DIR, POINTS_LAYER_FILE)
    save_napari_points_layer(centroids, points_path)

    # Load cell data
    print("Loading and filtering cell data...")
    cells_df = pd.read_csv(CELLS_FILE)
    sample_cells = cells_df[cells_df['Sample'] == SAMPLE_NAME]
    if len(sample_cells) == 0:
        raise ValueError(f"No cells found for sample {SAMPLE_NAME}")
    
    cell_coords = sample_cells[['Y_centroid', 'X_centroid']].values.astype(np.float32)  # Note: Axis handling for .tif files
    phenotypes, _ = pd.factorize(sample_cells['phenotype_key'])
    phenotypes = phenotypes.astype(np.int32)
    max_phenotype = phenotypes.max()

    # KDTree pre-filtering
    print("Building KDTree and pre-filtering cells...")
    cell_tree = KDTree(cell_coords)
    max_radius = MAX_STEPS * STEP_SIZE
    all_indices = [cell_tree.query_ball_point((cx, cy), max_radius) for cx, cy in centroids]
    all_indices = [np.array(indices, dtype=np.int32) for indices in all_indices]

    # Shared memory setup
    print("Setting up shared memory...")
    shm_coords = shared_memory.SharedMemory(create=True, size=cell_coords.nbytes)
    shm_pheno = shared_memory.SharedMemory(create=True, size=phenotypes.nbytes)
    
    coords_shared = np.ndarray(cell_coords.shape, dtype=cell_coords.dtype, buffer=shm_coords.buf)
    pheno_shared = np.ndarray(phenotypes.shape, dtype=phenotypes.dtype, buffer=shm_pheno.buf)
    coords_shared[:] = cell_coords
    pheno_shared[:] = phenotypes

    # Prepare arguments
    print("Preparing parallel arguments...")
    args = [
        (cx, cy, indices,
         shm_coords.name, cell_coords.shape, cell_coords.dtype,
         shm_pheno.name, phenotypes.shape, phenotypes.dtype,
         STEP_SIZE, MAX_STEPS, max_phenotype, METRIC)
        for (cx, cy), indices in zip(centroids, all_indices)
    ]

    # Parallel processing
    print(f"Processing {len(centroids)} centroids with {MAX_STEPS} steps each...")
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = executor.map(process_wrapper, args, chunksize=100)
        for (cx, cy), values in zip(centroids, futures):
            result_entry = {
                'sample': SAMPLE_NAME,
                'center_x': cx,
                'center_y': cy
            }
            if METRIC == "Shannon":
                result_entry.update({f'step_{i+1}': val for i, val in enumerate(values)})
            elif METRIC == "Ripley_K":
                result_entry.update({f'K_radius_{i+1}': val for i, val in enumerate(values)})
            results.append(result_entry)

    # Cleanup and output
    print("Cleaning up...")
    shm_coords.close()
    shm_pheno.close()
    shm_coords.unlink()
    shm_pheno.unlink()
    
    print("Saving results...")
    output_path = os.path.join(OUTPUT_DIR, f"{METRIC.lower()}_index_{SAMPLE_NAME}.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    
    print(f"\nTotal execution time: {time.perf_counter() - start_total:.2f} seconds")
    print(f"Results saved in: {output_path}")

if __name__ == "__main__":
    main()