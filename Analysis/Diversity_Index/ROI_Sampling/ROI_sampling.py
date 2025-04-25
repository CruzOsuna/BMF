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
USE_LINE_BUFFER = False              # Choose line (True) or polygon (False)
NUM_POINTS = 100000                   # Number of random points to sample
SIDE_DISTANCE = 100                  # Buffer distance for line sampling
STEP_SIZE = 10                        # Distance increment per radius step
MAX_STEPS = 100                      # Total number of radius steps
 
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
        # Load and parse numpy-style array coordinates
        with open(LINE_FILE) as f:
            content = f.read()
        
        # Extract all numerical values using regex
        numbers = list(map(float, re.findall(r'\d+\.\d+', content)))
        
        # Create coordinate pairs
        coords = list(zip(numbers[::2], numbers[1::2]))
        line = LineString(coords)
        return line.buffer(SIDE_DISTANCE, cap_style=2)
    else:
        # Load polygon coordinates (similar fix if needed)
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
    df['name'] = [f'centroid_{i}' for i in range(len(df))]  # Add identifier column
    df.to_csv(filename, index=False)


# ========================================================================
# 1. VECTORIZED RANDOM POINT GENERATOR (OPTIMIZED)
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
        
        # Vectorized containment check
        points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y))
        mask = points_gdf.within(polygon)
        valid = np.column_stack((x[mask], y[mask]))
        
        if valid.size > 0:
            needed = min(remaining, valid.shape[0])
            points[generated:generated+needed] = valid[:needed]
            generated += needed
            
    return points

# ========================================================================
# 2. OPTIMIZED CENTROID PROCESSING FUNCTION
# ========================================================================

@njit
def process_centroid(cx, cy, cell_coords, phenotypes, indices, step_size, max_steps, max_phenotype):
    """Optimized version using pre-filtered indices"""
    # Work only with relevant cells
    subset_coords = cell_coords[indices]
    subset_pheno = phenotypes[indices]
    
    dx = subset_coords[:, 0] - cx
    dy = subset_coords[:, 1] - cy
    squared_distances = dx**2 + dy**2
    sorted_idx = np.argsort(squared_distances)
    sorted_sq_dist = squared_distances[sorted_idx]
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

# ========================================================================
# 3. SHARED MEMORY WRAPPER FOR MULTIPROCESSING
# ========================================================================

def process_wrapper(args):
    """Updated wrapper to handle indices parameter"""
    (cx, cy, indices,
     shm_coords_name, coords_shape, coords_dtype,
     shm_pheno_name, pheno_shape, pheno_dtype,
     step_size, max_steps, max_phenotype) = args
    
    # Access shared memory
    shm_coords = shared_memory.SharedMemory(name=shm_coords_name)
    shm_pheno = shared_memory.SharedMemory(name=shm_pheno_name)
    
    cell_coords = np.ndarray(coords_shape, dtype=coords_dtype, buffer=shm_coords.buf)
    phenotypes = np.ndarray(pheno_shape, dtype=pheno_dtype, buffer=shm_pheno.buf)
    
    result = process_centroid(cx, cy, cell_coords, phenotypes, indices,
                            step_size, max_steps, max_phenotype)
    
    shm_coords.close()
    shm_pheno.close()
    
    return result

# ========================================================================
# MAIN FUNCTION
# ========================================================================

def main():
    """Optimized main function with KDTree pre-filtering"""
    start_total = time.perf_counter()
    
    # Create sampling area geometry
    print("Creating sampling area...")
    start = time.perf_counter()
    sampling_area = create_sampling_area()
    end = time.perf_counter()
    print(f"Time to create sampling area: {end - start:.2f} seconds")

    # Generate and save centroids
    print(f"Generating {NUM_POINTS} random points...")
    start = time.perf_counter()
    centroids = generate_random_points(sampling_area, NUM_POINTS)
    end = time.perf_counter()
    print(f"Time to generate centroids: {end - start:.2f} seconds")

    print("Saving centroids...")
    points_path = os.path.join(OUTPUT_DIR, POINTS_LAYER_FILE)
    save_napari_points_layer(centroids, points_path)

    # Load and filter cell data
    print("Loading and filtering cell data...")
    start = time.perf_counter()
    cells_df = pd.read_csv(CELLS_FILE)
    sample_cells = cells_df[cells_df['Sample'] == SAMPLE_NAME]
    if len(sample_cells) == 0:
        raise ValueError(f"No cells found for sample {SAMPLE_NAME}")
    
    # Convert to optimal data types
    cell_coords = sample_cells[['Y_centroid', 'X_centroid']].values.astype(np.float32) #Aqui esta invertido por como se manejan los ejes en archivos .tif
    phenotypes, _ = pd.factorize(sample_cells['phenotype_key'])
    phenotypes = phenotypes.astype(np.int32)
    max_phenotype = phenotypes.max()
    end = time.perf_counter()
    print(f"Time to load and process cell data: {end - start:.2f} seconds")

    # ========================================================================
    # KDTREE PRE-FILTERING
    # ========================================================================
    print("Building KDTree and pre-filtering cells...")
    start_kd = time.perf_counter()
    
    cell_tree = KDTree(cell_coords)
    max_radius = MAX_STEPS * STEP_SIZE
    all_indices = [cell_tree.query_ball_point((cx, cy), max_radius) 
                   for cx, cy in centroids]
    all_indices = [np.array(indices, dtype=np.int32) for indices in all_indices]
    
    end_kd = time.perf_counter()
    print(f"KDTree filtering completed in {end_kd - start_kd:.2f} seconds")
    print(f"Average cells per centroid: {np.mean([len(i) for i in all_indices]):.1f}")

    # ========================================================================
    # SHARED MEMORY SETUP
    # ========================================================================
    print("Setting up shared memory...")
    start = time.perf_counter()
    
    shm_coords = shared_memory.SharedMemory(create=True, size=cell_coords.nbytes)
    shm_pheno = shared_memory.SharedMemory(create=True, size=phenotypes.nbytes)
    
    coords_shared = np.ndarray(cell_coords.shape, dtype=cell_coords.dtype, 
                              buffer=shm_coords.buf)
    pheno_shared = np.ndarray(phenotypes.shape, dtype=phenotypes.dtype,
                             buffer=shm_pheno.buf)
    coords_shared[:] = cell_coords
    pheno_shared[:] = phenotypes

    end = time.perf_counter()
    print(f"Time to setup shared memory: {end - start:.2f} seconds")

    # ========================================================================
    # PARALLEL ARGUMENT PREPARATION
    # ========================================================================
    print("Preparing parallel arguments...")
    args = [
        (cx, cy, indices,
         shm_coords.name, cell_coords.shape, cell_coords.dtype,
         shm_pheno.name, phenotypes.shape, phenotypes.dtype,
         STEP_SIZE, MAX_STEPS, max_phenotype)
        for (cx, cy), indices in zip(centroids, all_indices)
    ]

    # ========================================================================
    # PARALLEL PROCESSING
    # ========================================================================

    print(f"Processing {len(centroids)} centroids with {MAX_STEPS} steps each...")
    start_processing = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = executor.map(process_wrapper, args, chunksize=100)
        for (cx, cy), shannon_values in zip(centroids, futures):
            results.append({
                'sample': SAMPLE_NAME,
                'center_x': cx,
                'center_y': cy,
                **{f'step_{i+1}': val for i, val in enumerate(shannon_values)}
            })
    end_processing = time.perf_counter()
    print(f"Parallel processing time: {end_processing - start_processing:.2f} seconds")


    # ========================================================================
    # CLEANUP AND OUTPUT
    # ========================================================================
    print("Cleaning up...")
    shm_coords.close()
    shm_pheno.close()
    shm_coords.unlink()
    shm_pheno.unlink()
    
    print("Saving results...")
    output_path = os.path.join(OUTPUT_DIR, f"shannon_index_{SAMPLE_NAME}.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    
    print(f"\nTotal execution time: {time.perf_counter() - start_total:.2f} seconds")
    print(f"Results saved in: {output_path}")

if __name__ == "__main__":
    main()
