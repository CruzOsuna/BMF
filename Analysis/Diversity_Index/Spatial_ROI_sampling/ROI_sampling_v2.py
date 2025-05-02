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
from itertools import combinations

# Author: Cruz Osuna (cruzosuna2003@gmail.com)

# Python script for calculating spatial metrics (Shannon Index, Ripley K Function, Phenotypic Proportions, and Spatial Co-occurrence) 
# in multiplex imaging datasets, optimized for large-scale biological analyses.


# ========================================================================
# CONFIGURATION
# ========================================================================
config.THREADING_LAYER = 'omp'

# Configuration parameters
USE_LINE_BUFFER = False
NUM_POINTS = 100000
SIDE_DISTANCE = 100
STEP_SIZE = 10
MAX_STEPS = 100

# Metric selection
METRICS = {
    1: "Shannon",
    2: "Ripley_K",
    3: "Phenotypic_Proportion",
    4: "Spatial_Cooccurrence"
}
selected = int(input("Choose metric to calculate:\n"
                     "1) Shannon Index\n"
                     "2) Ripley K Function\n"
                     "3) Phenotypic Proportion\n"
                     "4) Spatial Co-occurrence\n"
                     "Option: "))
METRIC = METRICS[selected]
print(f"\nCalculating metric: {METRIC}\n")

# Input files
LINE_FILE = '/media/HDD_1/ROI_Sampling-benchmark/input/line_try-3.txt'
POLYGON_FILE = '/media/HDD_1/BMF/Spatial_sampling/Shapes/FAHNSCC_14/FAHNSCC_14_partial_S1.txt'
CELLS_FILE = '/media/HDD_1/BMF/Spatial_sampling/Cell_data/FAHNSCC_14_phenotype_annotated.csv'
SAMPLE_NAME = 'FAHNSCC_14'

# Output files
OUTPUT_DIR = '/media/HDD_1/BMF/Spatial_sampling/Output/FAHNSCC_14'
os.makedirs(OUTPUT_DIR, exist_ok=True)
POINTS_LAYER_FILE = f'sampling_points_{SAMPLE_NAME}.csv'

# ========================================================================
# GEOMETRY FUNCTIONS
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
# CORE PROCESSING FUNCTION
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
        lambda_val = total_cells / (np.max(area) if total_cells > 0 else 1.0)
        
        for step in range(max_steps):
            current_radius = step_size * (step + 1)
            current_idx = np.searchsorted(sorted_sq_dist, current_radius**2, side='right')
            results[step] = current_idx / (lambda_val * area[step]) if lambda_val > 0 else 0
            
        return results
    
    elif metric == "Phenotypic_Proportion":
        subset_pheno = phenotypes[indices]
        sorted_pheno = subset_pheno[sorted_idx]
        results = np.zeros((max_steps, max_phenotype + 1))
        last_idx = 0
        
        for step in range(max_steps):
            current_radius = step_size * (step + 1)
            sq_radius = current_radius ** 2
            current_idx = np.searchsorted(sorted_sq_dist, sq_radius, side='right')
            window_pheno = sorted_pheno[last_idx:current_idx]
            
            counts = np.bincount(window_pheno, minlength=max_phenotype+1)
            total = len(window_pheno)
            if total > 0:
                results[step] = counts / total
            last_idx = current_idx
            
        return results
    
    elif metric == "Spatial_Cooccurrence":
        subset_pheno = phenotypes[indices]
        sorted_pheno = subset_pheno[sorted_idx]
        n_pheno = max_phenotype + 1
        results = np.zeros((max_steps, n_pheno, n_pheno), dtype=np.float32)
        last_idx = 0
        
        for step in range(max_steps):
            current_radius = step_size * (step + 1)
            sq_radius = current_radius ** 2
            current_idx = np.searchsorted(sorted_sq_dist, sq_radius, side='right')
            window_pheno = sorted_pheno[last_idx:current_idx]
            
            cooc_matrix = np.zeros((n_pheno, n_pheno), dtype=np.int32)
            for i in range(len(window_pheno)):
                for j in range(i+1, len(window_pheno)):
                    p1 = window_pheno[i]
                    p2 = window_pheno[j]
                    cooc_matrix[p1, p2] += 1
                    cooc_matrix[p2, p1] += 1
            
            total_pairs = len(window_pheno) * (len(window_pheno)-1) / 2
            if total_pairs > 0:
                results[step] = cooc_matrix / total_pairs
                
            last_idx = current_idx
            
        return results
    
    return np.empty(0)

# ========================================================================
# RESULT PROCESSING
# ========================================================================

def flatten_cooccurrence(results, max_steps, max_phenotype):
    """Convert 3D co-occurrence matrix to flat dictionary entries"""
    pheno_pairs = list(combinations(range(max_phenotype + 1), 2))
    output = {}
    
    for step in range(max_steps):
        for p1, p2 in pheno_pairs:
            key = f"step_{step+1}_pheno_{p1}_{p2}"
            output[key] = results[step, p1, p2]
            
    return output

def save_results(results, output_path):
    """Handle different metric output formats"""
    df = pd.DataFrame(results)
    
    if METRIC == "Spatial_Cooccurrence":
        df = df.melt(id_vars=['sample', 'center_x', 'center_y'],
                    var_name='measurement',
                    value_name='value')
        df[['step', 'pheno1', 'pheno2']] = df['measurement'].str.extract(
            r'step_(\d+)_pheno_(\d+)_(\d+)')
        df = df.drop(columns='measurement')
    
    df.to_csv(output_path, index=False)

# ========================================================================
# PARALLEL PROCESSING
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
    pd.DataFrame(centroids, columns=['center_x', 'center_y']).to_csv(points_path, index=False)

    # Load cell data
    print("Loading and filtering cell data...")
    cells_df = pd.read_csv(CELLS_FILE)
    sample_cells = cells_df[cells_df['Sample'] == SAMPLE_NAME]
    if len(sample_cells) == 0:
        raise ValueError(f"No cells found for sample {SAMPLE_NAME}")
    
    cell_coords = sample_cells[['Y_centroid', 'X_centroid']].values.astype(np.float32)
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
            elif METRIC == "Phenotypic_Proportion":
                for pheno in range(max_phenotype + 1):
                    result_entry.update({f'step_{i+1}_pheno_{pheno}': val for i, val in enumerate(values[:, pheno])})
            elif METRIC == "Spatial_Cooccurrence":
                cooc_dict = flatten_cooccurrence(values, MAX_STEPS, max_phenotype)
                result_entry.update(cooc_dict)
            
            results.append(result_entry)

    # Cleanup and output
    print("Cleaning up...")
    shm_coords.close()
    shm_pheno.close()
    shm_coords.unlink()
    shm_pheno.unlink()
    
    print("Saving results...")
    output_path = os.path.join(OUTPUT_DIR, f"{METRIC.lower()}_index_{SAMPLE_NAME}.csv")
    save_results(results, output_path)
    
    print(f"\nTotal execution time: {time.perf_counter() - start_total:.2f} seconds")
    print(f"Results saved in: {output_path}")

if __name__ == "__main__":
    main()