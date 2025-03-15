import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from typing import Dict, Tuple, Optional

# --------------------------
# Core Metrics Calculation
# --------------------------
def shannon_index(counts: np.ndarray, base: float = np.e) -> float:
    """
    Calculates the Shannon diversity index for given species counts.
    
    Parameters:
        counts (np.ndarray): Array of species counts
        base (float): Logarithm base (default: natural logarithm)
    
    Returns:
        float: Shannon diversity index value
    """
    counts = np.asarray(counts)
    total = counts.sum()
    if total <= 0:
        return 0.0
    proportions = counts[counts > 0] / total
    return -np.sum(proportions * np.log(proportions)) / np.log(base) if proportions.size > 0 else 0.0

# --------------------------
# Spatial Analysis
# --------------------------
def calculate_spatial_diversity(df: pd.DataFrame, 
                               community_col: str, 
                               species_col: str, 
                               sample: str, 
                               grid_size: int = 100) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Calculate spatial diversity for a given sample using grid-based approach.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with spatial and species data
        community_col (str): Column name containing community/sample identifiers
        species_col (str): Column name containing species identifiers
        sample (str): Specific sample to analyze
        grid_size (int): Size of grid cells for spatial binning
    
    Returns:
        Tuple: (Diversity DataFrame, x bins, y bins)
    """
    # Input validation
    required_cols = ['X_centroid', 'Y_centroid', community_col, species_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame missing required columns: {required_cols}")
    
    if grid_size <= 0:
        raise ValueError("Grid size must be a positive integer")

    # Sample filtering and coordinate handling
    df_sample = df[df[community_col] == sample].copy()
    if df_sample.empty:
        return pd.DataFrame(), np.array([]), np.array([])

    # Optimize memory usage
    df_sample = df_sample.astype({
        'X_centroid': 'float32',
        'Y_centroid': 'float32',
        community_col: 'category',
        species_col: 'category'
    })

    # Bin calculation with edge case handling
    x_min, x_max = df_sample['X_centroid'].min(), df_sample['X_centroid'].max()
    y_min, y_max = df_sample['Y_centroid'].min(), df_sample['Y_centroid'].max()
    
    # Handle single-point samples
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range == 0:
        x_bins = np.linspace(x_min - grid_size/2, x_max + grid_size/2, 3)
    else:
        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    
    if y_range == 0:
        y_bins = np.linspace(y_min - grid_size/2, y_max + grid_size/2, 3)
    else:
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

    # Spatial binning
    df_sample['X_bin'] = np.digitize(df_sample['X_centroid'], x_bins)
    df_sample['Y_bin'] = np.digitize(df_sample['Y_centroid'], y_bins)

    # Diversity calculation
    diversity_results = (
        df_sample
        .groupby(['X_bin', 'Y_bin'], observed=True)
        [species_col]
        .agg(lambda x: shannon_index(x.value_counts().values))
        .reset_index()
        .rename(columns={species_col: 'Shannon_Index'})
    )
    return diversity_results, x_bins, y_bins

# --------------------------
# Comparative Analysis (Fixed)
# --------------------------
def calc_avg_for_one_sample(sample_data: pd.DataFrame, 
                           species_col: str, 
                           sample: str, 
                           grid_size: int) -> Dict:
    """
    Calculate average Shannon index for a single sample-grid combination.
    
    Parameters:
        sample_data (pd.DataFrame): Pre-filtered sample data
        species_col (str): Column name for species
        sample (str): Target sample name
        grid_size (int): Grid size to test
    
    Returns:
        dict: Results dictionary
    """
    diversity_results, _, _ = calculate_spatial_diversity(
        sample_data,  # Use pre-filtered data
        community_col="Sample",  # Hardcoded as already filtered
        species_col=species_col,
        sample=sample,
        grid_size=grid_size
    )
    
    if diversity_results.empty:
        return {
            'sample': sample,
            'grid_size': grid_size,
            'avg_shannon_index': 0.0,
            'variance_shannon': 0.0
        }
    
    values = diversity_results['Shannon_Index']
    return {
        'sample': sample,
        'grid_size': grid_size,
        'avg_shannon_index': values.mean(skipna=True),
        'variance_shannon': values.var(skipna=True)
    }

def compare_grid_sizes_across_samples(df: pd.DataFrame, 
                                     community_col: str, 
                                     species_col: str, 
                                     grid_sizes: list, 
                                     n_jobs: int = -1, 
                                     output_csv: str = "comparative_avg_shannon_values.csv", 
                                     output_plot: str = "comparative_avg_shannon_plot.png") -> None:
    """
    Compare diversity metrics across different grid sizes using parallel processing.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        community_col (str): Column name for communities
        species_col (str): Column name for species
        grid_sizes (list): List of grid sizes to test
        n_jobs (int): Number of parallel jobs
        output_csv (str): Output CSV path
        output_plot (str): Output plot path
    """
    # Data preprocessing
    df = df.copy()
    df[community_col] = df[community_col].fillna("Unknown").astype('category')
    df[species_col] = df[species_col].fillna("Unknown").astype('category')
    
    # Pre-group samples for parallel processing with observed=True
    sample_groups = {sample: group for sample, group in df.groupby(community_col, observed=True)}
    
    # Create parallel tasks with correct arguments
    tasks = [
        (sample_group, species_col, sample_name, gsize)
        for sample_name, sample_group in sample_groups.items()
        for gsize in grid_sizes
    ]
    
    # Parallel execution
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(calc_avg_for_one_sample)(*task)
        for task in tasks
    )
    
    # Process and save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Resultados guardados en: {output_csv}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    for sample in df_results['sample'].unique():
        sample_data = df_results[df_results['sample'] == sample]
        plt.errorbar(
            x=sample_data['grid_size'],
            y=sample_data['avg_shannon_index'],
            yerr=np.sqrt(sample_data['variance_shannon']),
            marker='o',
            linestyle='-',
            markersize=8,
            linewidth=1.5,
            label=sample
        )
    
    plt.xlabel('Tamaño de grilla (grid_size)', fontsize=12)
    plt.ylabel('Promedio índice de Shannon ± SD', fontsize=12)
    plt.title('Comparativa del índice de Shannon con varianza', fontsize=14, pad=15)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xticks(grid_sizes, labels=grid_sizes, rotation=45)
    plt.legend(title='ROI', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_plot, dpi=600, bbox_inches='tight')
    print(f"Gráfico comparativo guardado en: {output_plot}")
    plt.close()

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "csv_file": "/media/HDD_1/BMF/FA/8_Results/Datasets/2_2_Phenotype_calling/aggregated_data.csv",
        "community_col": "Sample",
        "species_col": "phenotype_key",
        "grid_sizes": [10, 20, 30, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 
                      220, 240, 260, 280, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000],
        "output_csv": "comparative_avg_shannon_values.csv",
        "output_plot": "comparative_avg_shannon_plot.png",
        "n_jobs": -1
    }
    
    try:
        # Load data with memory optimization
        df_data = pd.read_csv(CONFIG["csv_file"])
        for col in ['X_centroid', 'Y_centroid']:
            if col in df_data.columns:
                df_data[col] = pd.to_numeric(df_data[col], downcast='float')
        
        # Execute analysis
        compare_grid_sizes_across_samples(
            df=df_data,
            community_col=CONFIG["community_col"],
            species_col=CONFIG["species_col"],
            grid_sizes=CONFIG["grid_sizes"],
            n_jobs=CONFIG["n_jobs"],
            output_csv=CONFIG["output_csv"],
            output_plot=CONFIG["output_plot"]
        )
        
    except Exception as e:
        print(f"Error en la ejecución: {str(e)}")
        raise