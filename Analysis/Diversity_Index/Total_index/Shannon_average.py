import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

def shannon_index(counts):
    total = sum(counts)
    if total == 0:
        return 0
    proportions = [n / total for n in counts if n > 0]
    if not proportions:
        return 0
    H = -sum(p * np.log(p) for p in proportions)
    return H

def calculate_spatial_diversity(df, community_col, species_col, sample, grid_size=100):
    df_sample = df[df[community_col] == sample].copy()
    x_min, x_max = df_sample['X_centroid'].min(), df_sample['X_centroid'].max()
    y_min, y_max = df_sample['Y_centroid'].min(), df_sample['Y_centroid'].max()
    x_bins = np.arange(x_min, x_max, grid_size)
    y_bins = np.arange(y_min, y_max, grid_size)
    df_sample['X_bin'] = np.digitize(df_sample['X_centroid'], x_bins)
    df_sample['Y_bin'] = np.digitize(df_sample['Y_centroid'], y_bins)
    diversity_results = (
        df_sample
        .groupby(['X_bin', 'Y_bin'])[species_col]
        .apply(lambda x: shannon_index(x.value_counts()))
        .reset_index()
    )
    diversity_results.columns = ['X_bin', 'Y_bin', 'Shannon_Index']
    return diversity_results, x_bins, y_bins

def plot_spatial_diversity(diversity_df, x_bins, y_bins, sample, output_plot="diversidad_espacial.png"):
    heatmap_data = np.full((len(y_bins), len(x_bins)), np.nan)
    for _, row in diversity_df.iterrows():
        if (1 <= row['X_bin'] <= len(x_bins)) and (1 <= row['Y_bin'] <= len(y_bins)):
            heatmap_data[int(row['Y_bin']) - 1, int(row['X_bin']) - 1] = row['Shannon_Index']
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='viridis', cbar=True, xticklabels=False, yticklabels=False)
    plt.xlabel("Posición X (bins)")
    plt.ylabel("Posición Y (bins)")
    plt.title(f"Diversidad Alfa Espacial en {sample}")
    plt.gca().invert_yaxis()
    plt.savefig(output_plot, dpi=600, bbox_inches='tight')
    print(f"Gráfico guardado en {output_plot}")
    plt.show()

def calc_avg_for_one_sample(df, community_col, species_col, sample, grid_size):
    diversity_results, _, _ = calculate_spatial_diversity(df, community_col, species_col, sample, grid_size)
    avg_val = diversity_results['Shannon_Index'].mean()
    var_val = diversity_results['Shannon_Index'].var()
    avg_val = avg_val if pd.notnull(avg_val) else 0
    var_val = var_val if pd.notnull(var_val) else 0
    return {'sample': sample, 'grid_size': grid_size, 'avg_shannon_index': avg_val, 'variance_shannon': var_val}

def compare_grid_sizes_across_samples(df, community_col, species_col, grid_sizes, n_jobs=-1, output_csv="comparative_avg_shannon_values.csv", output_plot="comparative_avg_shannon_plot.png"):
    df[community_col] = df[community_col].fillna("Unknown")
    df[species_col] = df[species_col].fillna("Unknown")
    all_samples = df[community_col].unique()
    tasks = [(sample, gsize) for sample in all_samples for gsize in grid_sizes]
    results = Parallel(n_jobs=n_jobs)(delayed(calc_avg_for_one_sample)(df, community_col, species_col, sample, gsize) for (sample, gsize) in tasks)
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Resultados guardados en: {output_csv}")
    plt.figure(figsize=(10, 7))
    for sample in df_results['sample'].unique():
        sample_data = df_results[df_results['sample'] == sample]
        plt.errorbar(
            x=sample_data['grid_size'],
            y=sample_data['avg_shannon_index'],
            yerr=np.sqrt(sample_data['variance_shannon']),
            marker='o',
            linestyle='-',
            label=sample
        )
    plt.xlabel('Tamaño de grilla (grid_size)')
    plt.ylabel('Promedio índice de Shannon')
    plt.title('Comparativa del índice de Shannon con varianza')
    plt.grid(True)
    plt.legend(title='ROI', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_plot, dpi=600)
    print(f"Gráfico comparativo guardado en: {output_plot}")
    plt.show()

if __name__ == "__main__":
    csv_file = "/home/cruz-osuna/Desktop/BMF/Analysis/Diversity_Index/Shannon_inputs/aggregated_data_2-2.csv"
    community_col = "Sample"
    species_col = "phenotype_key"
    grid_sizes_to_test = [10, 20, 30, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
    output_csv_name = "comparative_avg_shannon_values.csv"
    output_plot_name = "comparative_avg_shannon_plot.png"
    df_data = pd.read_csv(csv_file)
    compare_grid_sizes_across_samples(df=df_data, community_col=community_col, species_col=species_col, grid_sizes=grid_sizes_to_test, n_jobs=-1, output_csv=output_csv_name, output_plot=output_plot_name)