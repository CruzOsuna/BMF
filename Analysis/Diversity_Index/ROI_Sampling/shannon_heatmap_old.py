import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time  # Added import

# Configuration
INPUT_CSV = 'shannon_results.csv'
OUTPUT_IMAGE = 'shannon_heatmap.png'
COLOR_MAP = 'viridis'
FIG_SIZE = (32, 24)
DPI = 600
X_LABEL_INTERVAL = 5
Y_LABEL_INTERVAL = 50
FONT_SCALE = 0.8
THRESHOLD = 1  # Minimum value considered as "high" for the Shannon index

def calculate_sorting_metric(row):
    """Calculates sorting metric: first step that exceeds the threshold"""
    steps = row.filter(like='step_').values
    above_threshold = np.where(steps >= THRESHOLD)[0]
    if len(above_threshold) > 0:
        return above_threshold[0]  # First step that exceeds the threshold
    return len(steps)  # If it never exceeds, place at the end

def create_shannon_heatmap(csv_path):
    # Set style
    sns.set(font_scale=FONT_SCALE)
    
    # Load and process data
    df = pd.read_csv(csv_path)
    
    # Sort points by speed to reach high values
    df['sort_key'] = df.apply(calculate_sorting_metric, axis=1)
    df = df.sort_values(by='sort_key', ascending=True).drop('sort_key', axis=1)
    
    # Extract and sort step columns
    step_columns = sorted(
        [col for col in df.columns if col.startswith('step_')],
        key=lambda x: int(x.split('_')[1])
    )
    
    # Generate labels
    step_numbers = [int(col.split('_')[1]) for col in step_columns]
    x_labels = [
        str(num) if (num-1) % X_LABEL_INTERVAL == 0 else "" 
        for num in step_numbers]
    
    num_samples = df.shape[0]
    y_labels = [
        f"Point {i+1}" if i % Y_LABEL_INTERVAL == 0 else "" 
        for i in range(num_samples)]

    # Create heatmap
    plt.figure(figsize=FIG_SIZE)
    ax = sns.heatmap(
        df[step_columns].values,
        cmap=COLOR_MAP,
        cbar_kws={'label': 'Shannon Index'},
        yticklabels=y_labels,
        xticklabels=x_labels)
    
    # Customize axes
    ax.set_xlabel('Expansion Step', labelpad=15)
    ax.set_ylabel('Sampling Points (Sorted)', labelpad=15)
    plt.title(f'Shannon Index Progression (Sorted by Speed to Reach ≥{THRESHOLD})', pad=25)
    
    # Rotate X-axis labels
    ax.set_xticklabels(
        ax.get_xticklabels(), 
        rotation=45, 
        ha='right',
        fontsize=10)
    
    # Add grid lines
    ax.hlines(
        y=range(0, num_samples, 5), 
        xmin=0, 
        xmax=len(step_columns),
        colors='white',
        linewidths=0.1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    start_time = time.perf_counter()  # Added timer start
    create_shannon_heatmap(INPUT_CSV)
    elapsed = time.perf_counter() - start_time  # Calculate elapsed time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")  # Added time print