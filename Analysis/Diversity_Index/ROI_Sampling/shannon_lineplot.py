import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import time

# Configuration
INPUT_CSV = 'shannon_results.csv'
OUTPUT_IMAGE = 'shannon_lineplot.png'
FONT_SCALE = 1.5
FIG_SIZE = (12, 6)
DPI = 600

def plot_shannon_trend_with_error(csv_path):
    """Generate a line plot showing the trend of the Shannon Index with standard deviation"""
    start_time = time.time()  # Start timer
    
    # Seaborn style
    sns.set(style="whitegrid", font_scale=FONT_SCALE)

    # Load data
    df = pd.read_csv(csv_path)

    # Extract and sort step columns
    step_columns = sorted(
        [col for col in df.columns if col.startswith('step_')],
        key=lambda x: int(x.split('_')[1])
    )

    # Calculate mean and standard deviation of Shannon Index for each step
    mean_values = df[step_columns].mean(axis=0)
    std_values = df[step_columns].std(axis=0)
    step_numbers = [int(col.split('_')[1]) for col in step_columns]

    # Create plot
    plt.figure(figsize=FIG_SIZE)
    ax = sns.lineplot(x=step_numbers, y=mean_values.values, marker='o', linewidth=2, label='Shannon Index (mean)')

    # Add error band (standard deviation)
    ax.fill_between(
        step_numbers,
        mean_values - std_values,
        mean_values + std_values,
        color='b', alpha=0.2, label='Standard deviation'
    )

    # Labels and title
    ax.set_xlabel('Step number', labelpad=10)
    ax.set_ylabel('Shannon Index (mean)', labelpad=10)
    plt.title('Shannon Index Trend Across Steps', pad=20)

    # Scientific notation for Y-axis
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Legend
    ax.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=DPI)
    plt.close()
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"Plot saved to: {OUTPUT_IMAGE}")
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    plot_shannon_trend_with_error(INPUT_CSV)