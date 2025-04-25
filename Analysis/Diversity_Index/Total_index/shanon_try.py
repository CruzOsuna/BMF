import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv("comparative_avg_shannon_values.csv")

# Calculate standard deviation from variance
df["std_shannon"] = np.sqrt(df["variance_shannon"])

# Create plot
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

# Get unique samples and color palette
samples = df["sample"].unique()
palette = sns.color_palette("tab20", n_colors=len(samples))

# Plot each sample with error bars
for sample, color in zip(samples, palette):
    sample_df = df[df["sample"] == sample].sort_values("grid_size")
    
    # Plot main line
    plt.plot(sample_df["grid_size"], 
             sample_df["avg_shannon_index"], 
             color=color,
             label=sample,
             alpha=0.8)
    
    # Add error bars
    plt.errorbar(sample_df["grid_size"],
                 sample_df["avg_shannon_index"],
                 yerr=sample_df["std_shannon"],
                 fmt='none',
                 color=color,
                 alpha=0.3,
                 capsize=3)

# Set logarithmic scales
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Grid Size (log scale)")
plt.ylabel("Average Shannon Index ± SD (log scale)")
plt.title("Shannon Index vs. Grid Size Across Samples")

# Move legend outside
plt.legend(bbox_to_anchor=(1.05, 1), 
           loc="upper left",
           frameon=False)

plt.tight_layout()
plt.savefig("shannon_index_plot.pdf", bbox_inches="tight")
plt.show()