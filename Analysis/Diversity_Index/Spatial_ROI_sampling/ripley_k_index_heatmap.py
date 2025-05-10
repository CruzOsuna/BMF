import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.path import Path
from scipy.stats import gaussian_kde
import ast
import re

# === CONFIGURATION ===
INPUT_FILE = '/media/HDD_1/BMF/Spatial_sampling/Output/FAHNSCC_14/ripley_k_index_FAHNSCC_14.csv'
RADIUS_INDEX = 50
TUMOR_MASK_FILE = '/media/HDD_1/BMF/Spatial_sampling/Shapes/FAHNSCC_14/FAHNSCC_14_partial_S1.txt'
OUTPUT_DIR = '/media/HDD_1/BMF/Spatial_sampling/Output/FAHNSCC_14/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD RIPLEY DATA ===
df = pd.read_csv(INPUT_FILE)
radius_col = f'K_radius_{RADIUS_INDEX}'
if radius_col not in df.columns:
    raise ValueError(f"Column '{radius_col}' not found.")

# === LOAD TUMOR MASK ===
with open(TUMOR_MASK_FILE, 'r') as f:
    content = f.read()

# Extract all floats from the file
floats = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", content)))
print(f"Number of floats extracted: {len(floats)}")

# Optional: Handle odd-length list
if len(floats) % 2 != 0:
    print("Warning: Odd number of coordinates found. Ignoring the last one.")
    floats = floats[:-1]

tumor_coords = np.array(floats).reshape(-1, 2)



# Extract all floats from the file
floats = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", content)))
tumor_coords = np.array(floats).reshape(-1, 2)

# === CREATE GRID FOR KDE ===
xmin, xmax = df['center_x'].min(), df['center_x'].max()
ymin, ymax = df['center_y'].min(), df['center_y'].max()
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500),
                     np.linspace(ymin, ymax, 500))

# === KERNEL DENSITY ESTIMATION with weights ===
values = np.vstack([df['center_x'], df['center_y']])
weights = df[radius_col]
kde = gaussian_kde(values, weights=weights)
zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

# === MASK OUTSIDE TUMOR REGION ===
tumor_path = Path(tumor_coords)
grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
mask = tumor_path.contains_points(grid_points).reshape(xx.shape)
zz_masked = np.where(mask, zz, np.nan)

# === PLOT HEATMAP ===
plt.figure(figsize=(10, 8))
hm = plt.imshow(zz_masked, origin='lower', cmap='viridis',
                extent=[xmin, xmax, ymin, ymax], aspect='auto')
plt.colorbar(hm, label=f"Ripley's K at radius {RADIUS_INDEX}")
plt.plot(tumor_coords[:, 0], tumor_coords[:, 1], color='red', lw=1.5, label='Tumor Border')
plt.legend()
plt.xlabel('X Centroid')
plt.ylabel('Y Centroid')
plt.title(f"Smooth Ripley’s K Heatmap (Radius {RADIUS_INDEX})")
plt.gca().invert_yaxis()
plt.tight_layout()

# === SAVE ===
output_path = os.path.join(OUTPUT_DIR, f'smooth_ripley_k_heatmap_radius_{RADIUS_INDEX}.png')
plt.savefig(output_path, dpi=300)
plt.show()
print(f"Saved to: {output_path}")
