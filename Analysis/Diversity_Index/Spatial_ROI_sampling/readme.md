# Spatial Analysis Toolkit for Multiplex Imaging Data

A high-performance Python script for calculating spatial metrics (Shannon Index, Ripley K Function, Phenotypic Proportions, and Spatial Co-occurrence) in multiplex imaging datasets, optimized for large-scale biological analyses.

Spatial Cell Analysis Toolkit
This computational tool quantifies spatial heterogeneity and cellular diversity in tissue samples using spatially resolved single-cell data. It systematically analyzes biological patterns through an adaptive radial sampling approach:

Flexible Sampling Design

Generates statistically robust random sampling points across user-specified tissue regions (polygons/line buffers)

Implements concentric circular sampling areas at each point, with:
• User-configurable radial increments (step size: e.g., 10µm)
• Customizable analysis range (max steps: e.g., 0-100µm in 10 steps)

Multi-Metric Quantification
Computes five complementary spatial statistics at each radial expansion:

Diversity indices: Shannon (entropy), Simpson (dominance)

Spatial patterns: Ripley K (clustering), Co-occurrence (cell-type interactions)

Composition analysis: Phenotypic Proportions (cell-type frequencies)

High-Throughput Architecture

Optimized spatial indexing (KDTree) for rapid neighborhood queries

Parallel processing with shared memory management for large datasets

Numba-accelerated metric calculations for subcellular resolution

Standardized Outputs

CSV files with radial metrics linked to sampling coordinates

Structured tables enabling:
• Cross-sample comparisons
• Spatial trend visualization (e.g., diversity vs. distance gradients)
• Downstream statistical modeling

Applications: Characterizes tumor microenvironments, immune cell organization, and developmental patterning through quantitative spatial metrics.

## Key Features
- **Four spatial metrics** supported:
  - Shannon Diversity Index
  - Ripley's K Function
  - Phenotypic Proportions
  - Spatial Co-occurrence Matrix
- **Parallel processing** with shared memory optimization
- **Flexible sampling** via polygons or line buffers
- **CSV output** compatible with downstream analysis tools
- **Napari-compatible** sampling point export

## Prerequisites
- Python 3.8+
- Required packages:
  ```bash
  numpy pandas shapely geopandas numba scipy tqdm
```


## Installation

  ```bash

git clone https://github.com/yourusername/spatial-analysis-toolkit.git

cd spatial-analysis-toolkit
  
pip install -r requirements.txt
```


## Configuration
Edit the following parameters in the script header:

```bash
# Sampling parameters
USE_LINE_BUFFER = False  # True for line buffer sampling
NUM_POINTS = 100000      # Number of sampling points
SIDE_DISTANCE = 100      # Buffer distance (microns)
STEP_SIZE = 10           # Radial step size (microns)
MAX_STEPS = 100          # Maximum analysis radius

# File paths
LINE_FILE = '/path/to/line_coordinates.txt'
POLYGON_FILE = '/path/to/polygon_coordinates.txt'
CELLS_FILE = '/path/to/cell_data.csv'
SAMPLE_NAME = 'your_sample_name'
OUTPUT_DIR = '/path/to/output_directory'
```


## Usage


1) Run the script:

```bash
python ROI_sampling_integrated.py
```

2) Select metric at prompt:


Choose metric to calculate:
1) Shannon Index
2) Ripley K Function
3) Phenotypic Proportion
4) Spatial Co-occurrence
Option: 



## Output Files
- Sampling points: sampling_points_{sample}.csv

- Columns: center_x, center_y

- Metric results: {metric}_index_{sample}.csv

- Format varies by metric:

- Shannon: Step-wise diversity values

- Ripley K: Radius-specific K values

- Proportions: Phenotype fractions per radius

- Co-occurrence: Melted DataFrame of phenotype pairs



Citation
If using this tool in research, please cite:

bibtex
[Your publication citation here]


Maintainer: Your Name | your.email@institution.edu
Last Updated: {Month} {Year}



Notes:
- The phenotipyc proportion functionality need a loot of memory 
-



