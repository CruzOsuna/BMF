## Napari Viewer - Installation and Usage

### Overview
This script provides a set of interactive widgets for visualizing, analyzing, and processing multichannel image data using the Napari framework. It includes functions to open images and masks, manage shapes, adjust contrast, and export results.

#### Authors: Pablo Siliceo Portugal (psiliceop@gmail.com) & Cruz Francisco Osuna Aguirre (cruzosuna2003@gmail.com)

---

## Dependencies
The following libraries are required to run the script:

- napari
- pandas
- numpy
- random
- tifffile
- scimap
- dask
- zarr
- PyQt5
- magicgui

Install them using pip:
```bash
pip install napari pandas numpy tifffile scimap dask zarr PyQt5 magicgui
```

Or include them in your environment file if using Conda:
```yaml
name: napari-env
channels:
  - conda-forge
  - bioconda
dependencies:
  - python=3.9
  - napari
  - pandas
  - numpy
  - random
  - tifffile
  - scimap
  - dask
  - zarr
  - pyqt
  - magicgui
```

---

## Installation
Make sure you have Conda installed. To create the environment, run:
```bash
conda env create -f napari-env.yml
```

Activate the environment:
```bash
conda activate napari-env
```

---

## Usage
Run the script using:
```bash
python napari_viewer.py
```

This will open the Napari viewer with the available widgets, including options for image and mask loading, shape management, and cell quantification.

### GUI Features
- Open image: Load a multichannel image.
- Open mask: Load a segmentation mask.
- Load shapes: Import shape data from text files.
- Save shapes: Export the current shapes to a file.
- Contrast limits: Adjust image contrast interactively.
- Count cells: Analyze cell distributions within regions of interest.
- Export cells: Save cell data to CSV.
- Metadata: Display metadata associated with the current image.
- Voronoi: Generate Voronoi diagrams from cell positions.
- Close all: Close all active widgets.

---

## Troubleshooting
If you encounter any issues, make sure that all dependencies are correctly installed and that the environment is properly activated.

For additional help, please reach out to the authors at the emails provided above.

