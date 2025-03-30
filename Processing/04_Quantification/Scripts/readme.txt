# Installation instructions and usage of the quantification scripts

## Overview
The quantification scripts are designed to process image data from the CycIF platform. This version is adapted from the Laboratory of Systems Pharmacology (LSP) repository: [LSP GitHub](https://github.com/labsyspharm). 

**Author:** Cruz Osuna (cruzosuna2003@gmail.com)

---

## Installation
Make sure you have Conda installed on your system. To create the environment, run the following command:

```bash
conda env create -f quantification.yml
```

This will set up the required Python environment with all necessary dependencies.

---

## Usage
Activate the environment and run the script as follows:

```bash
conda activate quantification

python cli.py \
  --masks "/path/to/masks/mask.ome.tif" \
  --image "/path/to/images/image.ome.tif" \
  --channel_names "/path/to/metadata/channels.csv" \
  --output "/path/to/output"
```

### Parallel Execution
To run the quantification script in parallel for multiple images, first make the script executable:

```bash
chmod +x quantification_parallel.sh
```

Then, execute the script:

```bash
./quantification_parallel.sh
```

The script will utilize parallel processing with up to 8 jobs simultaneously, speeding up the quantification process.

---

## Configuration for Parallel Execution
To set up parallel execution, create a file named `quantification_parallel.sh` with the necessary paths and parameters. The script should include the following sections:

1. **Configuration:** Define the paths for the images, masks, channel names, and output directory. Example:

```bash
IMAGE_DIR="/path/to/images"
MASK_DIR="/path/to/masks"
CHANNEL_NAMES="/path/to/metadata/channels.csv"
OUTPUT_DIR="/path/to/output"
```

2. **Path Verification:** Check if the specified directories and files exist.

3. **Execution:** Use the `find` command to locate the image files and run the processing script in parallel using GNU Parallel.

By following these instructions, you will be able to perform batch image quantification efficiently using parallel processing.

