# Quantification Scripts - Installation and Usage

## Overview
The quantification scripts are designed to process image data from the CycIF platform. This version is adapted from the Laboratory of Systems Pharmacology (LSP) repository: [LSP GitHub](https://github.com/labsyspharm).

**Author:** Cruz Osuna (cruzosuna2003@gmail.com)

---

## Installation
Ensure you have Conda installed on your system. To create the environment, execute the following command:

```bash
conda env create -f quantification.yml
```

This command will set up the necessary Python environment with all required dependencies.

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

The script leverages parallel processing to handle multiple images simultaneously (up to 8 jobs), significantly reducing processing time.

---

## Setting Up Parallel Execution
To configure parallel execution, follow these steps:

1. **Create the script file:** Save the following script as `quantification_parallel.sh`.

2. **Configuration:** Set the paths for the images, masks, channel names, and output directory. Example:

```bash
IMAGE_DIR="/path/to/images"
MASK_DIR="/path/to/masks"
CHANNEL_NAMES="/path/to/metadata/channels.csv"
OUTPUT_DIR="/path/to/output"
```

3. **Path Verification:** Ensure the specified directories and files exist before running the script.

4. **Parallel Processing:** Use the `find` command to search for image files and process them in parallel using GNU Parallel.

By following these instructions, you will be able to efficiently perform batch image quantification using parallel processing. Make sure to check the output directory for the results after the script finishes.

