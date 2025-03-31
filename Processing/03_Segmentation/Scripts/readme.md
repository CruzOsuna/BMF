## Stardist Segmentation Script - Installation and Usage

### Source and Author Information
- **Modified from:** [Färkkilä Lab - Image processing pipeline](https://github.com/farkkilab/image_processing/blob/main/pipeline/2_segmentation/stardist_segmentation.py)
- **Author:** Cruz Francisco Osuna Aguirre (cruzosuna2003@gmail.com)

---

## Setting Up the Segmentation Environment

### 1. Installation Using a Conda `.yml` File
To create the environment from the `.yml` file, run:
```bash
conda env create -f stardist_env.yml
```

### 2. Installation Using a Shell Script (Linux)
#### Step 1: Grant Execution Permissions
```bash
chmod +x install_stardist.sh
```
#### Step 2: Run the Installation Script
```bash
./install_stardist.sh
```

### 3. Manual Installation
#### Step 1: Create and Activate the Conda Environment
```bash
conda create -n stardist python=3.10
conda activate stardist
```

#### Step 2: Install TensorFlow (Automatically Installs a Compatible NumPy Version)
```bash
conda install -c conda-forge tensorflow=2.18.0
```

#### Step 3: Install StarDist and Dependencies
```bash
conda install -c conda-forge stardist
```

#### Step 4: Verify the Installed NumPy Version
```bash
conda list | grep numpy  # Should display numpy ~1.26.0
```

---

## Running the Segmentation Script

### Step 1: Ensure the Conda Environment Is Activated
```bash
conda activate stardist
```

Ensure that the correct paths are specified in `stardist_segmentation.py` scripts before execution.

Select the script to use depending on the GPU you have available, if your GPU has more than 32 GB of VRAM use the `GPU_high-end` script, if it has less than 32 use the `GPU_mid-range`.


### Step 2: Execute the Segmentation Script
```bash
python stardist_segmentation.py
```

---

Follow these steps to properly set up and execute the segmentation script. Modify paths and package versions as needed for your specific setup.

