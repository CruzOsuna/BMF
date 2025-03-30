## Instructions for Using the Segmentation Script

### Source and Author Information
- **Modified from:** [Färkkilä Lab](https://github.com/farkkilab/image_processing/blob/main/pipeline/2_segmentation/stardist_segmentation.py)
- **Author:** Cruz Osuna

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

### Step 2: Execute the Segmentation Script
```bash
python stardist_segmentation.py
```

---

Follow these steps to properly set up and execute the segmentation script. Modify paths and package versions as needed for your specific setup.

