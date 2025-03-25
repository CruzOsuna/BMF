# Instructions for Using the Segmentation Script
# Modified from Färkkilä Lab
# https://github.com/farkkilab/image_processing/blob/main/pipeline/2_segmentation/stardist_segmentation.py
# Author: Cruz Osuna

## Setup the Segmentation Environment

### Installation using a Conda `.yml` file

To create the environment from the `.yml` file, run:

conda env create -f stardist_env.yml



### Installation using a `.sh` script (for linux)

1. Grant execution permissions to the script:

chmod +x install_stardist.sh


2. Run the installation script:

./install_stardist.sh



### Manual Installation

1. Create and activate the Conda environment:

conda create -n stardist python=3.10 conda activate stardist


2. Install TensorFlow (this will automatically install a compatible NumPy version):

conda install -c conda-forge tensorflow=2.18.0


3. Install StarDist and its dependencies:

conda install -c conda-forge stardist


4. Verify the installed NumPy version:

conda list | grep numpy # Should display numpy ~1.26.0



---

## Running the Segmentation Script

1. Ensure the Conda environment is activated:

conda activate stardist



2. Execute the segmentation script:

python stardist_segmentation.py