#!/bin/bash

# This script automates the setup of a Conda environment for StarDist by:
# 1) Creating a new Conda environment named stardist with Python 3.10.
# 2) Activating the environment.
# 3) Installing TensorFlow 2.18.0 from Conda-Forge.
# 4) Installing StarDist along with its dependencies.
# 5) Verifying the installed version of numpy.
# 
# It ensures a streamlined and reproducible installation process for using StarDist.
#
# To run the script, first give execution permissions:
# chmod +x install_stardist.sh
#
# Then execute:
# ./install_stardist.sh
#
# Author: Cruz Osuna

set -e  # Exit immediately if any command fails

echo "Creating the Conda environment: stardist..."
conda create -n stardist python=3.10 -y || { echo "Failed to create the Conda environment."; exit 1; }

echo "Activating the Conda environment..."
source activate stardist || conda activate stardist || { echo "Failed to activate the Conda environment."; exit 1; }

echo "Installing TensorFlow 2.18.0..."
conda install -c conda-forge tensorflow=2.18.0 -y || { echo "Failed to install TensorFlow."; exit 1; }

echo "Installing StarDist..."
conda install -c conda-forge stardist -y || { echo "Failed to install StarDist."; exit 1; }

echo "Verifying numpy version..."
conda list | grep numpy || { echo "Failed to find numpy."; exit 1; }

echo "Installation completed successfully."
