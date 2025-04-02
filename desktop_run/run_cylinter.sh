#!/bin/bash

# Source of conda
source /home/cruz-osuna/anaconda3/etc/profile.d/conda.sh

# Activate conda env
conda activate cylinter

# Ask the user
echo "Select the option to execute:"
echo "1) Mice 4NQO"
echo "2) Mice P53"
echo "3) Human 2024"
echo "4) Human 2025"
read -p "Enter the option number: " option

case $option in
    1)
        echo "Running Cylinter with 4NQO..."
        cylinter /media/cruz/Mice/CycIF_mice_4NQO/5_QC/Cylinter/INPUT_DIR/cylinter_config.yml
        ;;
    2)
        echo "Running Cylinter with P53..."
        cylinter /media/cruz/Mice/CycIF_mice_4NQO/5_QC/Cylinter/INPUT_DIR/cylinter_config.yml
        ;;
    3)
        echo "Running Cylinter with P53..."
        cylinter /media/cruz/Spatial/CycIF_mice_4NQO/5_QC/Cylinter/INPUT_DIR/cylinter_config.yml
        ;;
    4)
        echo "Running Cylinter with P53..."
        cylinter /media/cruz/Spatial/CycIF_mice_4NQO/5_QC/Cylinter/INPUT_DIR/cylinter_config.yml
        ;;
    *)
        echo "Invalid option. Exiting..."
        exit 1
        ;;
esac