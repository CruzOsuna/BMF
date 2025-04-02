#!/bin/bash

# Source of conda
source /home/cruz-osuna/anaconda3/etc/profile.d/conda.sh

# Activate conda env
conda activate cylinter

# Ask the user
echo "Select the option to execute:"
echo "1) 4NQO"
echo "2) P53"
read -p "Enter the option number: " option

case $option in
    1)
        echo "Running Cylinter with 4NQO..."
        cylinter /media/cruz-osuna/Mice/CycIF_mice_4NQO/5_QC/Cylinter/INPUT_DIR/cylinter_config.yml
        ;;
    2)
        echo "Running Cylinter with P53..."
        cylinter /media/cruz-osuna/Mice/CycIF_mice_p53/5_QC/Cylinter/INPUT_DIR/cylinter_config.yml
        ;;
    *)
        echo "Invalid option. Exiting..."
        exit 1
        ;;
esac