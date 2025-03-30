#!/bin/bash

# CORRECT configuration with full paths
IMAGE_DIR="/media/cruz/Mice/CycIF_mice_p53/2_Visualization/t-CycIF/images_illumination_corrected"
MASK_DIR="/media/cruz/Mice/CycIF_mice_p53/3_Segmentation/Mask_Illumination-corrected"
CHANNEL_NAMES="/media/cruz/Mice/CycIF_mice_p53/4_Quantification/Metadata/channels.csv"
OUTPUT_DIR="/media/cruz/Mice/CycIF_mice_p53/4_Quantification/output"

# Verify the existence of directories
echo "Checking paths..."
[ -d "$IMAGE_DIR" ] || { echo "ERROR: Image directory does not exist: $IMAGE_DIR"; exit 1; }
[ -d "$MASK_DIR" ] || { echo "ERROR: Mask directory does not exist: $MASK_DIR"; exit 1; }
[ -f "$CHANNEL_NAMES" ] || { echo "ERROR: Channel names file does not exist: $CHANNEL_NAMES"; exit 1; }

# Set up Python environment
export PYTHONPATH="/home/cruz/Escritorio/BMF/Processing/04_Quantification/Scripts:$PYTHONPATH"

# Process files
find "$IMAGE_DIR" -name "*.ome.tif" | parallel -j 8 --eta --bar --progress \
"python3 cli.py \
  --masks \"$MASK_DIR/{/}\" \
  --image \"{}\" \
  --channel_names \"$CHANNEL_NAMES\" \
  --output \"$OUTPUT_DIR\""

echo "Processing completed. Results in: $OUTPUT_DIR"

