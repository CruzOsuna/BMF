#!/bin/bash

# Configuration paths
IMAGE_DIR="\\NAS_BMF_LAB\Projects\t-CycIF\t-CycIF_human_2025\02_Visualization\t-CycIF\Images_IC"
MASK_DIR="\\NAS_BMF_LAB\Projects\t-CycIF\t-CycIF_human_2025\03_Segmentation\Mask_IC"
CHANNEL_NAMES="\\NAS_BMF_LAB\Projects\t-CycIF\t-CycIF_human_2025\04_Quantification\Metadata\channels.csv"
OUTPUT_DIR="\\NAS_BMF_LAB\Projects\t-CycIF\t-CycIF_human_2025\08_Results\Datasets\00_Raw_data"

# Output naming pattern (verify with your cli.py's output)
OUTPUT_SUFFIX="_quantified.csv"  # Example: image.ome.tif → image_quantified.csv

# Create output directory if missing
mkdir -p "$OUTPUT_DIR" || { echo "ERROR: Failed to create output directory"; exit 1; }

# Validate inputs
[ -d "$IMAGE_DIR" ] || { echo "ERROR: Image directory missing: $IMAGE_DIR"; exit 1; }
[ -d "$MASK_DIR" ] || { echo "ERROR: Mask directory missing: $MASK_DIR"; exit 1; }
[ -f "$CHANNEL_NAMES" ] || { echo "ERROR: Channel names file missing: $CHANNEL_NAMES"; exit 1; }

# Configure Python environment
export PYTHONPATH="/home/cruz/Escritorio/BMF/Processing/04_Quantification/Scripts:$PYTHONPATH"

# Process only images that meet:
# 1. Corresponding mask exists
# 2. Output CSV doesn't exist
LOG_FILE="$OUTPUT_DIR/processing.log"
echo "Processing started: $(date)" > "$LOG_FILE"

find "$IMAGE_DIR" -name "*.ome.tif" | while read -r IMAGE; do
    BASE_NAME=$(basename "$IMAGE" .ome.tif)
    MASK="$MASK_DIR/$(basename "$IMAGE")"
    OUTPUT_CSV="$OUTPUT_DIR/${BASE_NAME}${OUTPUT_SUFFIX}"
    
    # Check for mask existence
    if [ ! -f "$MASK" ]; then
        echo "WARNING: Missing mask for $IMAGE, skipping..." >> "$LOG_FILE"
        continue
    fi
    
    # Check for existing output
    if [ -f "$OUTPUT_CSV" ]; then
        echo "WARNING: Output exists for $IMAGE ($OUTPUT_CSV), skipping..." >> "$LOG_FILE"
        continue
    fi
    
    echo "$IMAGE"  # Pass valid images to parallel
done | parallel -j 2 --eta --bar --progress \
"python3 cli.py \
  --masks \"$MASK_DIR/{/}\" \
  --image \"{}\" \
  --channel_names \"$CHANNEL_NAMES\" \
  --output \"$OUTPUT_DIR\" 2>> \"$LOG_FILE\""

echo "Processing complete. CSV files in: $OUTPUT_DIR"
