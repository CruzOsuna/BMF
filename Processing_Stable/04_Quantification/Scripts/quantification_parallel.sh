#!/bin/bash

# Configuration paths
IMAGE_DIR="/home/bonem/NAS_Projects/t-CycIF/t-CycIF_human_2024/02_Visualization/t-CycIF/Images_IC/"
MASK_DIR="/home/bonem/NAS_Projects/t-CycIF/t-CycIF_human_2024/03_Segmentation/Mask_IC/"
CHANNEL_NAMES="/home/bonem/NAS_Projects/t-CycIF/t-CycIF_human_2024/04_Quantification/Metadata/channels.csv"
OUTPUT_DIR="/home/bonem/NAS_Projects/t-CycIF/t-CycIF_human_2024/08_Results/Datasets/0_Raw_data/IC/"

# Output naming pattern
OUTPUT_SUFFIX="_quantified.csv"

# Create output directory if missing
mkdir -p "$OUTPUT_DIR" || { echo "ERROR: Failed to create output directory"; exit 1; }

# Validate inputs
[ -d "$IMAGE_DIR" ] || { echo "ERROR: Image directory missing: $IMAGE_DIR"; exit 1; }
[ -d "$MASK_DIR" ] || { echo "ERROR: Mask directory missing: $MASK_DIR"; exit 1; }
[ -f "$CHANNEL_NAMES" ] || { echo "ERROR: Channel names file missing: $CHANNEL_NAMES"; exit 1; }

# Configure Python environment
export PYTHONPATH="/home/bonem/Desktop/BMF_t-CyCIF/Processing_Stable/04_Quantification/Scripts/:$PYTHONPATH"

LOG_FILE="$OUTPUT_DIR/processing.log"
echo "Processing started: $(date)" > "$LOG_FILE"

find "$IMAGE_DIR" -name "*.ome.tif" | while read -r IMAGE; do
    BASE_NAME=$(basename "$IMAGE" .ome.tif)
    OUTPUT_CSV="$OUTPUT_DIR/${BASE_NAME}${OUTPUT_SUFFIX}"
    
    # Check for existing output
    if [ -f "$OUTPUT_CSV" ]; then
        echo "WARNING: Output exists for $IMAGE ($OUTPUT_CSV), skipping..." >> "$LOG_FILE"
        continue
    fi
    
    # Try different mask filename patterns
    MASK_CANDIDATES=(
        "$MASK_DIR/${BASE_NAME}.ome.ome.tif"  # Most common case
        "$MASK_DIR/${BASE_NAME}.ome.tif"      # Some files
        "$MASK_DIR/${BASE_NAME}.tif"          # One file
    )
    
    MASK_FOUND=0
    for MASK in "${MASK_CANDIDATES[@]}"; do
        if [ -f "$MASK" ]; then
            MASK_FOUND=1
            break
        fi
    done
    
    if [ "$MASK_FOUND" -eq 0 ]; then
        echo "WARNING: Missing mask for $IMAGE (tried ${MASK_CANDIDATES[*]}), skipping..." >> "$LOG_FILE"
        continue
    fi
    
    echo "$IMAGE"
done | parallel -j 2 --eta --bar --progress \
"python3 cli.py \
  --masks \"$MASK_DIR/{/.}.ome.ome.tif\" \
  --image \"{}\" \
  --channel_names \"$CHANNEL_NAMES\" \
  --output \"$OUTPUT_DIR\" >> \"$LOG_FILE\" 2>&1"

echo "Processing complete. CSV files in: $OUTPUT_DIR"
