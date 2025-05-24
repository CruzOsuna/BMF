#!/bin/bash

# --------------------------
# Paths (UPDATE THESE!)
# --------------------------
IMAGES_DIR="/media/cruz-osuna/Mice/CycIF_mice_p53/2_Visualization/t-CycIF/images_illumination_corrected"
MASKS_DIR="/media/cruz-osuna/Mice/CycIF_mice_p53/3_Segmentation/Mask_Iluminattion-corrected"
CHANNEL_TXT="/media/cruz-osuna/Mice/CycIF_mice_p53/2_Visualization/t-CycIF/Metadata/channels.txt"  # Your original text file
CHANNEL_CSV="$(dirname "$CHANNEL_TXT")/channels_Mice-P53.csv"  # Auto-generated CSV
OUTPUT_DIR="/media/cruz-osuna/Mice/CycIF_mice_p53/4_Quantification"

# --------------------------
# Convert TXT to CSV
# --------------------------
echo "Converting channel list to CSV..."
echo "marker_name" > "$CHANNEL_CSV"  # Create header
cat "$CHANNEL_TXT" >> "$CHANNEL_CSV"  # Append channel names

# --------------------------
# Process all images
# --------------------------
for IMAGE in "$IMAGES_DIR"/*.ome.tif; do
  # Get filename without path
  FILENAME=$(basename "$IMAGE")
  
  # Find corresponding mask (same filename)
  MASK="$MASKS_DIR/$FILENAME"
  
  echo "Processing: $FILENAME"
  echo "Mask: $MASK"
  
  # Run quantification
  python -m mcquant \
    --masks "$MASK" \
    --image "$IMAGE" \
    --channel_names "$CHANNEL_CSV" \
    --output "$OUTPUT_DIR" \
    --mask_props "area" "eccentricity" \
    --intensity_props "gini_index" "intensity_median"

  echo "---------------------------------------"
done

echo "All images processed! Results saved to: $OUTPUT_DIR"