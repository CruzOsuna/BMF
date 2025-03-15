# Installation instructions and use of the quantification scripts
# Modified from Laboratory of Systems Pharmacology: https://github.com/labsyspharm


# Instalation

conda env create -f quantification.yml



# Use

conda activate quantification

python cli.py \
  --masks "/BMF/03_Segmentation/masks/mask.ome.tif" \
  --image "/BMF/01_Registration/output/FAHNSCC_14/FAHNSCC_14.ome-001.tif" \
  --channel_names "/BMF/04_Quantification/Metadata/channels.csv" \
  --output "/BMF/04_Quantification/Results" 