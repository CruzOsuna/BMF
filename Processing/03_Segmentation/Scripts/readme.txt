# Instructions for the use of the segmentation script
# Modified from Färkkilä Lab
# https://github.com/farkkilab/image_processing/blob/main/pipeline/2_segmentation/stardist_segmentation.py
# Cruz Osuna


# Setup segmentation enviroment

conda create -n stardist python=3.10
conda activate stardist

# Install TensorFlow first (it will pull compatible numpy automatically)
conda install -c conda-forge tensorflow=2.18.0

# Install StarDist with its dependencies
conda install -c conda-forge stardist

# Verify numpy version
conda list | grep numpy  # Should show numpy ~1.26.0


# Run segmentation

conda activate stardist

python cli.py \
  --masks /home/cruz/Escritorio/03_Segmentation/masks/FAHNSCC_14.ome.tif
  --image /home/cruz/Escritorio/03_Segmentation/image/FAHNSCC_14.ome.tif
  --channel_names /home/cruz/Documentos/Taller INP/visualization/Metadata/channels.txt
  --output /home/cruz/Escritorio/BMF/04_Quantification/output
