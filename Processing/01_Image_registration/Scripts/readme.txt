# Instructions for the use of image registration scripts
# Modified from: xxxxxxxxx
#
# Author: Cruz Osuna



## Create the conda enviroment

conda env create -f image_registration.yml


## Run stitching

conda activate image_registration

# Add your paths to python stitching.py

python stitching.py -c n #Where n is the number of threads that we want to use


