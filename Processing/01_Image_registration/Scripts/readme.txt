# Instructions for Using the Image Registration Scripts  
# Modified from: https://github.com/farkkilab/image_processing/blob/main/pipeline/1_stitching/ashlar_workflow.py  
#  
# Author: Cruz Osuna (cruzosuna2003@gmail.com)  


## Creating the Conda Environment  

To set up the required Conda environment, run:  

conda env create -f image_registration.yml



## Running the Stitching Script  

Activate the environment:  


conda activate image_registration



Before running the script, ensure that you have specified the appropriate paths in `stitching.py`.  

To execute the script, use the following command:  


python stitching.py -c n


Replace `n` with the number of threads you want to use for parallel processing. By default, 4 threads will be used unless you specify a different value.  
