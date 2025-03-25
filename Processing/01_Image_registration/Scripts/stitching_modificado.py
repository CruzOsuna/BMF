print("executed")
import os
import re
import argparse
import subprocess
from os import listdir
from os.path import join, exists, basename

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

# Configuration
rcpnl_path = r"/media/cruz-osuna/Mice/CycIF_mice_p53/1_Registration/RCPNLS/3_FA2664P53_2_5" # Use a directory containing the subdirectories of each image, not directly the folder of an image.
output_path = r"/media/cruz-osuna/Mice/CycIF_mice_p53/2_Visualization/t-CycIF/images_illumination_corrected"
subfolders = [f.path for f in os.scandir(rcpnl_path) if f.is_dir()]

# Illumination parameters
illumination_enabled = True          # Set to False to disable
illumination_type = 'both'           # 'both', 'ffp', 'dfp'
illumination_folder = r"/media/cruz-osuna/Mice/CycIF_mice_p53/00_Illumination_correction/Output/3_FA2664P53_2_5"

# Create output directory if missing
os.makedirs(output_path, exist_ok=True)

print("Subfolders found:", subfolders)

for subfolder in subfolders:
    print(f"\nProcessing: {subfolder}")
    name = basename(subfolder)
    output_file = join(output_path, f"{name}.ome.tif")

    # Get .rcpnl files with natural sorting
    files = sorted([f for f in listdir(subfolder) if f.endswith('.rcpnl')], key=natural_sort_key)
    if not files:
        print(f"No .rcpnl files found in {subfolder}. Skipping.")
        continue
    
    # Build quoted file list for Ashlar
    files_to_stitch = " ".join([f'{join(subfolder, f)}' for f in files])

    # Handle illumination correction
    ffp_str, dfp_str = "", ""
    if illumination_enabled and exists(illumination_folder):
        # Get illumination files with natural sorting
        illu_files = sorted(listdir(illumination_folder), key=natural_sort_key)
        
        # Filter files based on illumination type
        ffp_files = []
        dfp_files = []
        if illumination_type in ['ffp', 'both']:
            ffp_files = [f for f in illu_files if f.endswith('-ffp.tif')]
        if illumination_type in ['dfp', 'both']:
            dfp_files = [f for f in illu_files if f.endswith('-dfp.tif')]

        print(ffp_files)

        # Build file paths
        if ffp_files:
            ffp_str = " ".join([f'{join(illumination_folder, f)}' for f in ffp_files])
        if dfp_files:
            dfp_str = " ".join([f'{join(illumination_folder, f)}' for f in dfp_files])

    # Build Ashlar command
    cmd = f'ashlar {files_to_stitch} -o "{output_file}" --pyramid --filter-sigma 1 -m 30'
    if ffp_str:
        cmd += f' --ffp {ffp_str}'
    if dfp_str:
        cmd += f' --dfp {dfp_str}'

    print("\nCommand:", cmd)
    
    # Execute with error handling
    try:
        subprocess.run(cmd, check=True, shell=True)
        print(f"Successfully processed {name}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {name}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error with {name}: {str(e)}")

print("\nProcessing complete!")