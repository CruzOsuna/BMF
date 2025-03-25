print("executed")
import os
import re
import subprocess
import tifffile  # Added for channel detection
from os import listdir
from os.path import join, exists, basename

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

# Configuration
rcpnl_path = r"/media/cruz-osuna/Mice/CycIF_mice_p53/1_Registration/RCPNLS"
output_path = r"/media/cruz-osuna/Mice/CycIF_mice_p53/2_Visualization/t-CycIF/images_illumination_corrected"
subfolders = [f.path for f in os.scandir(rcpnl_path) if f.is_dir()]

# Illumination parameters
illumination_enabled = True
illumination_type = 'both'
illumination_folder = r"/media/cruz-osuna/Mice/CycIF_mice_p53/00_Illumination_correction/Output"

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
    
    files_to_stitch = " ".join([f'"{join(subfolder, f)}"' for f in files])

    ffp_str, dfp_str = "", ""
    if illumination_enabled:
        current_illumination_dir = join(illumination_folder, name)
        if not exists(current_illumination_dir):
            print(f"Illumination directory not found: {current_illumination_dir}. Skipping illumination correction.")
        else:
            # Get most recent files
            ffp_files = sorted(
                [f for f in listdir(current_illumination_dir) if f.endswith('-ffp.tif')],
                key=natural_sort_key,
                reverse=True
            )
            dfp_files = sorted(
                [f for f in listdir(current_illumination_dir) if f.endswith('-dfp.tif')],
                key=natural_sort_key,
                reverse=True
            )

            # Process FFP
            if ffp_files and illumination_type in ['ffp', 'both']:
                ffp_path = join(current_illumination_dir, ffp_files[0])
                # Check if multi-channel and add channel selector
                if tifffile.imread(ffp_path).ndim == 3:
                    ffp_str = f'"{ffp_path}@C=0"'
                else:
                    ffp_str = f'"{ffp_path}"'

            # Process DFP
            if dfp_files and illumination_type in ['dfp', 'both']:
                dfp_path = join(current_illumination_dir, dfp_files[0])
                # Check if multi-channel and add channel selector
                if tifffile.imread(dfp_path).ndim == 3:
                    dfp_str = f'"{dfp_path}@C=0"'
                else:
                    dfp_str = f'"{dfp_path}"'

    cmd = f'ashlar {files_to_stitch} -o "{output_file}" --pyramid --filter-sigma 1 -m 30'
    if ffp_str:
        cmd += f' --ffp {ffp_str}'
    if dfp_str:
        cmd += f' --dfp {dfp_str}'

    print("\nCommand:", cmd)
    
    try:
        subprocess.run(cmd, check=True, shell=True)
        print(f"Successfully processed {name}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {name}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error with {name}: {str(e)}")

print("\nProcessing complete!")