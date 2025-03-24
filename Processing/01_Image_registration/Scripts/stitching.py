import os
import re
import subprocess
from os import listdir
from os.path import join, exists, basename

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

# Configuration
rcpnl_path = r"/media/cruz-osuna/Mice/CycIF_mice_p53/1_Registration/RCPNLS"
output_path = r"/media/cruz-osuna/Mice/CycIF_mice_p53/2_Visualization/t-CycIF/images_illumination_corrected"  # Fixed typo in "illumination"
subfolders = [f.path for f in os.scandir(rcpnl_path) if f.is_dir()]

# Illumination parameters
illumination_enabled = True
illumination_type = 'both'
illumination_folder = r"/media/cruz-osuna/Mice/CycIF_mice_p53/00_Illumination_correction/Output"

# Create output directory if missing
os.makedirs(output_path, exist_ok=True)

print(f"Found {len(subfolders)} subfolders to process")

for subfolder in subfolders:
    print(f"\n{'='*40}\nProcessing: {subfolder}")
    name = basename(subfolder)
    output_file = join(output_path, f"{name}.ome.tif")

    # Get .rcpnl files with natural sorting
    files = sorted([f for f in listdir(subfolder) if f.endswith('.rcpnl')], key=natural_sort_key)
    if not files:
        print(f"No .rcpnl files found in {subfolder}. Skipping.")
        continue
    
    # Build quoted file list for Ashlar
    files_to_stitch = " ".join([f'"{join(subfolder, f)}"' for f in files])

    # Illumination correction handling
    ffp_str, dfp_str = "", ""
    if illumination_enabled:
        illumination_subfolder = join(illumination_folder, name)
        if not exists(illumination_subfolder):
            print(f"! Warning: No illumination folder found for {name}")
        else:
            try:
                # Look for TIFF illumination files with correct patterns
                ffp_files = []
                dfp_files = []
                
                for f in sorted(listdir(illumination_subfolder), key=natural_sort_key):
                    if f.endswith('-ffp.tif') and illumination_type in ['ffp', 'both']:
                        ffp_files.append(f)
                    elif f.endswith('-dfp.tif') and illumination_type in ['dfp', 'both']:
                        dfp_files.append(f)

                # Build file paths with full subfolder path
                if ffp_files:
                    ffp_str = ",".join([f'"{join(illumination_subfolder, f)}"' for f in ffp_files])
                if dfp_files:
                    dfp_str = ",".join([f'"{join(illumination_subfolder, f)}"' for f in dfp_files])

                print(f"Found {len(ffp_files)} FFP and {len(dfp_files)} DFP files for {name}")
                
            except Exception as e:
                print(f"Error loading illumination files: {str(e)}")

    # Build Ashlar command
    cmd = f'ashlar {files_to_stitch} -o "{output_file}" --pyramid --filter-sigma 1 -m 30'
    if ffp_str:
        cmd += f' --ffp {ffp_str}'
    if dfp_str:
        cmd += f' --dfp {dfp_str}'

    print(f"\nCommand:\n{cmd}")
    
    # Execute with better interrupt handling
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"Successfully processed {name}")
    except KeyboardInterrupt:
        print("\nUser interrupted process. Exiting...")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {name}:\n{e.stderr[:1000]}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

print("\nProcessing complete!")