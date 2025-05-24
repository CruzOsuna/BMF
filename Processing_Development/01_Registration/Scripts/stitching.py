import os
import re
import argparse
import multiprocessing
from os import listdir
from os.path import join, exists, basename, splitext, isfile
import subprocess
from tqdm import tqdm


# Author: Cruz Osuna (cruzosuna2003@gmail.com)
# Modified from: ... pending add the source 


# ----------------- Configuration (Update these paths) -----------------
input_path = "/media/cruz/TOSHIBA EXT1/BMF/ENSAYO CycIF INCAN/01_Registration"
output_path = "/media/cruz/TOSHIBA EXT1/BMF/ENSAYO CycIF INCAN/02_Visualization/Images"
illumination_base = "/media/cruz/TOSHIBA EXT1/BMF/ENSAYO CycIF INCAN/00_Illumination_correction"
# -----------------------------------------------------------------------

supported_formats = ('rcpnl', 'czi')
illumination_enabled = True

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', filename)]

def find_matching_illumination(file_path, illu_folder):
    """Find illumination files with CZI extension in names"""
    base_name = basename(file_path)  # Keep CZI extension for matching
    ffp = join(illu_folder, f"{base_name}-ffp.tif")
    dfp = join(illu_folder, f"{base_name}-dfp.tif")
    return (ffp if isfile(ffp) else None, dfp if isfile(dfp) else None)


# This function below was temporarily modified to find files with czi in the name, pending correction as they are generated in the lighting correction.

# Original version
#def find_matching_illumination(file_path, illu_folder):
#    """Find matching illumination correction files"""
#    base_name = splitext(basename(file_path))[0]  # Remove extension
#    ffp = join(illu_folder, f"{base_name}-ffp.tif")
#    dfp = join(illu_folder, f"{base_name}-dfp.tif")
#    return (ffp if isfile(ffp) else None, 
#            dfp if isfile(dfp) else None)


def process_subfolder(subfolder, output_root, illumination_root):
    folder_name = basename(subfolder)
    output_file = join(output_root, f"{folder_name}.ome.tif")
    
    if exists(output_file):
        print(f"Skipping {folder_name}, output exists")
        return
    
    print(f"\n🔍 Processing: {folder_name}")
    
    # Get input files with natural sort
    input_files = sorted(
        [join(subfolder, f) for f in listdir(subfolder) 
         if f.lower().endswith(supported_formats) and isfile(join(subfolder, f))],
        key=lambda x: natural_sort_key(basename(x))
    )
    
    if not input_files:
        print(f"No supported files in {folder_name}")
        return

    # CZI-specific parameters
    has_czi = any(f.lower().endswith('.czi') for f in input_files)
    czi_params = []
    if has_czi:
        czi_params = [
            "--align-channel", "0",
            "--flip-y",
            "--pyramid"
        ]

    # Illumination profiles
    ffp_files, dfp_files = [], []
    if illumination_enabled:
        illu_folder = join(illumination_root, folder_name)
        if exists(illu_folder):
            for file_path in input_files:
                ffp, dfp = find_matching_illumination(file_path, illu_folder)
                if ffp and dfp:
                    ffp_files.append(ffp)
                    dfp_files.append(dfp)
                else:
                    print(f"Missing illumination for {basename(file_path)}")
            if len(ffp_files) != len(input_files):
                print(f"Illumination mismatch in {folder_name}. Skipping correction.")
                ffp_files, dfp_files = [], []

    # Build command
    cmd = [
        "ashlar",
        *input_files,
        "-o", output_file,
        "--filter-sigma", "1",
        "-m", "30",
        *czi_params  # CZI-specific params added here
    ]
    
    if ffp_files and dfp_files:
        cmd += ["--ffp", *ffp_files, "--dfp", *dfp_files]

    print("\nCommand:", " ".join(cmd))
    
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
        print(f"Success: {folder_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed: {folder_name}\nError: {e.stderr}")
    except Exception as e:
        print(f"Crash: {folder_name}\n{str(e)}")

def process_wrapper(args):
    return process_subfolder(*args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process microscopy images with Ashlar')
    parser.add_argument('-c', '--threads', type=int, default=4)
    args = parser.parse_args()

    subfolders = [f.path for f in os.scandir(input_path) if f.is_dir()]
    os.makedirs(output_path, exist_ok=True)

    print(f"Found {len(subfolders)} samples")
    
    tasks = [(sf, output_path, illumination_base) for sf in subfolders]
    
    with multiprocessing.Pool(args.threads) as pool:
        list(tqdm(
            pool.imap_unordered(process_wrapper, tasks),
            total=len(tasks),
            desc="Processing",
            unit="sample"
        ))

    print("\n🏁 Complete!")