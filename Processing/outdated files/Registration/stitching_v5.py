import os
import re
import argparse
import multiprocessing
from os import listdir
from os.path import join, exists, basename, splitext, isfile
import subprocess
from tqdm import tqdm

# ----------------- Configuration (Update these paths) -----------------
my_path = "/media/cruz/Spatial/t-CycIF_human_2025/01_Registration/RCPNLS"
output_path = "/media/cruz/Spatial/t-CycIF_human_2025/02_Visualization/t-CycIF/Images_IC"
illumination_base = "/media/cruz/Spatial/t-CycIF_human_2025/00_Illumination_correction/output"
# -----------------------------------------------------------------------

file_type = 'rcpnl'
illumination_enabled = True  # Set to False to disable

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', filename)]

def find_matching_illumination(rcpnl_path, illu_folder):
    """Find matching illumination files with .rcpnl-ffp.tif pattern"""
    base = basename(rcpnl_path)  # Keep .rcpnl in the base name
    ffp = join(illu_folder, f"{base}-ffp.tif")
    dfp = join(illu_folder, f"{base}-dfp.tif")
    return (ffp if isfile(ffp) else None, 
            dfp if isfile(dfp) else None)

def process_subfolder(subfolder, output_root, illumination_root):
    """Process one sample subfolder"""
    name = basename(subfolder)
    output_file = join(output_root, f"{name}.ome.tif")
    
    if exists(output_file):
        print(f"Skipping {name}, output already exists.")
        return
    
    print(f"\n🔍 Processing: {name}")
    
    # Get RCPNL files
    rcpnl_files = sorted(
        [join(subfolder, f) for f in listdir(subfolder) 
         if f.endswith('.rcpnl') and isfile(join(subfolder, f))],
        key=lambda x: natural_sort_key(basename(x))
    )
    
    if not rcpnl_files:
        print(f"No .rcpnl files in {name}")
        return

    # Get illumination files
    ffp_files, dfp_files = [], []
    if illumination_enabled:
        illu_folder = join(illumination_root, name)
        if exists(illu_folder):
            for rcpnl in rcpnl_files:
                ffp, dfp = find_matching_illumination(rcpnl, illu_folder)
                if ffp and dfp:
                    ffp_files.append(ffp)
                    dfp_files.append(dfp)
                else:
                    print(f"Missing illumination for {basename(rcpnl)}")
            # Validate counts
            if len(ffp_files) != len(rcpnl_files) or len(dfp_files) != len(rcpnl_files):
                print(f"Illumination mismatch in {name}. Skipping correction.")
                ffp_files, dfp_files = [], []
        else:
            print(f"No illumination folder for {name}")

    # Build Ashlar command
    cmd = [
        "ashlar",
        *rcpnl_files,
        "-o", output_file,
        "--filter-sigma", "1",
        "-m", "30"
    ]
    if ffp_files and dfp_files:
        cmd += ["--ffp", *ffp_files, "--dfp", *dfp_files]

    print("\n Command:", " ".join(cmd))
    
    # Execute
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
        print(f"Success: {name}\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Failed: {name}\nError: {e.stderr}")
    except Exception as e:
        print(f"Crash: {name}\n{str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--threads', type=int, default=4)
    args = parser.parse_args()

    subfolders = [f.path for f in os.scandir(my_path) if f.is_dir()]
    os.makedirs(output_path, exist_ok=True)

    print(f"Found {len(subfolders)} samples to process")
    
    # Create a list of tasks for progress tracking
    tasks = [(sf, output_path, illumination_base) for sf in subfolders]
    
    # Process with progress bar
    with multiprocessing.Pool(args.threads) as pool:
        # Using tqdm to show progress
        list(tqdm(
            pool.imap_unordered(
                lambda x: process_subfolder(*x), 
                tasks
            ),
            total=len(tasks),
            desc="Processing samples",
            unit="sample"
        ))

    print("\n🏁 All processing complete!")