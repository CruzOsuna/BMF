import os
import re
import argparse
import multiprocessing
from os import listdir
from os.path import join, exists, basename, isfile
import subprocess

# ----------------- Configuration -----------------
my_path = "/media/cruz/Spatial/2025_2-2/01_Registration/rcpnl/"
output_path = "/media/cruz/Spatial/2025_2-2/02_Visualization/"
illumination_base = "/media/cruz/Spatial/2025_2-2/00_Illumination_Correction/"
# ------------------------------------------------

file_type = 'rcpnl'
illumination_enabled = True  # Set to False to disable

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', filename)]

def find_matching_illumination(rcpnl_path, illu_folder):
    """Match illumination files with .rcpnl in filename"""
    base = basename(rcpnl_path)  # e.g., "BM DAPI 10X 110625 4 CHNLS C1.rcpnl"
    ffp = join(illu_folder, f"{base}-ffp.tif")
    dfp = join(illu_folder, f"{base}-dfp.tif")
    return (ffp if isfile(ffp) else None, 
            dfp if isfile(dfp) else None)

def process_subfolder(subfolder, output_root, illumination_root):
    name = basename(subfolder)
    print(f"\n🔍 Processing: {name}")
    
    # Get rcpnl files
    rcpnl_files = sorted(
        [join(subfolder, f) for f in listdir(subfolder) 
         if f.endswith('.rcpnl') and isfile(join(subfolder, f))],
        key=lambda x: natural_sort_key(basename(x))
    )
    
    if not rcpnl_files:
        print(f"⚠️ No .rcpnl files in {name}")
        return

    # Get illumination files (CRITICAL FIX: keep .rcpnl in filenames)
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
                    print(f"⚠️ Missing illumination for {basename(rcpnl)}")
            # Validate counts
            if len(ffp_files) != len(rcpnl_files):
                print(f"❌ Illumination file count mismatch in {name}")
                ffp_files, dfp_files = [], []
        else:
            print(f"⚠️ No illumination folder: {illu_folder}")

    # Build Ashlar command
    output_file = join(output_root, f"{name}.ome.tif")
    cmd = [
        "ashlar",
        *rcpnl_files,
        "-o", output_file,
        "--filter-sigma", "1",
        "-m", "30",
        "--pyramid",
        "--tile-size", "512"
    ]
    
    # Add illumination if all files exist
    if ffp_files and dfp_files and len(ffp_files) == len(rcpnl_files):
        cmd += ["--ffp", *ffp_files, "--dfp", *dfp_files]
    elif illumination_enabled:
        print(f"⛔ Skipping illumination correction for {name}")

    print("\n🚀 Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
        print(f"✅ Success: {name}\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {name}\nError: {e.stderr}")
    except Exception as e:
        print(f"💥 Crash: {name}\n{str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--threads', type=int, default=4)
    args = parser.parse_args()

    subfolders = [
        f.path for f in os.scandir(my_path)
        if f.is_dir() and not f.name.startswith("._") and not f.name.startswith(".") # Filter folders whose names begin with ._ or . 
    ]

    print(f"Found {len(subfolders)} samples to process")
    
    with multiprocessing.Pool(args.threads) as pool:
        tasks = [(sf, output_path, illumination_base) for sf in subfolders]
        pool.starmap(process_subfolder, tasks)

    print("\n🏁 All processing complete!")
