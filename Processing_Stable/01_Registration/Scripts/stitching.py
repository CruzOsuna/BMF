import os
import re
import argparse
import multiprocessing
import logging
from datetime import datetime
from os import listdir
from os.path import join, exists, basename, splitext, isfile
import subprocess
from tqdm import tqdm

# ----------------- Configuration -----------------
input_path = "/media/cruz/Spatial/t-CycIF_human_2025_2/01_Registration/RCPNL/"
output_path = "/media/cruz/Spatial/t-CycIF_human_2025_2/02_Visualization/t-CycIF/Images_IC/"
illumination_base = "/media/cruz/Spatial/t-CycIF_human_2025_2/00_Illumination correction/IC_files"
# -------------------------------------------------

supported_formats = ('rcpnl', 'czi')
illumination_enabled = True

# Setup logging
def setup_logger():
    log_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"processing_log_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_path

def natural_sort_key(filename):
    cycle_match = re.search(r'CYCLE (\d+)_', filename)
    cycle = int(cycle_match.group(1)) if cycle_match else 0
    return [cycle] + [int(text) if text.isdigit() else text.lower() 
                     for text in re.split(r'(\d+)', filename)]

def find_matching_illumination(file_path, illu_folder):
    """Match illumination files with cycle prefixes"""
    base_name = basename(file_path)
    ffp = join(illu_folder, f"{base_name}-ffp.tif")
    dfp = join(illu_folder, f"{base_name}-dfp.tif")
    return (ffp if isfile(ffp) else None, dfp if isfile(dfp) else None)

def process_subfolder(subfolder, output_root, illumination_root, force=False):
    folder_name = basename(subfolder)
    output_file = join(output_root, f"{folder_name}.ome.tif")
    
    if exists(output_file) and not force:
        logging.info(f"Skipping {folder_name}, output exists")
        return
    
    logging.info(f"\n🔍 Processing: {folder_name}")
    
    try:
        input_files = sorted(
            [join(subfolder, f) for f in listdir(subfolder) 
             if f.lower().endswith(supported_formats)],
            key=lambda x: natural_sort_key(basename(x))
        )
        
        if not input_files:
            logging.warning(f"No supported files in {folder_name}")
            return

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
                        logging.warning(f"Missing illumination for {basename(file_path)}")
                if len(ffp_files) != len(input_files):
                    logging.warning(f"Illumination mismatch in {folder_name}. Skipping correction.")
                    ffp_files, dfp_files = [], []

        cmd = ["ashlar"]
        for i, input_file in enumerate(input_files):
            cmd.append(input_file)
            if ffp_files and dfp_files:
                cmd.extend(["--ffp", ffp_files[i], "--dfp", dfp_files[i]])
        
        cmd += [
            "-o", output_file,
            "--filter-sigma", "1",
            "-m", "30"
        ]

        logging.info("\nCommand: " + " ".join(cmd))
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        logging.info(f"Success: {folder_name}\n{result.stdout}")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed: {folder_name}\nError: {e.stdout}")
    except Exception as e:
        logging.error(f"Crash: {folder_name}\n{str(e)}", exc_info=True)
    finally:
        logging.info("-" * 50)

def process_wrapper(args):
    return process_subfolder(*args)

if __name__ == '__main__':
    # Initialize logging
    log_path = setup_logger()
    logging.info(f"Log file created at: {log_path}")
    
    parser = argparse.ArgumentParser(description='Process microscopy images with Ashlar')
    parser.add_argument('-c', '--threads', type=int, default=4)
    parser.add_argument('-f', '--force', action='store_true', help='Force reprocessing')
    args = parser.parse_args()

    try:
        subfolders = [f.path for f in os.scandir(input_path) if f.is_dir()]
        os.makedirs(output_path, exist_ok=True)

        logging.info(f"Found {len(subfolders)} samples")
        
        tasks = [(sf, output_path, illumination_base, args.force) for sf in subfolders]
        
        with multiprocessing.Pool(args.threads) as pool:
            list(tqdm(
                pool.imap_unordered(process_wrapper, tasks),
                total=len(tasks),
                desc="Processing",
                unit="sample"
            ))

        logging.info("\n🏁 Processing Complete!")
        logging.info(f"Full log available at: {log_path}")

    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}", exc_info=True)