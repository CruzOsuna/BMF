#!/bin/bash

export _JAVA_OPTIONS="-Xms25g -Xmx25g" # RAM reduction to 25gb


# Define paths and file type
#MY_PATH="/data/input"      # Name of path modified
#OUTPUT_PATH="/data/output" # Name of path modified
#FILE_TYPE="rcpnl"


# Modification to use the SSD
# Dentro de BaSiC_run.sh
MY_PATH="/mnt/external/CycIF_human_2024/1_Registration/RCPNLS/Done/FAHNSCC_11C"
OUTPUT_PATH="/mnt/external/CycIF_human_2024/1_Registration/00_Illumination_correction/output/FAHNSCC_11C"
FILE_TYPE="rcpnl"


# Function to execute the ImageJ command
BaSiC_call() {
    local illumination_to_correct="$1"
    local output_path="$2"
    local file_name="$3"

    # Construct and run the command
    cmd="ImageJ-linux64 --ij2 --headless --run imagej_basic_ashlar.py \"filename='${illumination_to_correct}',output_dir='${output_path}/',experiment_name='${file_name}'\""
    echo "${cmd}"
    eval "${cmd}"
}

# Function to get the list of files
get_file_list() {
    local folder="$1"
    local files=""
    
    # Loop through the files and filter based on the file type
    for f in "$folder"/*; do
        if [[ "$f" == *$FILE_TYPE* ]]; then
            files="$files $f"
        fi
    done
    echo "$files"
}

# Main logic
echo "Source folder: $MY_PATH"

# Get the list of files
illumination_to_correct=$(get_file_list "$MY_PATH")

# Convert file list to an array
file_list=($illumination_to_correct)

# Iterate through each file and run the BaSiC_call function
for file in "${file_list[@]}"; do
    # Extract file name (basename)
    file_name=$(basename "$file")
    
    # Run the BaSiC_call function
    BaSiC_call "$file" "$OUTPUT_PATH" "$file_name"
done
