#!/bin/bash

export _JAVA_OPTIONS="-Xms25g -Xmx25g" # RAM

# Define paths and file type
MY_PATH="/data/input/"     # Do not modify this path, indicate the path in the “sudo docker run...” command.
OUTPUT_PATH="/data/output" # Do not modify this path, indicate the path in the “sudo docker run...” command.
FILE_TYPE="rcpnl"           # Enter here the file format to be processed, either “czi” or “rcpnl”.

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

# Function to get the sorted list of files
get_file_list() {
    local folder="$1"
    local files=""

    # Loop through the files and filter based on the file type
    for f in "$folder"/*; do
        if [[ "$f" == *$FILE_TYPE* ]]; then
            files+=$'\n'"$f"
        fi
    done
    # Sort files numerically
    echo "$files" | sort -V
}

# Main logic
echo "Source folder: $MY_PATH"

# Process each immediate subdirectory under MY_PATH
find "$MY_PATH" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z | while IFS= read -r -d '' subdir; do
    echo "Processing subdirectory: $subdir"

    # Get the list of files in this subdir (sorted)
    illumination_to_correct=$(get_file_list "$subdir")
    file_list=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && file_list+=("$line")
    done <<< "$illumination_to_correct"

    # Iterate through each file in the subdir
    for file in "${file_list[@]}"; do
        # Extract file name (basename)
        file_name=$(basename "$file")
        # Get subdirectory name
        subdir_name=$(basename "$subdir")
        # Create output subdirectory
        output_subdir="${OUTPUT_PATH}/${subdir_name}"
        mkdir -p "$output_subdir"

        # Check if output files already exist
        ffp_file="${output_subdir}/${file_name}-ffp.tif"
        dfp_file="${output_subdir}/${file_name}-dfp.tif"
        if [[ -f "$ffp_file" && -f "$dfp_file" ]]; then
            echo "Skipping already processed file: $file_name"
            continue
        fi

        # Process the file
        echo "Processing file: $file_name"
        BaSiC_call "$file" "$output_subdir" "$file_name"
    done
done
