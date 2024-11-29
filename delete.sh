#!/bin/bash

# Set the paths for the directories
response_dir="response"
processed_images_dir="processed_images"

# Loop through each .txt file in the response directory
for txt_file in "$response_dir"/*.txt; do
    # Extract the base name of the .txt file (without extension)
    base_name=$(basename "$txt_file" .txt)

    # Construct the corresponding .jpeg file name
    jpeg_file="$processed_images_dir/$base_name.jpeg"

    # Check if the .jpeg file exists and delete it if it does
    if [ -f "$jpeg_file" ]; then
        rm "$jpeg_file"
        echo "Deleted $jpeg_file"
    fi
done
