#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Navigate to the script directory (e.g., ./shell/)
cd "$(dirname "$0")"

# Go up to the project root (adjust if necessary)
cd ..

# Define path to merge_dataset directory
MERGE_DIR="./preprocess"

# List of merge scripts to run
scripts=(
    "merge_dataset_large.py"
    "merge_dataset_nano.py"
)

# Loop through and run each one
for script in "${scripts[@]}"; do
    echo "=== Running $script ==="
    python "$MERGE_DIR/$script"
    echo "=== Finished $script ==="
    echo
done

