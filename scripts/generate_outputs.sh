#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Navigate to the script directory (e.g., ./shell/)
cd "$(dirname "$0")"

# Go up to the project root (adjust if necessary)
cd ..

# Define path to merge_dataset directory
ACC_DIR="./measure_accuracy"

# List of generate scripts to run
scripts=(
    "generate_outputs_large.py"
    "generate_outputs_nano.py"
)

# Loop through and run each one
for script in "${scripts[@]}"; do
    echo "=== Running $script ==="
    python "$ACC_DIR/$script"
    echo "=== Finished $script ==="
    echo
done
