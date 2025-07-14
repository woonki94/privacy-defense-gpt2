#!/bin/bash

# Exit on error
set -e

# Navigate to the script directory (shell/)
cd "$(dirname "$0")"

# Go up to the project root
cd ..

# Define path to preprocess directory
PREPROCESS_DIR="./preprocess"

# List of scripts to run
scripts=(
    "enron_preprocessing.py"
    "openwebtext_preprocessing.py"
    "PII_preprocessing.py"
    "wiki103_preprocessing.py"
)

# Loop through and run each one
for script in "${scripts[@]}"; do
    echo "=== Running $script ==="
    python "$PREPROCESS_DIR/$script"
    echo "=== Finished $script ==="
    echo
done

