#!/bin/bash

echo "==================================================================="
echo "Starting CARDBiomedBench Environment Initialization"

# Source conda.sh to enable conda commands
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Check if the 'cardbiomedbench-env' environment already exists
if conda env list | grep -q '/conda/envs/cardbiomedbench-env'; then
    echo "'cardbiomedbench-env' environment already exists. Skipping creation."
else
    echo "Creating the 'cardbiomedbench-env' environment..."
    conda env create -f environment.yml || { echo "Environment creation failed!"; exit 1; }
fi

# Activate the environment
echo "Activating 'cardbiomedbench-env' environment..."
conda activate cardbiomedbench-env || { echo "Activation failed!"; exit 1; }

# Final message
echo "Setup complete. Your environment is ready to use."
echo "==================================================================="