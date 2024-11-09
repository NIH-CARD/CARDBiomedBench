#!/bin/bash

# Function to display a streaming message effect
stream_message() {
    local message="$1"
    local delay="${2:-0.025}"
    for ((i=0; i<${#message}; i++)); do
        echo -n "${message:$i:1}"
        sleep "$delay"
    done
    echo
}

echo "==================================================================="
stream_message "🔧 Starting CARDBiomedBench Environment Initialization"

# Source conda.sh to allow environment activation in the script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Check if the 'cardbiomedbench-env' environment already exists
if conda env list | grep -q '/conda/envs/cardbiomedbench-env'; then
    stream_message "🔧 The 'cardbiomedbench-env' environment already exists. Skipping creation."
else
    # Create the environment if it does not exist
    stream_message "🔧 Creating the 'cardbiomedbench-env' environment from scratch..."
        
    conda env create -f environment.yml || { stream_message "❌ Environment creation failed!"; exit 1; }
fi

# Activating the environment with a loading spinner
stream_message "🔧 Activating the 'cardbiomedbench-env' environment..."

# Activate the environment
conda activate cardbiomedbench-env || { echo "❌ Activation failed!"; exit 1; }

# Final message using the streaming effect
stream_message "🔧 Setup complete, your environment is ready to use!"
echo "==================================================================="