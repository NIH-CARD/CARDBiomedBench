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

# Function to wait for user input before exiting
wait_for_exit() {
    echo "🚪Press any key to exit..."
    read -n 1 -s
    exit 1
}

echo "==================================================================="
stream_message "🔧 Starting CARDBiomedBench Environment Initialization"

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    stream_message "❌ Conda is not installed. Please install Conda first."
    wait_for_exit
fi

# Source conda.sh to allow environment activation in the script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Check if the 'cardbiomedbench-env' environment already exists
if conda env list | grep -q '/conda/envs/cardbiomedbench-env'; then
    stream_message "🔧 The 'cardbiomedbench-env' environment already exists. Skipping creation."
else
    # Create the environment if it does not exist
    stream_message "🔧 Creating the 'cardbiomedbench-env' environment from scratch..."
        
    if ! conda env create -f environment.yml; then
        stream_message "❌ Environment creation failed!"
        wait_for_exit
    fi
fi

# Activating the environment
stream_message "🔧 Activating the 'cardbiomedbench-env' environment..."
if ! conda activate cardbiomedbench-env; then
    stream_message "❌ Activation failed!"
    wait_for_exit
fi

# Final message using the streaming effect
stream_message "🔧 Setup complete, your environment is ready to use!"
echo "==================================================================="