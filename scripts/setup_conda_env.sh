#!/bin/bash

#===============================================================================
# setup_conda_env.sh
#
# This script sets up the 'cardbiomedbench-env' Conda environment for the
# CARDBiomedBench project. It checks for Conda installation, creates the
# environment if it doesn't exist, and activates it. Pass --minimal to use
# environment-minimal.yml instead of the pinned environment.yml snapshot.
#===============================================================================

ENVIRONMENT_FILE="environment.yml"
ENVIRONMENT_DESCRIPTION="pinned Linux/HPC snapshot"

if [[ "${1:-}" == "--minimal" ]]; then
    ENVIRONMENT_FILE="environment-minimal.yml"
    ENVIRONMENT_DESCRIPTION="portable minimal environment"
elif [[ -n "${1:-}" ]]; then
    echo "Usage: source scripts/setup_conda_env.sh [--minimal]"
    return 1 2>/dev/null || exit 1
fi

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
    if [ -t 0 ]; then
        echo "🚪 Press any key to exit..."
        read -n 1 -s
    fi
    exit 1
}

echo "============================================================================="
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
if conda env list | awk '{print $1}' | grep -qw '^cardbiomedbench-env$'; then
    stream_message "🔧 The 'cardbiomedbench-env' environment already exists. Skipping creation."
else
    # Create the environment if it does not exist
    stream_message "🔧 Creating the 'cardbiomedbench-env' environment from the ${ENVIRONMENT_DESCRIPTION}..."

    # Check for the selected environment file before creating the environment
    if [ ! -f "$ENVIRONMENT_FILE" ]; then
        stream_message "❌ The '$ENVIRONMENT_FILE' file is missing. Please ensure it is present in the current directory."
        wait_for_exit
    fi

    # Create the environment from the selected specification
    if ! conda env create -f "$ENVIRONMENT_FILE"; then
        stream_message "❌ Environment creation failed!"
        wait_for_exit
    fi
fi

# Activating the environment
stream_message "🔧 Activating the 'cardbiomedbench-env' environment..."
if ! conda activate cardbiomedbench-env; then
    stream_message "❌ Activation failed! Ensure that Conda is properly installed and initialized."
    wait_for_exit
fi

# Final message using the streaming effect
stream_message "✅ Setup complete, your environment is ready to use!"
echo "============================================================================="
