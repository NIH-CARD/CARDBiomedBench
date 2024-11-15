#!/bin/bash
# benchmark_runner.sh
# This script runs the run_benchmark.py script with appropriate arguments.
# It is designed to be submitted as a job script in SLURM.

# Default configuration file
CONFIG_FILE="configs/default_config.yaml"

# Initialize an array to hold additional arguments for the Python command
ARGS=()

# Parse command-line arguments for additional options
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --model)
            ARGS+=("--model" "$2")
            shift 2
            ;;
        --run_responses|--run_metrics|--run_graphs)
            ARGS+=("$1")
            shift
            ;;
        *)
            # Assume any positional argument is a model name
            if [[ "${1:0:2}" != "--" ]]; then
                ARGS+=("--model" "$1")
            else
                ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Build the command to run run_benchmark.py
CMD="python3 scripts/run_benchmark.py --config \"$CONFIG_FILE\" ${ARGS[@]}"

# Print the command for debugging
echo "Running command: $CMD"

# Run the command
eval $CMD