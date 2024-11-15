#!/bin/bash

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
      shift 1
      ;;
    *)
      ARGS+=("$1")
      shift 1
      ;;
  esac
done

# If a model name is provided directly without --model flag, add it as a --model argument
if [[ -n "${ARGS[0]}" && "${ARGS[0]}" != --* ]]; then
  ARGS=("--model" "${ARGS[0]}" "${ARGS[@]:1}")
fi

# Build the command to run run_benchmark.py
CMD="python3 scripts/run_benchmark.py --config $CONFIG_FILE ${ARGS[@]}"

# Print the command for debugging
echo "Running command: $CMD"

# Run the command
eval $CMD