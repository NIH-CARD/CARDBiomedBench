#!/bin/bash

# Define the path to the configuration file
CONFIG_FILE="configs/config.yaml"

# Extract the benchmark_filename value from the YAML file
BENCHMARK_FILE_NAME=$(grep 'benchmark_filename' $CONFIG_FILE | awk '{print $2}')

# Define local directories and paths accordingly
BENCHMARK_DIR="benchmark/"
RESULTS_DIR="results/"
RESULTS_BY_MODEL_DIR="${RESULTS_DIR}by_model/"
BENCHMARK_PATH="$BENCHMARK_DIR$BENCHMARK_FILE_NAME"
COMPILED_RES_PATH="${RESULTS_DIR}${BENCHMARK_FILE_NAME%.csv}_compiled.csv"

# Run response_runner.py to collect results from the benchmark file
python3 -m scripts.responses_runner --qa_path $BENCHMARK_PATH --res_dir $RESULTS_BY_MODEL_DIR

# Run metrics_runner.py to score the generated results
python3 -m scripts.metrics_runner --res_dir $RESULTS_BY_MODEL_DIR

# Run graphs_runner.py to create graphs of the scored results
python3 -m scripts.graphs_runner --res_dir $RESULTS_BY_MODEL_DIR --scored_path $COMPILED_RES_PATH