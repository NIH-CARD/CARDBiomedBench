#!/bin/bash

# Define the path to the configuration file
CONFIG_FILE="configs/config.yaml"

# Extract the benchmark_filename value from the YAML file
BENCHMARK_FILE_NAME=$(grep 'benchmark_filename' $CONFIG_FILE | awk '{print $2}')

# Set res, scored, and stats filenames accordingly
RES_FILE_NAME="${BENCHMARK_FILE_NAME%.csv}_res.csv"
SCORED_FILE_NAME="${BENCHMARK_FILE_NAME%.csv}_scored.csv"
METRICS_FILE_NAME="${BENCHMARK_FILE_NAME%.csv}_statistics.json"

# Define local directories and paths accordingly
RESULTS_DIR="results/"
BENCHMARK_DIR="benchmark/"
LOCAL_BENCHMARK_PATH="$BENCHMARK_DIR$BENCHMARK_FILE_NAME"
LOCAL_RES_PATH="$RESULTS_DIR$RES_FILE_NAME"
LOCAL_SCORED_PATH="$RESULTS_DIR$SCORED_FILE_NAME"

# Run response_runner.py to collect results from the benchmark file
python3 -m scripts.responses_runner --local_qa_path $LOCAL_BENCHMARK_PATH --local_res_path $LOCAL_RES_PATH

# Run metrics_runner.py to score the generated results
python3 -m scripts.metrics_runner --local_res_path $LOCAL_RES_PATH --local_scored_path $LOCAL_SCORED_PATH

# Run graphs_runner.py to create graphs of the scored results
python3 -m scripts.graphs_runner --local_scored_path $LOCAL_SCORED_PATH