#!/bin/bash

# Define the path to the configuration file
CONFIG_FILE="configs/config.yaml"

# Extract the benchmark_filename value from the YAML file
BENCHMARK_FILE_NAME=$(grep 'benchmark_filename' $CONFIG_FILE | awk '{print $2}')

# Extract run script flags from the YAML file
RUN_RESPONSES=$(grep 'run_responses:' $CONFIG_FILE | awk '{print $2}')
RUN_METRICS=$(grep 'run_metrics:' $CONFIG_FILE | awk '{print $2}')
RUN_GRAPHS=$(grep 'run_graphs:' $CONFIG_FILE | awk '{print $2}')
TEMPLATE_FLAG=$(grep 'run_template:' $CONFIG_FILE | awk '{print $2}')

# Define local directories and paths accordingly
BENCHMARK_DIR="benchmark/"

# Set results directory based on template flag
if [ "$TEMPLATE_FLAG" = "true" ]; then
  RESULTS_DIR="template_results/"
else
  RESULTS_DIR="results/"
fi

RESULTS_BY_MODEL_DIR="${RESULTS_DIR}by_model/"
BENCHMARK_PATH="$BENCHMARK_DIR$BENCHMARK_FILE_NAME"
COMPILED_RES_PATH="${RESULTS_DIR}${BENCHMARK_FILE_NAME%.csv}_compiled.csv"

# Run response_runner.py to collect results from the benchmark file if enabled
if [ "$RUN_RESPONSES" = "true" ]; then
  python3 -m scripts.responses_runner --qa_path $BENCHMARK_PATH --res_dir $RESULTS_BY_MODEL_DIR --template $TEMPLATE_FLAG
fi

# Run metrics_runner.py to score the generated results if enabled
if [ "$RUN_METRICS" = "true" ]; then
  python3 -m scripts.metrics_runner --res_dir $RESULTS_BY_MODEL_DIR
fi

# Run graphs_runner.py to create graphs of the scored results if enabled
if [ "$RUN_GRAPHS" = "true" ]; then
  python3 -m scripts.graphs_runner --qa_path $BENCHMARK_PATH --res_dir $RESULTS_DIR --scored_path $COMPILED_RES_PATH --template $TEMPLATE_FLAG
fi