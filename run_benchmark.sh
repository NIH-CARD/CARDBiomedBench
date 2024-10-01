#!/bin/bash

# Define the path to the configuration file
CONFIG_FILE="configs/config.yaml"

# Extract the benchmark_filename value from the YAML file
MANUAL_BENCHMARK_FILE_NAME=$(grep 'benchmark_filename' $CONFIG_FILE | awk '{print $2}')
TEMPLATE_BENCHMARK_FILE_NAME=$(grep 'template_benchmark_filename' $CONFIG_FILE | awk '{print $2}')

# Extract run script flags from the YAML file
RUN_RESPONSES=$(grep 'run_responses:' $CONFIG_FILE | awk '{print $2}')
RUN_METRICS=$(grep 'run_metrics:' $CONFIG_FILE | awk '{print $2}')
RUN_GRAPHS=$(grep 'run_graphs:' $CONFIG_FILE | awk '{print $2}')
TEMPLATE_FLAG=$(grep 'run_template:' $CONFIG_FILE | awk '{print $2}')

# Set results directory based on template flag
if [ "$TEMPLATE_FLAG" = "true" ]; then
  BENCHMARK_PATH="benchmark/$TEMPLATE_BENCHMARK_FILE_NAME"
  RESULTS_PATH="results/template_results/"
  COMPILED_RES_PATH="${RESULTS_PATH}${TEMPLATE_BENCHMARK_FILE_NAME%.csv}_compiled.csv"
  RESULTS_BY_MODEL_DIR="${RESULTS_PATH}by_model/"
else
  BENCHMARK_PATH="benchmark/$MANUAL_BENCHMARK_FILE_NAME"
  RESULTS_PATH="results/manual_results/"
  COMPILED_RES_PATH="${RESULTS_PATH}${MANUAL_BENCHMARK_FILE_NAME%.csv}_compiled.csv"
  RESULTS_BY_MODEL_DIR="${RESULTS_PATH}by_model/"
fi

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
  python3 -m scripts.graphs_runner --qa_path $BENCHMARK_PATH --res_dir $RESULTS_PATH --scored_path $COMPILED_RES_PATH --template $TEMPLATE_FLAG
fi