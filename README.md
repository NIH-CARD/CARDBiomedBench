# CARDBiomedBench

A repository for the CARDBiomedBench Q/A set (paper_link).

## Run hands free

### Setup Environment

Run the setup script to create and activate the `cardbiomedbench-env` environment with necessary dependencies:

   ```bash
   source scripts/setup_conda_env.sh
   ```

### Setup Benchmark

Run the setup script to create required directories, configure environment variables, load the benchmark, and prepare for running evaluations:

   ```bash
   python scripts/setup_benchmark_files.py
   ```

### Run Benchmark

Run the benchmark script to collect responses, grade them according to the metrics, and create the graphics.

   ```bash
   python scripts/run_benchmark.py --run_responses --run_metrics --run_graphs
   ```

## Run With Slurm Cluster 

Run each of the scripts in the slurm commands one at a time, one for each model with the corresponding models




# CARDBiomedBench

CARDBiomedBench is a benchmarking suite for evaluating Large Language Models (LLMs) on complex biomedical question-answering tasks.

For detailed methodology and results, please refer to our [paper](#).

## Setup Environment

Create a Conda environment with the necessary dependencies:

   ```bash
   bash scripts/setup_conda_env.sh
   ```

## Setup Benchmark

Prepare directories, configure environment variables, and download the dataset:

   ```bash
   python scripts/setup_benchmark_files.py
   ```

## Run Benchmark

### Hands-Free Execution

Run the benchmark end-to-end:

   ```bash
   python scripts/run_benchmark.py --run_responses --run_metrics --run_graphs
   ```

### Running with Slurm Cluster

If using a Slurm cluster, submit jobs for each model:

## Customizing the Configuration

Modify configs/default_config.yaml to adjust settings:

* Dataset Settings: Change dataset split or name.
* Prompts: Customize system or grading prompts.
* Model Parameters: Adjust max_tokens, temperature, etc.
* Paths: Modify directories for data and outputs.
* Models: Add or remove models.
* Metrics: Enable or disable evaluation metrics.

## Project Structure

* configs/: Configuration files and environment variables.
* data/: Dataset files.
* results/: Output results.
* logs/: Log files.
* scripts/: Scripts for setup, execution, and utilities.