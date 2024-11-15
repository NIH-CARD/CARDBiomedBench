# CARDBiomedBench 🧬

CARDBiomedBench is a benchmarking suite for evaluating Large Language Models on complex biomedical question-answering tasks. For detailed methodology and results, please refer to our paper [CARDBiomedBench: Benchmarking Large Language Model Performance Gaps in Biomedical Research](#). The CARDBiomedBench dataset is [hosted on Hugging Face](https://huggingface.co/datasets/NIH-CARD/CARDBiomedBench)🤗.

## Setup Environment 

Create an environment with the necessary dependencies in the environment.yml file
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

If using a Slurm cluster, submit jobs for each model with example commands specified in the slurm_commands.txt file.

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
