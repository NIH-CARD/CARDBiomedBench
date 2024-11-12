# CARDBiomedBench

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

   ```bash
   python scripts/run_benchmark.py --run_responses --run_metrics --run_graphs
   ```