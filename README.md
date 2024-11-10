# CARDBiomedBench

### Setup Environment

Run the setup script to create and activate the `cardbiomedbench-env` environment with necessary dependencies

   ```bash
   source scripts/setup_conda_env.sh
   ```

### Configure API Keys

Duplicate the .env.example file, rename it to .env, and add your API keys as instructed in the file.
   ```bash
   cp configs/.env.example configs/.env
   ```

### Setup Benchmark

Run the setup script to create directories, load the benchmark, and prep for running evaluation

   ```bash
   python scripts/setup_benchmark_files.py
   ```