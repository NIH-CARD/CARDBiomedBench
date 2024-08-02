## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/NIH-CARD/card-benchmark.git
    cd card-benchmark
    ```

2. Modify the configuration files:
* Rename the example.env file to .env and populate it with your API keys
* Set the values accordingly in the config.yaml to which models, metrics, etc you would like to run

3. Run the setup script to create and activate the virtual environment and execute the benchmark workflow:
    ```bash
    bash run_benchmark.sh
    ```

## Workflow

The setup script will:
1. Create and activate a virtual environment named `eval-virtual-env`.
2. Install all required packages from `requirements.txt`.
3. Run `response_runner.py` to collect results from the benchmark file.
4. Run `metrics_runner.py` to score the generated results.
5. Run `graphs_runner.py` to create graphs of the scored results.