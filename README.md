# CARDBiomedBench 🧬

CARDBiomedBench is a benchmarking suite for evaluating Large Language Models on complex biomedical question-answering tasks. For detailed methodology and results, please refer to our paper [CARDBiomedBench: Benchmarking Large Language Model Performance Gaps in Biomedical Research](https://www.biorxiv.org/content/10.1101/2025.01.15.633272v1). The CARDBiomedBench dataset is [hosted on Hugging Face](https://huggingface.co/datasets/NIH-CARD/CARDBiomedBench)🤗.

## Setup Environment 

Create a Conda environment with the necessary dependencies:

   ```bash
   source scripts/setup_conda_env.sh
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

## Citing
```BibTex
@article {Bianchi2025.01.15.633272,
	author = {Bianchi, Owen and Willey, Maya and Avarado, Chelsea X and Danek, Benjamin and Khani, Marzieh and Kuznetsov, Nicole and Dadu, Anant and Shah, Syed and Koretsky, Mathew J and Makarious, Mary B and Weller, Cory and Levine, Kristin S and Kim, Sungwon and Jarreau, Paige and Vitale, Dan and Marsan, Elise and Iwaki, Hirotaka and Leonard, Hampton and Bandres-Ciga, Sara and Singleton, Andrew B and Nalls, Mike A. and Mokhtari, Shekoufeh and Khashabi, Daniel and Faghri, Faraz},
	title = {CARDBiomedBench: A Benchmark for Evaluating Large Language Model Performance in Biomedical Research},
	year = {2025},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/01/19/2025.01.15.633272},
	journal = {bioRxiv}
}

```
