"""
run_benchmark.py

This script orchestrates the execution of the CARDBiomedBench benchmark.
It allows running response generation, metric evaluations, and graph generation
steps for specified models and configurations.
"""

import argparse
import yaml
import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Define the base directory as the parent of the script's directory
BASE_DIR = Path(__file__).resolve().parent.parent

def stream_message(message, delay=0.025):
    """
    Display a streaming message effect.

    Args:
        message (str): The message to display.
        delay (float, optional): Delay between characters. Defaults to 0.025 seconds.
    """
    for char in message:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Run CARDBiomedBench Benchmark')
    parser.add_argument(
        '--config', type=str, default=str(BASE_DIR / 'configs' / 'default_config.yaml'),
        help='Path to the configuration file'
    )
    parser.add_argument('--model', type=str,
        help='Name of a specific model to run (optional)'
    )
    parser.add_argument('--run_responses', action='store_true',
        help='Run response generation step'
    )
    parser.add_argument('--run_metrics', action='store_true',
        help='Run metrics evaluation step'
    )
    parser.add_argument('--run_graphs', action='store_true',
        help='Run graphs generation step'
    )
    return parser.parse_args()

def load_configuration(config_path):
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Loaded configuration dictionary.

    Raises:
        SystemExit: If the configuration file cannot be loaded.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        stream_message(f"🔧 Loaded configuration from {config_path.relative_to(BASE_DIR)}")
        return config
    except Exception as e:
        stream_message(f"❌ Error loading configuration file: {e}")
        sys.exit(1)

def setup_environment(config):
    """
    Sets up environment variables for Hugging Face caches.

    Args:
        config (dict): Configuration dictionary.
    """
    hf_cache_dir = BASE_DIR / config['paths'].get('hf_cache_directory', '.cache/huggingface')
    os.environ['HF_HOME'] = str(hf_cache_dir)
    os.environ['HF_DATASETS_CACHE'] = str(hf_cache_dir / 'datasets')

    # Convert paths to be relative to BASE_DIR
    relative_hf_home = os.path.relpath(hf_cache_dir, start=BASE_DIR)
    relative_hf_datasets_cache = os.path.relpath(hf_cache_dir / 'datasets', start=BASE_DIR)

    stream_message(f"🔧 Set HF_HOME to {relative_hf_home}")
    stream_message(f"🔧 Set HF_DATASETS_CACHE to {relative_hf_datasets_cache}")

def run_responses(args, config):
    """
    Run the response generation step.

    Args:
        args (argparse.Namespace): Command-line arguments.
        config (dict): Configuration dictionary.
    """
    # Extract model hyperparameters
    model_params = config.get('model_params', {})
    system_prompt = config['prompts']['system_prompt'].rstrip()

    # Prepare a dictionary of model hyperparameters
    model_hyperparams = {
        'system_prompt': system_prompt,
        'max_new_tokens': model_params.get('max_tokens', 1024),
        'temperature': model_params.get('temperature', 0.0),
    }

    # Get paths from the config
    dataset_directory = config['paths'].get('dataset_directory', './data/')
    split_type = config['dataset'].get('split', 'test')
    dataset_name = f"CARDBiomedBench_{split_type}.csv"
    qa_path = os.path.abspath(os.path.join(dataset_directory, dataset_name))
    res_dir = config['paths'].get('output_directory', './results/')
    res_by_model_dir = os.path.abspath(os.path.join(res_dir, 'by_model/'))

    # Determine models to run
    models_to_run = [model['name'] for model in config['models'] if model.get('use', False)]

    # If a specific model is specified via command-line, override
    if args.model:
        if args.model in [model['name'] for model in config['models']]:
            models_to_run = [args.model]
        else:
            stream_message(f"❌ Model '{args.model}' not found in configuration.")
            sys.exit(1)

    stream_message("🚀 Running response generation step")
    for model_name in models_to_run:
        model_hyperparams_str = json.dumps(model_hyperparams)
        cmd = [
            'python', '-m', 'scripts.responses_runner',
            '--qa_path', qa_path,
            '--res_by_model_dir', res_by_model_dir,
            '--model_name', model_name,
            '--hyperparams', model_hyperparams_str
        ]
        stream_message(f"🔧 Starting response generation for model: {model_name}")
        try:
            subprocess.run(cmd, check=True)
            stream_message(f"✅ Completed response generation for model: {model_name}")
        except subprocess.CalledProcessError as e:
            stream_message(f"❌ Response generation failed for model: {model_name}")
    stream_message("✅ Completed response generation for all models")

def run_metrics(args, config):
    """
    Run the metrics evaluation step.

    Args:
        args (argparse.Namespace): Command-line arguments.
        config (dict): Configuration dictionary.
    """
    # Extract model hyperparameters
    model_params = config.get('model_params', {})
    bioscore_system_prompt = config['prompts']['bioscore_system_prompt'].rstrip()
    bioscore_grading_prompt = config['prompts']['bioscore_grading_prompt'].rstrip()

    # Prepare a dictionary of model hyperparameters
    model_hyperparams = {
        'system_prompt': bioscore_system_prompt,
        'max_new_tokens': model_params.get('max_tokens', 1024),
        'temperature': model_params.get('temperature', 0.0),
    }

    # Get paths from the config
    res_dir = config['paths'].get('output_directory', './results/')
    res_by_model_dir = os.path.abspath(os.path.join(res_dir, 'by_model/'))

    # Determine which models to grade and which metrics to use
    models_to_grade = [model['name'] for model in config['models'] if model.get('use', False)]
    metrics_to_use = [metric['name'] for metric in config['metrics'] if metric.get('use', False)]

    stream_message("🚀 Running metrics evaluation step")

    # Prepare the command
    cmd = [
        'python', '-m', 'scripts.metrics_runner',
        '--res_by_model_dir', res_by_model_dir,
        '--models_to_grade', *models_to_grade,
        '--metrics_to_use', *metrics_to_use,
        '--hyperparams', json.dumps(model_hyperparams),
        '--bioscore_grading_prompt', bioscore_grading_prompt
    ]

    try:
        subprocess.run(cmd, check=True)
        stream_message("✅ Metric grading completed for all models")
    except subprocess.CalledProcessError as e:
        stream_message(f"❌ Metric grading failed")
        sys.exit(1)

def run_graphs(args, config):
    """
    Run the graphs generation step.

    Args:
        args (argparse.Namespace): Command-line arguments.
        config (dict): Configuration dictionary.
    """
    # Extract the necessary paths from the config
    dataset_directory = config['paths'].get('dataset_directory', './data/')
    split_type = config['dataset'].get('split', 'test')
    dataset_name = f"CARDBiomedBench_{split_type}.csv"
    qa_path = os.path.abspath(os.path.join(dataset_directory, dataset_name))
    res_dir = config['paths'].get('output_directory', './results/')
    scored_path = os.path.abspath(os.path.join(res_dir, f"CARDBiomedBench_{split_type}_compiled.csv"))

    # Determine models to process and metrics to use
    models_to_process = [model['name'] for model in config['models'] if model.get('use', False)]
    metrics_to_use = [metric['name'] for metric in config['metrics'] if metric.get('use', False)]

    stream_message("🚀 Running graphs generation step")
    # Prepare the command
    cmd = [
        'python', '-m', 'scripts.graphs_runner',
        '--qa_path', qa_path,
        '--res_dir', res_dir,
        '--scored_path', scored_path,
        '--models_to_process', *models_to_process,
        '--metrics_to_use', *metrics_to_use
    ]

    try:
        subprocess.run(cmd, check=True)
        stream_message("✅ Graphs generation completed")
    except subprocess.CalledProcessError as e:
        stream_message(f"❌ Graphs generation failed")
        sys.exit(1)

def main():
    """
    Main function to orchestrate the benchmark steps.
    """
    print("=" * 100)
    stream_message("🎆 Benchmarking LLMs on CARDBiomedBench 🎆")
    args = parse_arguments()
    config_path = Path(args.config)
    config = load_configuration(config_path)

    # Setup environment variables for caching
    setup_environment(config)

    # Determine if at least one step is selected
    if not (args.run_responses or args.run_metrics or args.run_graphs):
        stream_message("❌ No execution flags provided. Please specify at least one of --run_responses, --run_metrics, --run_graphs.")
        sys.exit(1)

    if args.run_responses:
        run_responses(args, config)
    else:
        stream_message("⚠️  Skipping response generation step  ⚠️")

    if args.run_metrics:
        run_metrics(args, config)
    else:
        stream_message("⚠️  Skipping metrics evaluation step  ⚠️")

    if args.run_graphs:
        run_graphs(args, config)
    else:
        stream_message("⚠️  Skipping graphs generation step  ⚠️")

    stream_message("🎉 Benchmark run completed successfully! 🎉")
    print("=" * 100)

if __name__ == '__main__':
    main()