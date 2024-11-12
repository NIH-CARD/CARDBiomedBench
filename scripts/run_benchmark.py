import argparse
import yaml
import os
import sys
from pathlib import Path
from functools import partial

def stream_message(message, delay=0.025):
    """Display a streaming message effect."""
    for char in message:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run CARDBiomedBench Benchmark')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--model', type=str, help='Name of a specific model to run (optional)')
    parser.add_argument('--run_responses', action='store_true', help='Run response generation step')
    parser.add_argument('--run_metrics', action='store_true', help='Run metrics evaluation step')
    parser.add_argument('--run_graphs', action='store_true', help='Run graphs generation step')
    return parser.parse_args()

def load_configuration(config_path):
    """Load the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        stream_message(f"🔧 Loaded configuration from {config_path}")
        return config
    except Exception as e:
        stream_message(f"❌ Error loading configuration file: {e}")
        sys.exit(1)

def run_responses_runner(model_name, qa_path, res_dir, template_flag):
    """Run the responses_runner.py script for a specific model."""
    cmd = [
        'python', '-m', 'scripts.responses_runner',
        '--qa_path', qa_path,
        '--res_dir', res_dir,
        '--template', template_flag,
        '--model_name', model_name
    ]
    stream_message(f"🚀 Starting response generation for model: {model_name}")
    exit_code = os.system(' '.join(cmd))
    if exit_code != 0:
        stream_message(f"❌ Response generation failed for model: {model_name}")
    else:
        stream_message(f"✅ Completed response generation for model: {model_name}")

def run_metrics_runner(res_dir):
    """Run the metrics_runner.py script."""
    cmd = [
        'python', '-m', 'scripts.metrics_runner',
        '--res_dir', res_dir
    ]
    stream_message("🚀 Starting metrics evaluation")
    exit_code = os.system(' '.join(cmd))
    if exit_code != 0:
        stream_message("❌ Metrics evaluation failed")
    else:
        stream_message("✅ Completed metrics evaluation")

def run_graphs_runner(qa_path, res_dir, scored_path, template_flag):
    """Run the graphs_runner.py script."""
    cmd = [
        'python', '-m', 'scripts.graphs_runner',
        '--qa_path', qa_path,
        '--res_dir', res_dir,
        '--scored_path', scored_path,
        '--template', template_flag
    ]
    stream_message("🚀 Starting graphs generation")
    exit_code = os.system(' '.join(cmd))
    if exit_code != 0:
        stream_message("❌ Graphs generation failed")
    else:
        stream_message("✅ Completed graphs generation")

def main():
    args = parse_arguments()
    config = load_configuration(args.config)
    print(args)
    print(config)
    # # Get paths from the config
    # qa_path = config['paths'].get('qa_path')
    # if not qa_path:
    #     # Assuming the benchmark file is in data directory
    #     dataset_directory = config['paths'].get('dataset_directory', './data/')
    #     dataset_name = config['dataset'].get('dataset_name', 'CARDBiomedBench.csv')
    #     qa_path = os.path.join(dataset_directory, dataset_name)
    # res_dir = config['paths'].get('output_directory', './results/')
    # res_by_model_dir = os.path.join(res_dir, 'by_model')
    # template_flag = str(config['dataset'].get('template_flag', False)).lower()

    # # Ensure output directories exist
    # os.makedirs(res_by_model_dir, exist_ok=True)

    # # Determine models to run
    # models_to_run = []
    # for model in config['models']:
    #     if model.get('use', False):
    #         models_to_run.append(model['name'])
    # # If a specific model is specified via command-line, override
    # if args.model:
    #     if args.model in [model['name'] for model in config['models']]:
    #         models_to_run = [args.model]
    #     else:
    #         stream_message(f"❌ Model '{args.model}' not found in configuration.")
    #         sys.exit(1)

    # # Determine if at least one step is selected
    # if not (args.run_responses or args.run_metrics or args.run_graphs):
    #     stream_message("❌ No execution flags provided. Please specify at least one of --run_responses, --run_metrics, --run_graphs.")
    #     sys.exit(1)

    # # Step 1: Run Responses Runner
    # if args.run_responses:
    #     stream_message("🚀 Running response generation step")
    #     for model_name in models_to_run:
    #         run_responses_runner(model_name, qa_path, res_by_model_dir, template_flag)
    #     stream_message("✅ Completed response generation for all models")
    # else:
    #     stream_message("⚠️  Skipping response generation step")

    # # Step 2: Run Metrics Runner
    # if args.run_metrics:
    #     stream_message("🚀 Running metrics evaluation step")
    #     run_metrics_runner(res_by_model_dir)
    #     stream_message("✅ Completed metrics evaluation")
    # else:
    #     stream_message("⚠️  Skipping metrics evaluation step")

    # # Step 3: Run Graphs Runner
    # if args.run_graphs:
    #     stream_message("🚀 Running graphs generation step")
    #     # Determine the compiled results path
    #     if template_flag == 'true':
    #         benchmark_filename = config['paths'].get('template_benchmark_filename', 'CARDBiomedBench.csv')
    #     else:
    #         benchmark_filename = config['paths'].get('benchmark_filename', 'CARDBiomedBench.csv')
    #     compiled_results_filename = os.path.splitext(benchmark_filename)[0] + '_compiled.csv'
    #     compiled_results_path = os.path.join(res_dir, compiled_results_filename)
    #     run_graphs_runner(qa_path, res_dir, compiled_results_path, template_flag)
    #     stream_message("✅ Completed graphs generation")
    # else:
    #     stream_message("⚠️  Skipping graphs generation step")

    # stream_message("🎉 Benchmark run completed successfully!")

if __name__ == '__main__':
    main()
