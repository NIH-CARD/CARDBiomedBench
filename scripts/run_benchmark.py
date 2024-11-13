import argparse
import yaml
import os
import sys
import json
import time

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

def run_responses(args, config):
    """Run the response generation step."""
    # Extract model_hyperparameters
    model_params = config.get('model_params', {})
    system_prompt = config['prompts']['system_prompt'].rstrip()
    
    # Prepare a dictionary of model_hyperparams
    model_hyperparams = {
        'system_prompt': system_prompt,
        'max_new_tokens': model_params.get('max_tokens', 1024),
        'temperature': model_params.get('temperature', 0.0),
    }
    
    # Get paths from the config
    dataset_directory = config['paths'].get('dataset_directory', './data/')
    split_type = config['dataset'].get('split', 'test')
    dataset_name = f"CARDBiomedBench_{split_type}.csv"
    qa_path = os.path.join(dataset_directory, dataset_name)
    res_dir = config['paths'].get('output_directory', './results/')
    res_by_model_dir = os.path.join(res_dir, 'by_model/')
    
    # Determine models to run
    models_to_run = []
    for model in config['models']:
        if model.get('use', False):
            models_to_run.append(model['name'])
    
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
            '--hyperparams', f"'{model_hyperparams_str}'"
        ]
        stream_message(f"     🔧 Starting response generation for model: {model_name}")
        exit_code = os.system(' '.join(cmd))
        if exit_code != 0:
            stream_message(f"     ❌ Response generation failed for model: {model_name}")
        else:
            stream_message(f"     ✅ Completed response generation for model: {model_name}")
    stream_message("✅ Completed response generation for all models")

def run_metrics(args, config):
    """Run the metrics evaluation step."""
    stream_message("🚀 Running metrics evaluation step")
    # TODO
    stream_message("✅ Completed metrics evaluation")

def run_graphs(args, config):
    """Run the graphs generation step."""
    stream_message("🚀 Running graphs generation step")
    # TODO
    stream_message("✅ Completed graphs generation")

def main():
    print("=" * 100)
    stream_message("🎆 Benchmarking LLMs on CARDBiomedBench 🎆")
    args = parse_arguments()
    config = load_configuration(args.config)
    
    # Determine if at least one step is selected
    if not (args.run_responses or args.run_metrics or args.run_graphs):
        stream_message("❌ No execution flags provided. Please specify at least one of --run_responses, --run_metrics, --run_graphs.")
        sys.exit(1)
    
    if args.run_responses:
        run_responses(args, config)
    else:
        stream_message("⚠️ Skipping response generation step ⚠️")
    
    if args.run_metrics:
        run_metrics(args, config)
    else:
        stream_message("⚠️ Skipping metrics evaluation step ⚠️")
    
    if args.run_graphs:
        run_graphs(args, config)
    else:
        stream_message("⚠️ Skipping graphs generation step ⚠️")
    
    stream_message("🎉 Benchmark run completed successfully! 🎉")
    print("=" * 100)

if __name__ == '__main__':
    main()