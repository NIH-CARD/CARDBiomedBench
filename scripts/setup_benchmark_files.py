import argparse
import os
import sys
import yaml
from dotenv import load_dotenv
from datasets import load_dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description='Setup Benchmark Files for CARDBiomedBench')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to the configuration file')
    return parser.parse_args()

def load_configuration(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except Exception as e:
        print(f"Error loading configuration file: {e}")
    return {}

def download_dataset(config):
    dataset_name = config['dataset'].get('dataset_name', 'CARDBiomedBench')  # Replace with actual name
    dataset_version = config['dataset'].get('dataset_version', 'latest')
    save_path = config['paths'].get('dataset_directory', 'data/')
    
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Downloading dataset '{dataset_name}' version '{dataset_version}'...")
    dataset = load_dataset(dataset_name, name=dataset_version)
    dataset.save_to_disk(save_path)
    print(f"Dataset saved to '{save_path}'")

def check_api_keys(config):
    models = [model for model in config['models'] if model['use']]
    missing_keys = []
    
    for model in models:
        model_type = model.get('type')
        if model_type == 'openai':
            if not os.getenv('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
        elif model_type == 'google':
            if not os.getenv('GOOGLE_API_KEY'):
                missing_keys.append('GOOGLE_API_KEY')
        elif model_type == 'anthropic':
            if not os.getenv('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
        elif model_type == 'huggingface':
            if not os.getenv('HF_TOKEN'):
                missing_keys.append('HF_TOKEN')
    
    if missing_keys:
        print("Missing API keys for the following services:")
        for key in set(missing_keys):
            print(f"- {key}")
        print("\nPlease set these API keys in your environment variables or .env file.")
        sys.exit(1)
    else:
        print("All necessary API keys are present.")

def setup_directories(config):
    directories = {
        'output_directory': config['paths'].get('output_directory', 'results/'),
        'logs_directory': config['paths'].get('logs_directory', 'logs/'),
        'cache_directory': config['paths'].get('cache_directory', '.cache/'),
        'dataset_directory': config['paths'].get('dataset_directory', 'data/'),
    }
    
    created_dirs = []
    existing_dirs = []

    for dir_name, dir_path in directories.items():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            created_dirs.append(dir_path)
        else:
            existing_dirs.append(dir_path)
    
    if created_dirs:
        print("Created directories:")
        for dir_path in created_dirs:
            print(f"- {dir_path}")
    if existing_dirs:
        print("Directories already exist:")
        for dir_path in existing_dirs:
            print(f"- {dir_path}")

def main():
    # Load environment variables from .env
    load_dotenv()
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_configuration(args.config)
    
    # Setup directories
    setup_directories(config)
    
    # # Download dataset
    # download_dataset(config)
    
    # Check API keys
    check_api_keys(config)
    
    # print("\nSetup complete. You can now run the benchmark.")

if __name__ == '__main__':
    main()