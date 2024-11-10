import argparse
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
import os

# Define the base directory as the parent of the script's directory
BASE_DIR = Path(__file__).resolve().parent.parent

def parse_arguments():
    parser = argparse.ArgumentParser(description='Setup Benchmark Files for CARDBiomedBench')
    parser.add_argument('--config', type=str, default=BASE_DIR / 'configs' / 'default_config.yaml',
                        help='Path to the configuration file')
    return parser.parse_args()

def load_configuration(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)

def setup_directories(config):
    directories = {
        'output_directory': BASE_DIR / config['paths'].get('output_directory', 'results'),
        'logs_directory': BASE_DIR / config['paths'].get('logs_directory', 'logs'),
        'cache_directory': BASE_DIR / config['paths'].get('cache_directory', '.cache'),
        'dataset_directory': BASE_DIR / config['paths'].get('dataset_directory', 'data'),
    }

    created_dirs = []
    existing_dirs = []

    for dir_name, dir_path in directories.items():
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(dir_path.relative_to(BASE_DIR))
        else:
            existing_dirs.append(dir_path.relative_to(BASE_DIR))

    if created_dirs:
        print("Created directories:")
        for dir_path in created_dirs:
            print(f"- {dir_path}")
    if existing_dirs:
        print("Directories already exist:")
        for dir_path in existing_dirs:
            print(f"- {dir_path}")

def download_dataset(config):
    dataset_name = config['dataset'].get('dataset_name', 'CARDBiomedBench')
    dataset_version = config['dataset'].get('dataset_version', 'latest')
    save_path = BASE_DIR / config['paths'].get('dataset_directory', 'data')

    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{dataset_name}' version '{dataset_version}'...")
    dataset = load_dataset(dataset_name, name=dataset_version)
    dataset.save_to_disk(str(save_path))
    print(f"Dataset saved to '{save_path}'")

def check_api_keys(config):
    models = [model for model in config['models'] if model['use']]
    required_keys = set()

    for model in models:
        model_type = model.get('type')
        if model_type == 'openai':
            required_keys.add('OPENAI_API_KEY')
        elif model_type == 'anthropic':
            required_keys.add('ANTHROPIC_API_KEY')
        elif model_type == 'google':
            required_keys.add('GOOGLE_API_KEY')
        elif model_type == 'huggingface':
            required_keys.add('HF_TOKEN')
        elif model_type == 'perplexity':
            required_keys.add('PERPLEXITY_API_KEY')

    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print("Missing API keys for the following services:")
        for key in missing_keys:
            print(f"- {key}")
        print("\nPlease set these API keys in your environment variables or .env file.")
        sys.exit(1)
    else:
        print("All necessary API keys are present.")

def main():
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    config_path = Path(args.config)
    config = load_configuration(config_path)

    # Load environment variables from .env
    dotenv_path = BASE_DIR / 'configs' / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded environment variables from {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}")
        print("Proceeding without loading .env file. Ensure that API keys are set as environment variables.")

    # Setup directories
    setup_directories(config)

    # # Download dataset
    # download_dataset(config)

    # Check API keys
    check_api_keys(config)

    print("\nSetup complete. You can now run the benchmark.")

if __name__ == '__main__':
    main()