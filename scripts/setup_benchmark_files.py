"""
setup_benchmark_files.py

This script sets up the necessary directories, checks for required API keys,
prompts for any missing ones, saves them to a .env file, and downloads the
required dataset for the CARDBiomedBench project.
"""

import argparse
import sys
import yaml
import os
import time
import getpass
from pathlib import Path
from dotenv import load_dotenv, set_key
from datasets import load_dataset

# Define the base directory as the parent of the script's directory
BASE_DIR = Path(__file__).resolve().parent.parent

def stream_message(message, delay=0.025):
    """
    Displays a message one character at a time with an optional delay.

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
    Parses command-line arguments, including the config file path.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Setup Benchmark Files for CARDBiomedBench')
    parser.add_argument(
        '--config',
        type=str,
        default=str(BASE_DIR / 'configs' / 'default_config.yaml'),
        help='Path to the configuration file'
    )
    return parser.parse_args()

def load_configuration(config_path):
    """
    Loads configuration from a YAML file.

    Args:
        config_path (Path): Path to the configuration YAML file.

    Returns:
        dict: Loaded configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        stream_message(f"🔧 Loaded configuration from {config_path.relative_to(BASE_DIR)}")
        return config
    except FileNotFoundError:
        stream_message(f"❌ Configuration file not found at {config_path.relative_to(BASE_DIR)}")
        sys.exit(1)
    except yaml.YAMLError as e:
        stream_message(f"❌ Error parsing YAML file: {e}")
        sys.exit(1)
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

def setup_directories(config):
    """
    Sets up directories specified in the configuration file.

    Args:
        config (dict): Configuration dictionary.
    """
    directories = {
        'output_directory': BASE_DIR / config['paths'].get('output_directory', 'results'),
        'by_model_directory': BASE_DIR / config['paths'].get('output_directory', 'results') / 'by_model',
        'logs_directory': BASE_DIR / config['paths'].get('logs_directory', 'logs'),
        'cache_directory': BASE_DIR / config['paths'].get('cache_directory', '.cache'),
        'dataset_directory': BASE_DIR / config['paths'].get('dataset_directory', 'data'),
    }

    created_dirs = []
    existing_dirs = []

    for dir_name, dir_path in directories.items():
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(f"{dir_path.relative_to(BASE_DIR)}")
        else:
            existing_dirs.append(f"{dir_path.relative_to(BASE_DIR)}")

    if created_dirs:
        stream_message("🔧 Created directories:")
        for dir_path in created_dirs:
            stream_message(f"     * {dir_path}")
    if existing_dirs:
        stream_message("🔧 Directories already exist:")
        for dir_path in existing_dirs:
            stream_message(f"     * {dir_path}")

def check_api_keys(config, dotenv_path):
    """
    Checks for required API keys, prompts for missing ones, and saves them to .env.

    Args:
        config (dict): Configuration dictionary.
        dotenv_path (Path): Path to the .env file.
    """
    models = [model for model in config['models'] if model.get('use')]
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
        stream_message("❌ Missing API keys for the following services:")
        for key in missing_keys:
            stream_message(f"     * {key}")
        stream_message("Please enter the missing API keys.")

        for key in missing_keys:
            api_key = getpass.getpass(prompt=f"🔑 Enter your {key}: ")
            # Set the key in both environment and the .env file
            os.environ[key] = api_key
            set_key(dotenv_path, key, api_key)
        stream_message("🔧 Missing API keys have been added to the .env file.")
    else:
        stream_message("🔧 All necessary API keys are present.")

def create_env_file(config):
    """
    Creates a .env file by prompting the user for required API keys.

    Args:
        config (dict): Configuration dictionary.
    """
    models = [model for model in config['models'] if model.get('use')]
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

    env_vars = {}
    for key in required_keys:
        api_key = getpass.getpass(prompt=f"🔑 Please enter your {key}: ")
        env_vars[key] = api_key

    # Write the keys into the .env file
    dotenv_path = BASE_DIR / 'configs' / '.env'
    with open(dotenv_path, 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")

    stream_message(f"🔧 Created .env file at {dotenv_path.relative_to(BASE_DIR)}")

def download_dataset(config):
    """
    Downloads the dataset hosted on Hugging Face.

    Args:
        config (dict): Configuration dictionary.
    """
    # Retrieve dataset name and split from config with defaults
    dataset_name = config['dataset'].get('dataset_name', 'NIH-CARD/CARDBiomedBench')
    split_type = config['dataset'].get('split', 'test')

    # Retrieve dataset directory path from config
    save_path = BASE_DIR / config['paths'].get('dataset_directory', 'data')
    save_path.mkdir(parents=True, exist_ok=True)

    # Set the CSV file name based on the split type
    csv_file_name = f'CARDBiomedBench_{split_type}.csv'
    csv_file_path = save_path / csv_file_name

    stream_message(f"🔧 Downloading the '{split_type}' split of dataset '{dataset_name}'...")

    try:
        # Load the specified split of the dataset
        split_dataset = load_dataset(
            dataset_name,
            split=split_type,
            cache_dir=os.environ['HF_DATASETS_CACHE']
        )

        # Save the specified split to CSV
        split_dataset.to_csv(str(csv_file_path))
        stream_message(f"🔧 Saved '{split_type}' split dataset to '{csv_file_path.relative_to(BASE_DIR)}'")

    except ValueError as ve:
        stream_message(f"❌ The dataset '{dataset_name}' does not have a '{split_type}' split: {ve}")
        sys.exit(1)
    except Exception as e:
        stream_message(f"❌ Failed to download the '{split_type}' split of the dataset: {e}")
        sys.exit(1)

def main():
    """
    Main function that orchestrates the setup process.
    """
    print("=" * 75)
    stream_message("🔧 Starting CARDBiomedBench Directory Setup")

    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration
    config_path = Path(args.config)
    config = load_configuration(config_path)

    # Setup environment variables for caching
    setup_environment(config)

    # Load environment variables from .env
    dotenv_path = BASE_DIR / 'configs' / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        stream_message(f"🔧 Loaded environment variables from {dotenv_path.relative_to(BASE_DIR)}")
        # Check for missing API keys if the .env file exists
        check_api_keys(config, dotenv_path)
    else:
        stream_message(f"❌ .env file not found at {dotenv_path.relative_to(BASE_DIR)}")
        stream_message("🔧 Creating .env file and prompting for API keys.")
        create_env_file(config)
        load_dotenv(dotenv_path=dotenv_path)
        stream_message(f"🔧 Loaded environment variables from {dotenv_path.relative_to(BASE_DIR)}")

    # Setup directories
    setup_directories(config)

    # Download dataset to local
    download_dataset(config)

    stream_message("✅ Setup complete. You can now run the benchmark.")
    print("=" * 75)

if __name__ == '__main__':
    main()