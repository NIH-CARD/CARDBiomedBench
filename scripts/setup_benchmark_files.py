import argparse
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv, set_key
from datasets import load_dataset
import os
import time
import getpass

# Define the base directory as the parent of the script's directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Function to display a streaming message effect, one character at a time
def stream_message(message, delay=0.025):
    for char in message:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")

# Parse command-line arguments, including the config file path
def parse_arguments():
    parser = argparse.ArgumentParser(description='Setup Benchmark Files for CARDBiomedBench')
    parser.add_argument('--config', type=str, default=BASE_DIR / 'configs' / 'default_config.yaml',
                        help='Path to the configuration file')
    return parser.parse_args()

# Load configuration from a YAML file
def load_configuration(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        stream_message(f"🔧 Loaded configuration from {BASE_DIR.name}/{config_path.relative_to(BASE_DIR)}")
        return config
    except FileNotFoundError:
        stream_message(f"❌ Configuration file not found at {BASE_DIR.name}/{config_path.relative_to(BASE_DIR)}")
        sys.exit(1)
    except yaml.YAMLError as e:
        stream_message(f"❌ Error parsing YAML file: {e}")
        sys.exit(1)
    except Exception as e:
        stream_message(f"❌ Error loading configuration file: {e}")
        sys.exit(1)

# Setup directories specified in the configuration file
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
            created_dirs.append(f"{BASE_DIR.name}/{dir_path.relative_to(BASE_DIR)}")
        else:
            existing_dirs.append(f"{BASE_DIR.name}/{dir_path.relative_to(BASE_DIR)}")

    if created_dirs:
        stream_message("🔧 Created directories: ")
        for dir_path in created_dirs:
            stream_message(f"     * {dir_path}")
    if existing_dirs:
        stream_message("🔧 Directories already exist:")
        for dir_path in existing_dirs:
            stream_message(f"     * {dir_path}")

# Check for required API keys, prompt for missing ones, and save them to .env
def check_api_keys(config, dotenv_path):
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

# Create a .env file by prompting the user for required API keys
def create_env_file(config):
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
        stream_message(f"🔑 Please enter your {key}:")
        api_key = getpass.getpass(prompt='> ')
        env_vars[key] = api_key

    # Write the keys into the .env file
    dotenv_path = BASE_DIR / 'configs' / '.env'
    with open(dotenv_path, 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")

    stream_message(f"🔧 Created .env file at {BASE_DIR.name}/{dotenv_path.relative_to(BASE_DIR)}")

# Download the dataset hosted on huggface 
def download_dataset(config):
    dataset_name = config['dataset'].get('dataset_name', 'NIH-CARD/CARDBiomedBench')
    save_path = BASE_DIR / config['paths'].get('dataset_directory', 'data')
    save_path.mkdir(parents=True, exist_ok=True)
    
    csv_file_name = 'CARDBiomedBench.csv'
    csv_file_path = save_path / csv_file_name

    stream_message(f"🔧 Downloading dataset '{dataset_name}'...")

    try:
        # Load the dataset
        dataset = load_dataset(dataset_name)
        
        # Since there's only one split, get it directly
        split_name = list(dataset.keys())[0]
        split_dataset = dataset[split_name]
        
        # Save the dataset to CSV
        split_dataset.to_csv(str(csv_file_path))
        stream_message(f"🔧 Saved dataset to '{BASE_DIR.name}/{csv_file_path.relative_to(BASE_DIR)}'")
        
    except Exception as e:
        stream_message(f"❌ Failed to download dataset: {e}")
        sys.exit(1)

# Main setup script
def main():
    print("=============================================================================")
    stream_message("🔧 Starting CARDBiomedBench Directory Setup")
    
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration
    config_path = Path(args.config)
    config = load_configuration(config_path)

    # Load environment variables from .env
    dotenv_path = BASE_DIR / 'configs' / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        stream_message(f"🔧 Loaded environment variables from {BASE_DIR.name}/{dotenv_path.relative_to(BASE_DIR)}")
        # Check for missing API keys if the .env file exists
        check_api_keys(config, dotenv_path)
    else:
        stream_message(f"❌ .env file not found at {BASE_DIR.name}/{dotenv_path.relative_to(BASE_DIR)}")
        stream_message(f"🔧 Creating .env file and prompting for API keys.")
        create_env_file(config)
        load_dotenv(dotenv_path=dotenv_path)
        stream_message(f"🔧 Loaded environment variables from {BASE_DIR.name}/{dotenv_path.relative_to(BASE_DIR)}")

    # Setup directories
    setup_directories(config)

    # Download dataset to local
    download_dataset(config)

    stream_message("✅ Setup complete, you can now run the benchmark.")
    print("=============================================================================")

if __name__ == '__main__':
    main()