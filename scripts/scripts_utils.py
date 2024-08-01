import os
import yaml
import pandas as pd

def load_config():
    """
    Load the configuration from the config.yaml file.

    Returns:
    - dict: Configuration dictionary.
    """
    try:
        config_path = os.path.join(os.path.dirname(__file__), '../configs/config.yaml')
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

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file using pandas.

    Parameters:
    - filepath (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    try:
        dataset = pd.read_csv(filepath)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

def save_dataset(filepath: str, responses: pd.DataFrame):
    """
    Save responses to a CSV file using pandas.

    Parameters:
    - filepath (str): The path to the output CSV file.
    - responses (pd.DataFrame): DataFrame containing the responses.
    """
    try:
        responses.to_csv(filepath, index=False)
    except Exception as e:
        print(f"Error saving responses: {e}")