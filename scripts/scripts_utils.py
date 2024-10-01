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

def save_dataset(filepath: str, data: pd.DataFrame):
    """
    Save responses to a CSV file using pandas.

    Parameters:
    - filepath (str): The path to the output CSV file.
    - responses (pd.DataFrame): DataFrame containing the responses.
    """
    try:
        data.to_csv(filepath, index=False)
    except Exception as e:
        print(f"Error saving responses: {e}")

def sample_by_template(data: pd.DataFrame, n: int, batch_size: int = 10, random_state: int = 12) -> pd.DataFrame:
    """
    Groups the data by 'template uuid' and samples deterministically from each group.
    Ensures that when n increases, all previously sampled rows are included.
    
    Parameters:
    - data (pd.DataFrame): The input dataframe containing 'template uuid'.
    - n (int): The total number of samples to select per 'template uuid'.
    - batch_size (int): The size of each sampling batch.
    - random_state (int): Seed for reproducibility.
    
    Returns:
    - pd.DataFrame: A new dataframe with `n` rows sampled per 'template uuid'.
    """
    final_sampled_df = pd.DataFrame()

    def deterministic_group_sample(group, n):
        sampled_group_df = pd.DataFrame()
        n = min(n, len(group))
        for _ in range(0, n, batch_size):
            current_batch_size = min(batch_size, n - len(sampled_group_df))
            current_sample = group.sample(n=current_batch_size, random_state=random_state)
            sampled_group_df = pd.concat([sampled_group_df, current_sample])
            group = group.drop(current_sample.index)

        return sampled_group_df

    for template_uuid, group in data.groupby('template uuid'):
        final_sampled_df = pd.concat([final_sampled_df, deterministic_group_sample(group, n)])

    return final_sampled_df.reset_index(drop=True)
    
# def sample_data_by_template(data: pd.DataFrame, n: int) -> pd.DataFrame:
#     """
#     Groups the data by 'template uuid' and samples `n` rows from each group.
#     Throws a ValueError if a group has fewer than `n` rows.
    
#     Parameters:
#     - data (pd.DataFrame): The input dataframe containing 'template uuid'.
#     - n (int): The number of samples to select from each group.
    
#     Returns:
#     - pd.DataFrame: A new dataframe with `n` rows sampled per 'template uuid'.
    
#     Raises:
#     - ValueError: If any group has fewer than `n` rows.
#     """
#     def check_and_sample(group):
#         if len(group) < n:
#             raise ValueError(f"Group {group['template uuid'].iloc[0]} has fewer than {n} rows.")
#         return group.sample(n=n, random_state=12)

#     return data.groupby('template uuid').apply(check_and_sample).reset_index(drop=True)