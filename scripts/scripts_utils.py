import os
import pandas as pd

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

    for template_uuid, group in data.groupby('template_uuid'):
        final_sampled_df = pd.concat([final_sampled_df, deterministic_group_sample(group, n)])

    return final_sampled_df.reset_index(drop=True)

if __name__ == "__main__":
    # Load the original data
    orig = pd.read_csv("data/CARDBiomedBench.csv")
    print(f"Original Data: {len(orig)}")
    print(orig.head())

    # Sample the data
    sampled = sample_by_template(orig, 270)
    print(f"\nSampled Data (Test Set): {len(sampled)}")
    print(len(sampled))

    # Identify the train set by excluding sampled indices
    train = orig.loc[~orig.index.isin(sampled.index)]
    print(f"\nTrain Data: {len(train)}")
    print(train.head())

    # Save the splits to CSV files
    sampled.to_csv("data/test.csv", index=False)
    train.to_csv("data/train.csv", index=False)

    print("\nData has been successfully split into 'test.csv' and 'train.csv'.")