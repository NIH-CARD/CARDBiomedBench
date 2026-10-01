"""
scripts_utils.py

Utility functions for loading and saving datasets, and for sampling data.

This script can also be run directly to split the CARDBiomedBench dataset into train and test sets.
"""

import os
import argparse
import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file using pandas.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.ParserError: If the CSV cannot be parsed.
    """
    try:
        dataset = pd.read_csv(filepath)
        return dataset
    except Exception as e:
        print(f"Error loading dataset from '{filepath}': {e}")
        return pd.DataFrame()


def save_dataset(filepath: str, data: pd.DataFrame) -> None:
    """
    Save a DataFrame to a CSV file using pandas.

    Args:
        filepath (str): The path to the output CSV file.
        data (pd.DataFrame): DataFrame containing the data to save.

    Raises:
        IOError: If the file cannot be written.
    """
    try:
        data.to_csv(filepath, index=False)
    except Exception as e:
        print(f"Error saving dataset to '{filepath}': {e}")


def sample_by_template(
    data: pd.DataFrame,
    n: int,
    batch_size: int = 10,
    random_state: int = 12
) -> pd.DataFrame:
    """
    Group the data by 'template_uuid' and sample deterministically from each group.
    Ensures that when 'n' increases, all previously sampled rows are included.

    Args:
        data (pd.DataFrame): The input DataFrame containing 'template_uuid'.
        n (int): The total number of samples to select per 'template_uuid'.
        batch_size (int): The size of each sampling batch.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: A new DataFrame with 'n' rows sampled per 'template_uuid'.
    """
    final_sampled_df = pd.DataFrame()

    def deterministic_group_sample(group: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        sampled_group_df = pd.DataFrame()
        n_samples = min(n_samples, len(group))
        remaining_group = group.copy()
        for _ in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - len(sampled_group_df))
            current_sample = remaining_group.sample(n=current_batch_size, random_state=random_state)
            sampled_group_df = pd.concat([sampled_group_df, current_sample])
            remaining_group = remaining_group.drop(current_sample.index)
        return sampled_group_df

    for template_uuid, group in data.groupby('template_uuid'):
        sampled_group = deterministic_group_sample(group, n)
        final_sampled_df = pd.concat([final_sampled_df, sampled_group])

    return final_sampled_df.reset_index(drop=True)


def sample_template_subset(
    data: pd.DataFrame,
    subset_size: int,
    template_col: str = "template_uuid",
    random_state: int = 12,
) -> pd.DataFrame:
    """Select an exact-size, balanced subset containing every template.

    Rows within each template are deterministically shuffled. Slots are then
    allocated round-robin across templates, skipping templates that have no
    rows left. This keeps template counts as even as their available row counts
    permit and makes smaller requested subsets nested within larger ones.
    """
    if template_col not in data.columns:
        raise ValueError(f"Dataset does not contain '{template_col}'")
    if data[template_col].isna().any():
        raise ValueError(f"Dataset contains missing values in '{template_col}'")
    if subset_size <= 0:
        raise ValueError("Subset size must be greater than zero")
    if subset_size > len(data):
        raise ValueError(
            f"Subset size {subset_size} exceeds dataset size {len(data)}"
        )

    grouped_indices = {
        template: group.sample(frac=1, random_state=random_state).index.tolist()
        for template, group in data.groupby(template_col, sort=True)
    }
    template_count = len(grouped_indices)
    if subset_size < template_count:
        raise ValueError(
            f"Subset size must be at least {template_count} to include every "
            f"'{template_col}' (received {subset_size})"
        )

    selected_indices = []
    offsets = {template: 0 for template in grouped_indices}
    while len(selected_indices) < subset_size:
        for template, indices in grouped_indices.items():
            offset = offsets[template]
            if offset < len(indices):
                selected_indices.append(indices[offset])
                offsets[template] += 1
                if len(selected_indices) == subset_size:
                    break

    # Preserve source order so downstream output remains easy to compare with
    # the full benchmark dataset.
    return data.loc[sorted(selected_indices)].reset_index(drop=True)


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
