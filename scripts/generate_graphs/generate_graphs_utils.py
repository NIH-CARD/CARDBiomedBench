import os
import pandas as pd

def merge_model_responses(res_dir: str, output_csv: str, merge_on: str='uuid', question_col: str='question', answer_col: str='answer') -> pd.DataFrame:
    """
    Merge all individual model response CSV files in a directory into a single DataFrame, merging on a specific column.
    The question and answer columns are included only once in the final DataFrame.

    Parameters:
    - res_dir (str): Directory containing the saved response CSVs.
    - output_csv (str): The file path to save the merged CSV.
    - merge_on (str): The column name to merge on (default is 'uuid').
    - question_col (str): The column name for the question (default is 'question').
    - answer_col (str): The column name for the answer (default is 'answer').

    Returns:
    - pd.DataFrame: The merged DataFrame containing all model responses.
    """

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(res_dir) if f.endswith('_responses.csv')]
    
    # Initialize an empty DataFrame
    merged_df = pd.DataFrame()

    # Iterate over all the CSV files and merge them
    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(res_dir, csv_file)
        model_df = pd.read_csv(file_path)
        if i == 0:
            # On the first merge, include all columns
            merged_df = model_df
        else:
            # On subsequent merges, drop the question and answer columns to avoid duplication
            model_df = model_df.drop(columns=[question_col, answer_col])
            merged_df = pd.merge(merged_df, model_df, on=merge_on, how='outer')

    # Save the final merged DataFrame
    merged_df.to_csv(output_csv, index=False)
    print(f"All responses merged and saved to {output_csv}.")
    return merged_df