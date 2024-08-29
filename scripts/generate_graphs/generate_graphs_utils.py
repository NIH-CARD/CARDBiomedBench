import os
import pandas as pd

def merge_model_responses(res_dir: str, output_csv: str, merge_on: str='uuid', question_col: str='question', answer_col: str='answer', category_col: str='bio_category') -> pd.DataFrame:
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
            model_df = model_df.drop(columns=[question_col, answer_col, category_col])
            merged_df = pd.merge(merged_df, model_df, on=merge_on, how='outer')

    # Save the final merged DataFrame
    merged_df.to_csv(output_csv, index=False)
    print(f"All responses merged and saved to {output_csv}.")
    return merged_df

def get_model_order(data: pd.DataFrame, metric: str, models: dict) -> list:
    """Get the order of models based on the median first, then IDK %, then spread (IQR) of the metric values."""
    model_stats = []
    
    for model in models:
        col_name = f'{model}_{metric}'
        if col_name in data.columns:
            model_data = data[col_name].copy()
            idk_count = (model_data == -1).sum()
            total_count = len(model_data)
            idk_rate = idk_count / total_count
            
            if metric == "BioScore":
                model_data = model_data[model_data != -1]  # Exclude -1 values for BioScore
            
            median_val = model_data.median()
            spread_val = model_data.quantile(0.75) - model_data.quantile(0.25)  # IQR for spread
            
            model_stats.append((model, median_val, idk_rate, spread_val))
    
    # Sort by median (descending), then by IDK rate (ascending), then by spread (ascending)
    model_stats_sorted = sorted(model_stats, key=lambda x: (-x[1], x[2], x[3]))

    # Extract the sorted model names
    sorted_models = [model[0] for model in model_stats_sorted]
    
    return sorted_models