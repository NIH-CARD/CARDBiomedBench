import os
import tiktoken
import pandas as pd
from scripts import TEMPLATE_SAMPLES
from scripts.scripts_utils import load_dataset, sample_by_template

def merge_model_responses(qa_path: str, res_dir: str, output_csv: str, template_flag: bool, merge_on: str='uuid') -> pd.DataFrame:
    """
    Merge all individual model response CSV files in a directory into a single DataFrame, merging on a specific column.
    The question answer, and category columns are included only once in the final DataFrame.
    """

    merged_df = load_dataset(qa_path)
    if merged_df.empty:
        print("No data to process. Exiting.")
        return
    
    merge_cols = ['uuid', 'question', 'answer', 'SQL_Category', 'Bio_Category']
    if template_flag:
        merged_df = sample_by_template(merged_df, TEMPLATE_SAMPLES)
        merge_cols += ['template uuid']
    merged_df = merged_df[merge_cols]
    merged_df.dropna(inplace=True)

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(res_dir) if f.endswith('_responses.csv')]

    # Iterate over all the CSV files and merge them
    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(res_dir, csv_file)
        model_df = pd.read_csv(file_path)
        model_df = model_df.drop(columns=[col for col in merge_cols if col != merge_on])
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

def count_tokens_tiktoken(string: str, model: str = "gpt-4o") -> int:
    """Returns the number of tokens in a text string using the appropriate encoding for a given model."""
    try:
        # Get the encoding for the specified model
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # If the model is not recognized, default to cl100k_base encoding
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Encode the string and return the number of tokens
    string = str(string) if string is not None else ""
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_token_counts(data: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Add a token_count column for question, answer, and each model_response column in the DataFrame."""
    for col in ['question', 'answer']:
        data[f'{col}_token_count'] = data[col].apply(lambda x: count_tokens_tiktoken(x))
    for model in models:
        col = f'{model}_response'
        data[f'{model}_response_token_count'] = data[col].apply(lambda x: count_tokens_tiktoken(x))
    return data