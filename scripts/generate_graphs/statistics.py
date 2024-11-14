import pandas as pd

def statistics_txt(data: pd.DataFrame, models: list, title: str, save_path: str):
    """Write statistics to a text file for each model."""
    
    # Construct the full file path with title
    full_save_path = f"{save_path}{title}.txt"
    
    # Open the file for writing
    with open(full_save_path, 'w') as file:
        # Write the title
        file.write(f"{title}\n")
        file.write("=" * len(title) + "\n")
        
        # Iterate over each question, answer, and model response column and write the token statistics
        for col in ['question', 'answer']:
            token_col = f'{col}_token_count'
            sum_token_count = data[token_col].sum()
            median_token_count = data[token_col].median()
            file.write(f"Sum token count for {col}: {sum_token_count}\n")
            file.write(f"Median token count for {col}: {median_token_count}\n")
        for model in models:
            token_col = f'{model}_response_token_count'
            sum_token_count = data[token_col].sum()
            median_token_count = data[token_col].median()
            file.write(f"Sum token count for {model} responses: {sum_token_count}\n")
            file.write(f"Median token count for {model} responses: {median_token_count}\n")
