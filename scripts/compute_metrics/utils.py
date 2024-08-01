import pandas as pd
from scripts.collect_responses.utils import collect_model_responses
from .prompts import biomedical_grading_prompt

def get_all_model_LLMEVAL(data: pd.DataFrame, grading_model: str, model_dict: dict, query_col: str='question', gold_col: str='answer', response_col: str='response', retries: int=3, max_workers: int=10) -> pd.DataFrame:
    """
    Grade responses from multiple LLMs with a specific prompt & GPT-4o for each query in the dataset, with retry on failure.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing questions, gold responses, and model responses.
    - model_dict (dict): Dictionary containing model instances.
    - query_col (str): The column name containing the queries.
    - gold_col (str): The column name containing the gold responses.
    - response_col (str): The column name containing the model responses.
    - retries (int): Number of retries for each API call in case of failure.
    - max_workers (int): Maximum number of threads to use for parallel processing.

    Returns:
    - pd.DataFrame: The DataFrame with added or updated response columns for each model.
    """
    for model in model_dict:
        data[f'{model}_LLMEVAL'] = 0.0

    for model in model_dict:
        grading_prompts = [
            biomedical_grading_prompt(row[query_col], row[gold_col], row[f'{model}_{response_col}'])
            for _, row in data.iterrows()
        ]
        responses = collect_model_responses(grading_model, grading_prompts, model_dict, retries, max_workers)
        data[f'{model}_LLMEVAL'] = responses

    return data