import pandas as pd
import re
from .prompts import biomedical_grading_prompt
from scripts.collect_responses.collect_responses_utils import collect_model_responses

def check_LLMEVAL_response(response: str) -> float:
    """
    Check the LLMEVAL evaluation response for a valid score.

    Parameters:
    - response (str): The response from LLMEVAL evaluation.

    Returns:
    - float: The valid score extracted from the response, or None if no valid score is found.
    """
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", response)
    if match:
        number = float(match.group(0))
        if number in [-1, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            return number
    return None

def get_all_model_LLMEVAL(data: pd.DataFrame, grading_model: str, model_dict: dict, max_workers: int, query_col: str='question', gold_col: str='answer', response_col: str='response', retries: int=3) -> pd.DataFrame:
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
        responses = collect_model_responses(grading_model, grading_prompts, check_LLMEVAL_response, model_dict, max_workers, retries)
        data[f'{model}_LLMEVAL'] = responses

    return data