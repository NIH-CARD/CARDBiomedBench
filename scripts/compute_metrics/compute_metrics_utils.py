import pandas as pd
import re
from .prompts import biomedical_grading_prompt
from scripts.scripts_utils import load_dataset, save_dataset
from scripts.collect_responses.collect_responses_utils import collect_model_responses, initialize_model

def check_LLMEVAL_response(response: str) -> tuple:
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
        if number in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            return number, True
    return None, False

def get_all_model_LLMEVAL(res_dir: str, grading_model: str, model_dict: dict, max_workers: int, query_col: str='question', gold_col: str='answer', response_col: str='response', retries: int=3, initial_delay: int=1) -> pd.DataFrame:
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
        data = load_dataset(f'{res_dir}/{model}_responses.csv')
        data[f'{model}_LLMEVAL'] = 0.0
        grading_prompts = [
            biomedical_grading_prompt(row[query_col], row[gold_col], row[f'{model}_{response_col}'])
            for _, row in data.iterrows()
        ]
        query_instance = initialize_model(grading_model)
        responses = collect_model_responses(grading_model, query_instance, grading_prompts, check_LLMEVAL_response, max_workers, retries, initial_delay)
        query_instance.delete()
        data[f'{model}_LLMEVAL'] = responses
        save_dataset(f'{res_dir}/{model}_responses.csv', data)