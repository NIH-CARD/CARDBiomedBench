import pandas as pd
from tqdm import tqdm
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_model_response(response: str) -> bool:
    ''''''
    if "Error in" not in response:
        return response
    else:
        return None

def query_model_retries(query: str, query_instance: object, query_checker: Callable[[str], bool], retries: int) -> str:
    """
    Query the model with retries in case of failure.

    Parameters:
    - query (str): The input query string.
    - query_instance (object): The model query instance.
    - retries (int): Number of retries in case of failure.

    Returns:
    - str: The response from the model or an error message.
    """
    retry_count = 0
    while retry_count < retries:
        response = query_instance.query(query)
        response = query_checker(response)
        if response:
            return response
        else:
            retry_count += 1
            print(f"Error querying model. Retry {retry_count}/{retries}")
    return f"ERROR: Failed after {retries} retries."

def collect_model_responses(model: str, queries: list, query_checker: Callable[[str], bool], model_dict: dict, max_workers: int, retries: int) -> list:
    """
    Collect responses from a specific model for a list of queries in parallel.

    Parameters:
    - model (str): The model name.
    - queries (list): List of queries to be processed.
    - model_dict (dict): Dictionary containing model instances.
    - retries (int): Number of retries for each query in case of failure.
    - max_workers (int): Maximum number of threads to use for parallel processing.

    Returns:
    - list: List of responses from the model.
    """
    query_instance = model_dict[model]['query_instance']
    responses = [None] * len(queries)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(query_model_retries, query, query_instance, query_checker, retries): i for i, query in enumerate(queries)}
        for future in tqdm(as_completed(future_to_index), total=len(queries), desc=f"Running queries on {model}"):
            index = future_to_index[future]
            try:
                response = future.result()
            except Exception as e:
                response = f"ERROR: Exception {str(e)}"
            responses[index] = response

    return responses

def get_all_model_responses(data: pd.DataFrame, model_dict: dict, max_workers: int, query_col: str='question', retries: int=3) -> pd.DataFrame:
    """
    Get responses from multiple LLMs for each query in the dataset, with retry on failure.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing queries.
    - model_dict (dict): Dictionary containing model instances.
    - query_col (str): The column name containing the queries.
    - retries (int): Number of retries for each API call in case of failure.
    - max_workers (int): Maximum number of threads to use for parallel processing.

    Returns:
    - pd.DataFrame: The DataFrame with added or updated response columns for each model.
    """
    for model in model_dict:
        data[f'{model}_response'] = ''
    for model in model_dict:
        responses = collect_model_responses(model, data[query_col].tolist(), check_model_response, model_dict, max_workers, retries)
        data[f'{model}_response'] = responses

    return data