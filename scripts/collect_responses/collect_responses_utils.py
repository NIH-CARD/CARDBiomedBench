import time
import pandas as pd
from tqdm import tqdm
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts import SYSTEM_PROMPT, MODELS_DICT
from .gpt4o_query import GPT4OQuery
from .gemini_query import GeminiQuery
from .huggingface_query import HuggingFaceQuery

def initialize_model(model: str):
    """
    Initialize the model client and create an instance of the query class for the specified model.
    """
    if model == 'gpt-4o':
        return GPT4OQuery(SYSTEM_PROMPT)
    elif model == 'gemini-1.5-pro':
        return GeminiQuery(SYSTEM_PROMPT)
    elif model == 'gemma-2-2b-it':
        return HuggingFaceQuery(SYSTEM_PROMPT, 'google/gemma-2-2b-it')   
    else:
        return None

def delete_model(model: str):
    """
    Delete the model instance and release any resources.
    """
    if model in MODELS_DICT and MODELS_DICT[model]['query_instance'] is not None:
        query_instance = MODELS_DICT[model]['query_instance']
        query_instance.delete()
        MODELS_DICT[model]['query_instance'] = None
        print(f"{model} deleted successfully.")
    else:
        print(f"Model {model} is not initialized or not found in MODELS_DICT.")

def check_model_response(response: str) -> bool:
    """
    Checks that a model response to a query was valid and not an error returned by the query instance.

    Parameters:
    - response (str): The models response

    Returns:
    - str response if valid, else None
    """
    if "Error in" not in response:
        return response
    else:
        return None

def query_model_retries(query: str, query_instance: object, query_checker: Callable[[str], bool], retries: int, initial_delay: int) -> str:
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
    delay = initial_delay
    while retry_count < retries:
        response = query_instance.query(query)
        response = query_checker(response)
        if response is not None:
            return response
        else:
            retry_count += 1
            print(f"Error querying model. Retry {retry_count}/{retries}")
            time.sleep(delay)
            delay *= 2
    return f"ERROR: Failed getting response for {query} after {retries} retries."

def collect_model_responses(model: str, query_instance: object, queries: list, query_checker: Callable[[str], bool], model_dict: dict, max_workers: int, retries: int, initial_delay: int) -> list:
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
    responses = [None] * len(queries)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(query_model_retries, query, query_instance, query_checker, retries, initial_delay): i for i, query in enumerate(queries)}
        for future in tqdm(as_completed(future_to_index), total=len(queries), desc=f"Running queries on {model}"):
            index = future_to_index[future]
            try:
                response = future.result()
            except Exception as e:
                response = f"ERROR: Exception {str(e)}"
            responses[index] = response

    return responses

def get_all_model_responses(data: pd.DataFrame, model_dict: dict, max_workers: int, query_col: str='question', retries: int=3, initial_delay: int=1) -> pd.DataFrame:
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
        MODELS_DICT[model]['query_instance'] = initialize_model(model)
        responses = collect_model_responses(model, MODELS_DICT[model]['query_instance'], data[query_col].tolist(), check_model_response, model_dict, max_workers, retries, initial_delay)
        data[f'{model}_response'] = responses
        delete_model(model)

    return data