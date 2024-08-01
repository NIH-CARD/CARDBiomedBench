import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from .prompts import biomedical_grading_prompt

def query_model_retries(query: str, query_instance: object, retries: int) -> str:
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
        if "Error in" not in response:
            return response
        else:
            retry_count += 1
            print(f"Error querying model. Retry {retry_count}/{retries}")
    return f"ERROR: Failed after {retries} retries."

def collect_model_responses(model: str, queries: list, model_dict: dict, retries: int=3, max_workers: int=10) -> list:
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
    print(model_dict)
    query_instance = model_dict[model]['query_instance']
    responses = [None] * len(queries)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(query_model_retries, query, query_instance, retries): i for i, query in enumerate(queries)}
        for future in tqdm(as_completed(future_to_index), total=len(queries), desc=f"Processing {model}"):
            index = future_to_index[future]
            try:
                response = future.result()
            except Exception as e:
                response = f"ERROR: Exception {str(e)}"
            responses[index] = response

    return responses

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