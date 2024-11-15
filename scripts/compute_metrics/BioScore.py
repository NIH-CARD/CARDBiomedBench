"""
BioScore.py

This script processes batch queries and responses for BioScore grading using a GPT-based model.
It handles caching, batch submission, result polling, and mapping of scores back to the dataset.
"""

import os
import re
import json
from typing import Tuple, Dict, List

import pandas as pd

from scripts.scripts_utils import load_dataset, save_dataset
from scripts.responses_runner import initialize_model

# Define the new cache subdirectory for batch queries
CACHE_DIR = ".cache/batch_queries"
os.makedirs(CACHE_DIR, exist_ok=True)


def check_BioScore_response(response: str) -> Tuple[float, bool]:
    """
    Check the BioScore evaluation response for a valid score.

    Args:
        response (str): The response string from the grading model.

    Returns:
        Tuple[float, bool]: A tuple containing the normalized BioScore and a validity flag.
    """
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", response)
    if match:
        number = float(match.group(0))
        if number in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            return number / 3.0, True
        if number == -1:
            return number, True
    return None, False


def process_batch_results(
    batch_result_path: str,
    batch_file_path: str,
    grading_model
) -> Dict[str, float]:
    """
    Load and process the batch results from the .jsonl file.
    Cache only valid responses using the queries found in the original batch file.

    Args:
        batch_result_path (str): Path to the file containing the batch results.
        batch_file_path (str): Path to the original batch file.
        grading_model: The grading model instance with cache support.

    Returns:
        Dict[str, float]: A dictionary mapping custom IDs to BioScore results.
    """
    bioscore_results = {}
    cache = grading_model.cache

    # Load the original batch queries from the batch file
    batch_queries = {}
    with open(batch_file_path, 'r') as batch_file:
        for line in batch_file:
            batch_request = json.loads(line)
            custom_id = batch_request.get('custom_id')
            if custom_id:
                query = batch_request.get("body", {}).get("messages", [{}])[0].get("content", "")
                batch_queries[custom_id] = query

    # Read and process the batch results
    with open(batch_result_path, 'r') as result_file:
        for line in result_file:
            result = json.loads(line)
            custom_id = result.get("custom_id")  # Map back to the original query
            response_content = result.get("response", {}).get("body", {}).get(
                "choices", [{}])[0].get("message", {}).get("content", "")

            if custom_id in batch_queries:
                original_query = batch_queries[custom_id]

                # Check the response and extract the BioScore
                bioscore, valid = check_BioScore_response(response_content)
                if valid:
                    # Generate the cache key using the original query
                    cache_key = grading_model.get_cache_key(original_query)

                    # Cache the response if it's valid and not already cached
                    if cache_key not in cache:
                        grading_model.cache[cache_key] = response_content

                    bioscore_results[custom_id] = bioscore
                else:
                    print(f"Invalid response for {custom_id}: {response_content}")
            else:
                print(f"Custom ID {custom_id} not found in batch queries.")

    # Save the updated cache (only valid responses will be cached)
    grading_model.save_cache()

    return bioscore_results


def generate_batch_file(
    grading_prompts: List[str],
    batch_file_path: str,
    grading_model,
    uuids: List[str]
) -> bool:
    """
    Generate a .jsonl batch file with grading prompts for batch querying, excluding cached responses.

    Args:
        grading_prompts (List[str]): List of grading prompts.
        batch_file_path (str): Path to save the batch file.
        grading_model: The grading model instance with cache support.
        uuids (List[str]): List of UUIDs corresponding to the prompts.

    Returns:
        bool: True if a new batch file is created, False otherwise.
    """
    cache = grading_model.cache
    new_batch_requests = []

    # Only include prompts that are not in the cache
    for prompt, uuid in zip(grading_prompts, uuids):
        cache_key = grading_model.get_cache_key(prompt)

        if cache_key not in cache:
            batch_request = {
                "custom_id": str(uuid),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": grading_model.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1024,
                    "temperature": 0
                }
            }
            new_batch_requests.append(batch_request)

    # Delete any old batch file for the model
    if os.path.exists(batch_file_path):
        os.remove(batch_file_path)

    # If there are new requests, write them to the batch file
    if new_batch_requests:
        with open(batch_file_path, 'w') as f:
            for request in new_batch_requests:
                f.write(json.dumps(request) + '\n')
        return True
    else:
        return False


def map_bioscore_results_to_dataframe(
    data: pd.DataFrame,
    bioscore_results: Dict[str, float],
    bioscore_grading_prompt: str,
    grading_model,
    model: str,
    query_col: str,
    gold_col: str,
    response_col: str
) -> pd.DataFrame:
    """
    Map the BioScore results to the DataFrame. If the result is not in bioscore_results,
    check if it exists in the cache and retrieve it if valid.

    Args:
        data (pd.DataFrame): DataFrame containing the model responses.
        bioscore_results (Dict[str, float]): Dictionary with BioScore results mapped by UUID.
        bioscore_grading_prompt (str): The grading prompt template.
        grading_model: The grading model instance with cache support.
        model (str): The model name to map BioScore results for.
        query_col (str): Column name for the query text in the DataFrame.
        gold_col (str): Column name for the gold answer text in the DataFrame.
        response_col (str): Column name for the model response in the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with BioScore results mapped for the specific model.
    """
    for i, row in data.iterrows():
        uuid = row['uuid']
        if str(uuid) in bioscore_results:
            # If the result is in bioscore_results, use it
            data.at[i, f'{model}_BioScore'] = bioscore_results[str(uuid)]
        else:
            # Check the cache for the response if it's not in bioscore_results
            prompt = bioscore_grading_prompt.format(
                question=row[query_col],
                gold_res=row[gold_col],
                pred_res=row[f'{model}_{response_col}']
            )

            # Generate the cache key
            cache_key = grading_model.get_cache_key(prompt)

            if cache_key in grading_model.cache:
                cached_response = grading_model.cache[cache_key]
                bioscore, valid = check_BioScore_response(cached_response)
                if valid:
                    data.at[i, f'{model}_BioScore'] = bioscore
                else:
                    print(f"Invalid cached response for UUID {uuid}")
            else:
                print(f"No BioScore found for UUID {uuid}")

    return data


def submit_batches(
    grading_model,
    models_to_use: List[str],
    bioscore_grading_prompt: str,
    res_dir: str,
    query_col: str = 'question',
    gold_col: str = 'answer',
    response_col: str = 'response'
) -> Dict[str, str]:
    """
    Submit batch files for BioScore grading for all models in models_to_use.
    Returns a dictionary mapping models to batch IDs.

    Args:
        grading_model: The grading model instance with batch query support.
        models_to_use (List[str]): List of model names to process.
        bioscore_grading_prompt (str): The grading prompt template.
        res_dir (str): Directory containing the model response CSV files.
        query_col (str, optional): Column name for queries. Defaults to 'question'.
        gold_col (str, optional): Column name for gold answers. Defaults to 'answer'.
        response_col (str, optional): Column name for model responses. Defaults to 'response'.

    Returns:
        Dict[str, str]: Dictionary mapping model names to batch IDs.
    """
    batch_ids = {}

    for model in models_to_use:
        # Load the dataset
        data = load_dataset(f'{res_dir}/{model}_responses.csv')

        # Format BioScore grading prompts
        bioscore_grading_prompts = [
            bioscore_grading_prompt.format(
                question=row[query_col],
                gold_res=row[gold_col],
                pred_res=row[f'{model}_{response_col}']
            )
            for _, row in data.iterrows()
        ]

        # Get the UUIDs
        uuids = data['uuid'].astype(str).tolist()

        # Generate the batch file for this model
        batch_file_path = f"{CACHE_DIR}/{model}_grading_batch.jsonl"
        batch_file_created = generate_batch_file(
            bioscore_grading_prompts,
            batch_file_path,
            grading_model,
            uuids
        )

        # Submit batch only if a new batch file was created
        if batch_file_created:
            print(f"Submitting BioScore grading for {model} to GPT-4o batch API...")
            batch_id = grading_model.submit_batch_query(batch_file_path)
            batch_ids[model] = batch_id
            print(f"Batch ID {batch_id} submitted for {model}")
        else:
            print(f"No new batch file created for {model}. Skipping submission.")

    return batch_ids


def poll_batch_results(
    grading_model,
    model: str,
    batch_ids: Dict[str, str],
    res_dir: str,
    query_col: str = 'question',
    gold_col: str = 'answer',
    response_col: str = 'response'
) -> Dict[str, float]:
    """
    Poll the batch results for a specific model and process the results.

    Args:
        grading_model: The grading model instance with batch query support.
        model (str): The model name to process.
        batch_ids (Dict[str, str]): Dictionary mapping models to batch IDs.
        res_dir (str): Directory containing the model response CSV files.
        query_col (str, optional): Column name for queries. Defaults to 'question'.
        gold_col (str, optional): Column name for gold answers. Defaults to 'answer'.
        response_col (str, optional): Column name for model responses. Defaults to 'response'.

    Returns:
        Dict[str, float]: Dictionary mapping UUIDs to BioScore results.
    """
    batch_id = batch_ids[model]
    print(f"Polling BioScore batch results for {model} with batch ID {batch_id}...")
    batch_results = grading_model.poll_batch_status(batch_id)

    # Save the batch results to a JSONL file
    batch_result_path = f"{CACHE_DIR}/{model}_grading_batch_results.jsonl"
    with open(batch_result_path, 'w') as f:
        f.write(batch_results)
    print(f"Batch results saved for {model} to {batch_result_path}")

    # Process the results and validate them
    batch_file_path = f"{CACHE_DIR}/{model}_grading_batch.jsonl"
    bioscore_results = process_batch_results(
        batch_result_path,
        batch_file_path,
        grading_model
    )

    return bioscore_results


def get_all_model_BioScore(
    res_dir: str,
    models_to_use: List[str],
    hyperparams: dict,
    bioscore_grading_prompt: str,
    query_col: str = 'question',
    gold_col: str = 'answer',
    response_col: str = 'response'
) -> None:
    """
    Grade responses from multiple LLMs with a specific prompt using GPT-4o for each query in the dataset.

    Args:
        res_dir (str): Directory containing the model response CSV files.
        models_to_use (List[str]): List of model names to grade.
        hyperparams (dict): Hyperparameters for the grading model.
        bioscore_grading_prompt (str): The grading prompt template.
        query_col (str, optional): Column name for queries. Defaults to 'question'.
        gold_col (str, optional): Column name for gold answers. Defaults to 'answer'.
        response_col (str, optional): Column name for model responses. Defaults to 'response'.
    """
    # Extract the system prompt from hyperparams
    grading_model_name = 'gpt-4o'
    bioscore_system_prompt = hyperparams.get('system_prompt', '')
    max_new_tokens = hyperparams.get('max_new_tokens', 1024)
    temperature = hyperparams.get('temperature', 0.0)
    grading_model = initialize_model(
        grading_model_name,
        bioscore_system_prompt,
        max_new_tokens,
        temperature
    )

    # Step 1: Submit batch files
    batch_ids = submit_batches(
        grading_model,
        models_to_use,
        bioscore_grading_prompt,
        res_dir,
        query_col,
        gold_col,
        response_col
    )

    # Step 2: Poll each model for batch results after all submissions
    for model in models_to_use:
        if model in batch_ids:
            new_bioscore_results = poll_batch_results(
                grading_model,
                model,
                batch_ids,
                res_dir,
                query_col,
                gold_col,
                response_col
            )
        else:
            new_bioscore_results = {}

        # Load the original dataset
        data = load_dataset(f'{res_dir}/{model}_responses.csv')

        # Map the BioScore results to the DataFrame
        data = map_bioscore_results_to_dataframe(
            data,
            new_bioscore_results,
            bioscore_grading_prompt,
            grading_model,
            model,
            query_col,
            gold_col,
            response_col
        )

        # Save the updated DataFrame
        save_dataset(f'{res_dir}{model}_responses.csv', data)
        print(f"BioScore computed and saved for {model} to {res_dir}{model}_responses.csv")

    # Cleanup
    print("All batches submitted and results processed.")
    grading_model.delete()