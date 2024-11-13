import os
import pandas as pd
import re
import json
from .prompts import biomedical_grading_prompt
from scripts import BIOSCORE_SYSTEM_PROMPT
from scripts.scripts_utils import load_dataset, save_dataset
from scripts.collect_responses.collect_responses_utils import initialize_model

# Define the new cache subdirectory for batch queries
CACHE_DIR = ".cache/batch_queries"
os.makedirs(CACHE_DIR, exist_ok=True)

def check_BioScore_response(response: str) -> tuple:
    """Check the BioScore evaluation response for a valid score."""
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", response)
    if match:
        number = float(match.group(0))
        if number in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            return number / 3.0, True
        if number == -1:
            return number, True
    return None, False

def process_batch_results(batch_result_path: str, batch_file_path: str, grading_model) -> dict:
    """
    Load and process the batch results from the .jsonl file. Cache only valid responses using the query found in the original batch file.
    
    Parameters:
    - batch_result_path (str): The path to the file containing the batch results.
    - batch_file_path (str): The path to the original batch file.
    - grading_model (GPTQuery): The grading model instance with cache support.

    Returns:
    - dict: A dictionary with the BioScore results mapped by custom_id.
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
            custom_id = result.get("custom_id")  # Use this to map back to the original query
            response_content = result.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")

            if custom_id in batch_queries:
                original_query = batch_queries[custom_id]

                # Check the response and extract the BioScore
                bioscore, valid = check_BioScore_response(response_content)
                if valid:
                    # Generate the cache key using the original query
                    cache_key = grading_model.get_cache_key(original_query)
                    
                    # Cache the response if it isn't already cached and is valid
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

def generate_batch_file(grading_prompts, batch_file_path, grading_model, uuids) -> bool:
    """Generate a .jsonl batch file with grading prompts for batch querying, excluding cached responses.
    
    Returns True if a new batch file is created, False otherwise.
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

def map_bioscore_results_to_dataframe(data: pd.DataFrame, bioscore_results: dict, grading_model, model: str, query_col: str, gold_col: str, response_col: str) -> pd.DataFrame:
    """
    Map the BioScore results to the DataFrame. If the result is not in bioscore_results, 
    check if it exists in the cache and retrieve it if valid.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the model responses.
    - bioscore_results (dict): The dictionary with BioScore results mapped by uuid.
    - grading_model (GPTQuery): The grading model instance with cache support.
    - model (str): The model name to map BioScore results for.
    - query_col (str): The column name for the query text in the DataFrame.
    - gold_col (str): The column name for the gold answer text in the DataFrame.
    - response_col (str): The column name for the model response in the DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with BioScore results mapped for the specific model.
    """
    for i, row in data.iterrows():
        uuid = row['uuid']
        if str(uuid) in bioscore_results:
            # If the result is in bioscore_results, use it
            data.at[i, f'{model}_BioScore'] = bioscore_results[str(uuid)]
        else:
            # Check the cache for the response if it's not in bioscore_results
            cache_key = grading_model.get_cache_key(
                biomedical_grading_prompt(row[query_col], row[gold_col], row[f'{model}_{response_col}'])
            )
            if cache_key in grading_model.cache:
                cached_response = grading_model.cache[cache_key]
                bioscore, valid = check_BioScore_response(cached_response)
                if valid:
                    data.at[i, f'{model}_BioScore'] = bioscore
                else:
                    print(f"Invalid cached response for uuid {uuid}")
            else:
                print(f"No BioScore found for uuid {uuid}")
    
    return data

def submit_batches(grading_model, model_dict, res_dir, query_col='question', gold_col='answer', response_col='response'):
    """
    Submits batch files for BioScore grading for all models in the model_dict.
    Returns a dictionary mapping models to batch IDs.
    """
    batch_ids = {}

    for model in model_dict:
        # Load the dataset
        data = load_dataset(f'{res_dir}/{model}_responses.csv')

        # Format BioScore grading prompts
        bioscore_grading_prompts = [
            biomedical_grading_prompt(row[query_col], row[gold_col], row[f'{model}_{response_col}'])
            for _, row in data.iterrows()
        ]

        # Get the uuids
        uuids = data['uuid'].tolist()

        # Generate the batch file for this model
        batch_file_path = f"{CACHE_DIR}/{model}_grading_batch.jsonl"
        batch_file_created = generate_batch_file(bioscore_grading_prompts, batch_file_path, grading_model, uuids)

        # Submit batch only if a new batch file was created
        if batch_file_created:
            print(f"Submitting BioScore grading for {model} to gpt-4o batch API...")
            batch_id = grading_model.submit_batch_query(batch_file_path)
            batch_ids[model] = batch_id
            print(f"Batch ID {batch_id} submitted for {model}")
        else:
            print(f"No new batch file created for {model}. Skipping submission.")

    return batch_ids

def poll_batch_results(grading_model, model, batch_ids, res_dir, query_col='question', gold_col='answer', response_col='response'):
    """
    Polls the batch results for all models in batch_ids and processes the results.
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
    bioscore_results = process_batch_results(batch_result_path, batch_file_path, grading_model)

    return bioscore_results

def get_all_model_BioScore(res_dir: str, model_dict: dict, query_col: str='question', gold_col: str='answer', response_col: str='response') -> pd.DataFrame:
    """Grade responses from multiple LLMs with a specific prompt & GPT-4o for each query in the dataset."""
    grading_model = initialize_model("gpt-4o", system_prompt=BIOSCORE_SYSTEM_PROMPT)

    # Step 1: Submit batch files
    batch_ids = submit_batches(grading_model, model_dict, res_dir, query_col, gold_col, response_col)

    # Step 2: Poll each model for batch results after all submissions
    for model in model_dict:
        if model in batch_ids:
            new_bioscore_results = poll_batch_results(grading_model, model, batch_ids, res_dir, query_col, gold_col, response_col)
        else: 
            new_bioscore_results = {}

        # Load the original dataset
        data = load_dataset(f'{res_dir}/{model}_responses.csv')

        # Map the BioScore results to the dataframe
        data = map_bioscore_results_to_dataframe(data, new_bioscore_results, grading_model, model, query_col, gold_col, response_col)

        # Save the updated dataframe
        save_dataset(f'{res_dir}/{model}_responses.csv', data)
        print(f"BioScore computed and saved for {model} to {res_dir}{model}_responses.csv")

    # Cleanup
    print("All batches submitted and results processed.")
    grading_model.delete()