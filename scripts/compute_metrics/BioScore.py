import pandas as pd
import re
import json
from .prompts import biomedical_grading_prompt
from scripts.scripts_utils import load_dataset, save_dataset
from scripts.collect_responses.collect_responses_utils import collect_single_model_responses, initialize_model

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

def process_batch_results(batch_result_path: str) -> dict:
    """Load and process the batch results from the .jsonl file."""
    bioscore_results = {}

    # Read the .jsonl file line by line
    with open(batch_result_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result.get("custom_id")  # This is now the uuid
            response_content = result.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Check the response using check_BioScore_response
            bioscore, valid = check_BioScore_response(response_content)
            
            if valid:
                bioscore_results[custom_id] = bioscore
            else:
                print(f"Invalid response for {custom_id}: {response_content}")
    
    return bioscore_results

def generate_batch_file(grading_prompts, batch_file_path, grading_model, uuids):
    """Generate a .jsonl batch file with grading prompts for batch querying."""
    with open(batch_file_path, 'w') as f:
        for i, (prompt, uuid) in enumerate(zip(grading_prompts, uuids)):
            batch_request = {
                "custom_id": str(uuid),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": grading_model.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1024
                }
            }
            f.write(json.dumps(batch_request) + '\n')

def get_all_model_BioScore(res_dir: str, model_dict: dict, query_col: str='question', gold_col: str='answer', response_col: str='response') -> pd.DataFrame:
    """Grade responses from multiple LLMs with a specific prompt & GPT-4o for each query in the dataset, with retry on failure."""
    
    grading_model = initialize_model("gpt-4o", system_prompt="")

    # Create all .jsonl batch files
    for model in model_dict:
        # Load the results file for the specific model
        data = load_dataset(f'{res_dir}/{model}_responses.csv')

        # Format into BioScore grading prompts
        bioscore_grading_prompts = [
            biomedical_grading_prompt(row[query_col], row[gold_col], row[f'{model}_{response_col}'])
            for _, row in data.iterrows()
        ]

        # Get the uuids
        uuids = data['uuid'].tolist()

        # Generate the .jsonl batch file for this model
        batch_file_path = f"temp/{model}_grading_batch.jsonl"
        generate_batch_file(bioscore_grading_prompts, batch_file_path, grading_model, uuids)

    # Submit each batch and process the responses
    batch_ids = {}
    for model in model_dict:
        cur_batch_file = f"temp/{model}_grading_batch.jsonl"
        print(f"Submitting BioScore grading for {model} to gpt-4o batch API...")

        # Submit the batch and store the batch ID
        batch_id = grading_model.submit_batch_query(cur_batch_file)
        batch_ids[model] = batch_id
        print(f"Batch ID {batch_id} submitted for {model}")

    # Poll for batch completion and retrieve results
    for model, batch_id in batch_ids.items():
        print(f"Polling BioScore batch results for {model} with batch ID {batch_id}...")

        # Poll the batch and retrieve the results
        batch_results = grading_model.poll_batch_status(batch_id)

        # Save the batch results to a JSONL file
        batch_result_path = f"temp/{model}_grading_batch_results.jsonl"
        with open(batch_result_path, 'w') as f:
            f.write(batch_results)
        print(f"Batch results saved for {model} to {batch_result_path}")

        # Process the results and validate them using check_BioScore_response
        bioscore_results = process_batch_results(batch_result_path)

        # Load the original dataframe for this model
        data = load_dataset(f'{res_dir}/{model}_responses.csv')

        # Map the results to the dataframe using the custom_id
        for i, row in data.iterrows():
            uuid = row['uuid']
            if str(uuid) in bioscore_results:
                data.at[i, f'{model}_BioScore'] = bioscore_results[str(uuid)]
            else:
                print(f"No BioScore found for uuid {uuid}")

        # Save the updated dataframe
        save_dataset(f'{res_dir}/{model}_responses.csv', data)
        print(f"BioScore computed and saved for {model} to {res_dir}/{model}_responses.csv")

    print("All batches submitted and results processed.")
    grading_model.delete()