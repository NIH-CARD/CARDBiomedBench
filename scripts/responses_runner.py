import argparse
import json
import time
import pandas as pd
from tqdm import tqdm
from typing import List, Callable

from scripts.scripts_utils import load_dataset, save_dataset

from scripts.collect_responses.gpt_query import GPTQuery
from scripts.collect_responses.gemini_query import GeminiQuery
from scripts.collect_responses.claude_query import ClaudeQuery
from scripts.collect_responses.perplexity_query import PerplexityQuery
from scripts.collect_responses.huggingface_query import HuggingFaceQuery

def initialize_model(model_name: str, system_prompt: str, max_new_tokens: int, temperature: float):
    """Initialize the model client and create an instance of the query class for the specified model."""

    if model_name == 'gpt-4o':
        return GPTQuery(system_prompt, 'gpt-4o-2024-05-13', max_tokens=max_new_tokens, temperature=temperature)
    elif model_name == 'gpt-3.5-turbo':
        return GPTQuery(system_prompt, 'gpt-3.5-turbo-0125', max_tokens=max_new_tokens, temperature=temperature)
    elif model_name == 'gemini-1.5-pro':
        return GeminiQuery(system_prompt, 'gemini-1.5-pro', max_tokens=max_new_tokens, temperature=temperature)
    elif model_name == 'claude-3.5-sonnet':
        return ClaudeQuery(system_prompt, 'claude-3-5-sonnet-20240620', max_tokens=max_new_tokens, temperature=temperature)
    elif model_name == 'perplexity-sonar-huge':
        return PerplexityQuery(system_prompt, 'llama-3.1-sonar-huge-128k-online', max_tokens=max_new_tokens, temperature=temperature)
    elif model_name == 'gemma-2-27b-it':
        return HuggingFaceQuery(system_prompt, 'google/gemma-2-27b-it', max_tokens=max_new_tokens, do_sample=False) 
    elif model_name == 'llama-3.1-70b-it':
        return HuggingFaceQuery(system_prompt, 'meta-llama/Meta-Llama-3.1-70B-Instruct', max_tokens=max_new_tokens, do_sample=False)
    else:
        raise ValueError(f"❌ Model {model_name} is not recognized.")

def delete_model(query_instance):
    """Delete the model instance and release any resources."""

    if query_instance is not None:
        query_instance.delete()

def check_model_response(response: str) -> tuple:
    """Checks that a model response to a query was valid and not an error returned by the query instance."""

    if "Error in" not in response:
        return response, True
    else:
        return response, False

def query_model_retries(query: str, query_instance: object, query_checker: Callable[[str], bool], retries: int, initial_delay: int) -> str:
    """Query the model with retries in case of failure."""

    retry_count = 0
    delay = initial_delay
    while retry_count < retries:
        response = query_instance.query(query)
        response, valid = query_checker(response)
        if valid:
            return response
        else:
            retry_count += 1
            print(f"❌ Error querying model. Retry {retry_count}/{retries}")
            time.sleep(delay)
            delay *= 2
    return f"ERROR: Failed getting response for '{query}' after {retries} retries. Error: {response}"

def collect_single_model_responses(model_name: str, query_instance: object, queries: list, query_checker: Callable[[str], bool], retries: int, initial_delay: int) -> list:
    """Collect responses from a specific model for a list of queries sequentially."""

    responses = []
    for query in tqdm(queries, desc=f"🔧 Running queries on {model_name}"):
        response = query_model_retries(query, query_instance, query_checker, retries, initial_delay)
        responses.append(response)

    return responses

def get_model_responses(data: pd.DataFrame, model_name: str, res_by_model_dir: str, 
    hyperparams: dict, query_col: str='question', retries: int=3, initial_delay: int=2,) -> pd.DataFrame:
    """Get responses from a single LLM for each query in the dataset, save the results."""
    data[f'{model_name}_response'] = ''
    # Extract hyperparameters
    system_prompt = hyperparams.get('system_prompt', '')
    max_new_tokens = hyperparams.get('max_new_tokens', 1024)
    temperature = hyperparams.get('temperature', 0.0)

    query_instance = initialize_model(model_name, system_prompt, max_new_tokens, temperature)
    responses = collect_single_model_responses(
        model_name,
        query_instance,
        data[query_col].tolist(),
        check_model_response,
        retries,
        initial_delay,
    )
    data[f'{model_name}_response'] = responses
    delete_model(query_instance)

    save_dataset(f'{res_by_model_dir}{model_name}_responses.csv', data)
    return data

def main():
    parser = argparse.ArgumentParser(description="Get LLM results on a QA benchmark.")
    parser.add_argument('--qa_path', type=str, required=True, help='Path to the QA CSV file')
    parser.add_argument('--res_by_model_dir', type=str, required=True, help='Directory to save the response CSV files')
    parser.add_argument('--model_name', type=str, required=True, help="Specify a single model to run")
    parser.add_argument('--hyperparams', type=str, required=True, help='Model hyperparameters as JSON string')
    args = parser.parse_args()

    # Deserialize hyperparameters
    hyperparams = json.loads(args.hyperparams)

    qa_path = args.qa_path
    res_by_model_dir = args.res_by_model_dir
    model_name = args.model_name

    data = load_dataset(qa_path)
    if data.empty:
        print("❌ No data to process. Exiting.")
        return

    print(f"🔧 Getting model responses on {len(data)} Q/A for {model_name}")
    data = get_model_responses(data, model_name=model_name, res_by_model_dir=res_by_model_dir, hyperparams=hyperparams)
    print(f"🔧 Responses collected and saved to {res_by_model_dir} ##")

if __name__ == "__main__":
    main()