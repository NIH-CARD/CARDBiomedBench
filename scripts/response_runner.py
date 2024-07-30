import argparse
import pandas as pd
from tqdm import tqdm
from scripts import MODELS
from scripts.utils import load_dataset, save_responses
from scripts.collect_responses.gpt4_utils import query_gpt4o
from scripts.collect_responses.gemini_utils import query_gemini

def get_model_responses(query: str, model_dict: dict) -> dict:
    """
    Get responses from all specified models for a given query.

    Parameters:
    - query (str): The input query string.

    Returns:
    - dict: A dictionary with model names as keys and their responses as values.
    """
    responses = {}
    if 'gpt-4o' in model_dict:
        responses['gpt-4o'] = query_gpt4o(query)
    if 'gemini-1.5-pro' in model_dict:
        responses['gemini-1.5-pro'] = query_gemini(query)
    return responses

def get_llm_responses(data: pd.DataFrame, model_dict: dict, query_col: str='question', retries: int=3) -> pd.DataFrame:
    """
    Get responses from multiple LLMs for each query in the dataset, with retry on failure.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing queries.
    - query_col (str): The column name containing the queries.
    - retries (int): Number of retries for each API call in case of failure.

    Returns:
    - pd.DataFrame: The DataFrame with added or updated response columns for each model.
    """
    data_to_process = data.copy()
    total_tasks = len(data_to_process) * len(model_dict)

    with tqdm(total=total_tasks, desc="Getting model responses") as pbar:
        for index, row in data_to_process.iterrows():
            query = row[query_col]
            for model in model_dict:
                retry_count = 0
                while retry_count < retries:
                    responses = get_model_responses(query, model_dict)
                    response = responses.get(model, "ERROR")
                    if "Error in" not in response:
                        data.loc[index, f'{model}_response'] = response
                        break
                    else:
                        retry_count += 1
                        if retry_count == retries:
                            data.loc[index, f'{model}_response'] = "ERROR"
                pbar.update(1)

    return data

def main():
    parser = argparse.ArgumentParser(description="Get LLM results on a QA benchmark.")
    parser.add_argument('--local_qa_path', type=str, required=True, help='Path to the local QA CSV file')
    parser.add_argument('--local_res_path', type=str, required=True, help='Path to save the local res CSV file')
    args = parser.parse_args()

    local_qa_path = args.local_qa_path
    local_res_path = args.local_res_path

    data = load_dataset(local_qa_path)
    if data.empty:
        print("No data to process. Exiting.")
        return

    # TODO DELETE
    data = data[['uuid', 'question', 'answer']]
    data.dropna(inplace=True)
    data = data[:2]
    # TODO DELETE

    data = get_llm_responses(data, MODELS)
    save_responses(local_res_path, data)
    print(f"Responses collected and saved to {local_res_path}.")

if __name__ == "__main__":
    main()