import pandas as pd
import re
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

def get_all_model_BioScore(res_dir: str, grading_model: str, model_dict: dict, query_col: str='question', gold_col: str='answer', response_col: str='response', retries: int=3, initial_delay: int=1) -> pd.DataFrame:
    """Grade responses from multiple LLMs with a specific prompt & GPT-4o for each query in the dataset, with retry on failure."""

    for model in model_dict:
        data = load_dataset(f'{res_dir}/{model}_responses.csv')
        data[f'{model}_BioScore'] = 0.0
        grading_prompts = [
            biomedical_grading_prompt(row[query_col], row[gold_col], row[f'{model}_{response_col}'])
            for _, row in data.iterrows()
        ]
        query_instance = initialize_model(grading_model, system_prompt="")
        responses = collect_single_model_responses(grading_model, query_instance, grading_prompts, check_BioScore_response, retries, initial_delay)
        query_instance.delete()
        data[f'{model}_BioScore'] = responses
        save_dataset(f'{res_dir}/{model}_responses.csv', data)
        print(f"BioScore computed and saved for {model} to {res_dir}/{model}_responses.csv")