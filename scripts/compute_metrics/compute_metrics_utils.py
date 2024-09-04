import pandas as pd
import re
from evaluate import load
from .prompts import biomedical_grading_prompt
from scripts.scripts_utils import load_dataset, save_dataset
from scripts.collect_responses.collect_responses_utils import collect_single_model_responses, initialize_model
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')

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

def get_all_model_BLEU_ROUGE_BERT(res_dir: str, model_dict: dict, gold_col: str='answer', response_col: str='response') -> None:
    """Compute BLEU, ROUGE, BERTScore for each model's responses and save the results back to CSV files."""

    # Load BLEU, ROUGE, and BERT evaluators once
    bleu = load('bleu')
    rouge = load('rouge')
    bertscore = load("bertscore")

    for model in model_dict:
        # Load the dataset for the current model
        data = load_dataset(f'{res_dir}/{model}_responses.csv')

        # Initialize columns for BLEU, ROUGE, and BERTScore
        for metric in ['BLEU', 'ROUGE2', 'ROUGEL', 'BERTScore']:
            data[f'{model}_{metric}'] = 0.0

        # Compute scores for each model response
        for index, row in data.iterrows():
            answer = row[gold_col]
            model_response = row[f'{model}_{response_col}']

            # Compute BLEU score
            bleu_score = bleu.compute(predictions=[model_response], references=[[answer]])
            data.at[index, f'{model}_BLEU'] = bleu_score['bleu']

            # Compute ROUGEL score
            rouge_score = rouge.compute(predictions=[model_response], references=[answer])
            for metric in ['rouge2', 'rougeL']:
                data.at[index, f'{model}_{metric.upper()}'] = rouge_score[metric]
            
            # Compute BERTScore
            bertscore_result = bertscore.compute(predictions=[model_response], references=[answer], lang="en")
            data.at[index, f'{model}_BERTScore'] = bertscore_result['f1']

        # Save the updated dataset back to the CSV file
        save_dataset(f'{res_dir}{model}_responses.csv', data)
        print(f"BLEU/ROUGE/BERTScore computed and saved for {model} to {res_dir}{model}_responses.csv")