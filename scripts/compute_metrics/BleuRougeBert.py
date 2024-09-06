import pandas as pd
from evaluate import load
from scripts.scripts_utils import load_dataset, save_dataset
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')

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

            # Compute BLEU score, default to 0 on failure
            try:
                bleu_score = bleu.compute(predictions=[model_response], references=[[answer]])
                data.at[index, f'{model}_BLEU'] = bleu_score['bleu']
            except Exception:
                data.at[index, f'{model}_BLEU'] = 0.0

            # Compute ROUGE score, default to 0 on failure
            try:
                rouge_score = rouge.compute(predictions=[model_response], references=[answer])
                for metric in ['rouge2', 'rougeL']:
                    data.at[index, f'{model}_{metric.upper()}'] = rouge_score[metric]
            except Exception:
                for metric in ['rouge2', 'rougeL']:
                    data.at[index, f'{model}_{metric.upper()}'] = 0.0

            # Compute BERTScore, default to 0 on failure
            try:
                bertscore_result = bertscore.compute(predictions=[model_response], references=[answer], lang="en")
                data.at[index, f'{model}_BERTScore'] = bertscore_result['f1']
            except Exception:
                data.at[index, f'{model}_BERTScore'] = 0.0

        # Save the updated dataset back to the CSV file
        save_dataset(f'{res_dir}{model}_responses.csv', data)
        print(f"BLEU/ROUGE/BERTScore computed and saved for {model} to {res_dir}{model}_responses.csv")