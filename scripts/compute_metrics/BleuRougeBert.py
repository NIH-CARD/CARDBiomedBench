import pandas as pd
from evaluate import load
from scripts.scripts_utils import load_dataset, save_dataset
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')

def get_all_model_BLEU_ROUGE_BERT(res_dir: str, models_to_grade: list, gold_col: str='answer', response_col: str='response') -> None:
    """Compute BLEU, ROUGE, BERTScore for each model's responses and save the results back to CSV files (vectorized)."""

    # Load BLEU, ROUGE, and BERT evaluators once
    bleu = load('bleu')
    rouge = load('rouge')
    bertscore = load("bertscore")

    for model in models_to_grade:
        print(f"\n=== Working on model: {model} ===")

        # Load the dataset for the current model
        data = load_dataset(f'{res_dir}/{model}_responses.csv')

        # Initialize columns for BLEU, ROUGE2, ROUGEL, BERTScore
        for metric in ['BLEU', 'ROUGE2', 'ROUGEL', 'BERTScore']:
            data[f'{model}_{metric}'] = 0.0

        # Prepare data for batch scoring
        all_preds = data[f'{model}_{response_col}'].fillna("").tolist()
        all_refs = data[gold_col].fillna("").tolist()

        # Compute BLEU (batch)
        try:
            print("Computing BLEU...")
            bleu_score = bleu.compute(predictions=all_preds, references=[[ref] for ref in all_refs])
            data[f'{model}_BLEU'] = bleu_score['bleu']
        except Exception as e:
            print(f"Error in BLEU for {model}: {e}")

        # Compute ROUGE (batch)
        try:
            print("Computing ROUGE...")
            rouge_score = rouge.compute(predictions=all_preds, references=all_refs)
            data[f'{model}_ROUGE2'] = rouge_score['rouge2']
            data[f'{model}_ROUGEL'] = rouge_score['rougeL']
        except Exception as e:
            print(f"Error in ROUGE for {model}: {e}")

        # Compute BERTScore (batch)
        try:
            print("Computing BERTScore...")
            bertscore_result = bertscore.compute(predictions=all_preds, references=all_refs, lang="en", device="cuda")
            data[f'{model}_BERTScore'] = bertscore_result['f1']
        except Exception as e:
            print(f"Error in BERTScore for {model}: {e}")

        print(f"=== Finished model: {model} ===")

        # Save updated dataset
        save_dataset(f'{res_dir}/{model}_responses.csv', data)
        print(f"Results saved to {res_dir}/{model}_responses.csv\n")