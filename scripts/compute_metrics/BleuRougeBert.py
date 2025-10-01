import pandas as pd
from evaluate import load
from scripts.scripts_utils import load_dataset, save_dataset
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')

def get_all_model_BLEU_ROUGE_BERT(res_dir: str, models_to_grade: list, gold_col: str='answer', response_col: str='response') -> None:
    """Compute BLEU (per-example), ROUGE (per-example), BERTScore (batched) for each model's responses and save to CSV."""

    # Load BLEU, ROUGE, and BERT evaluators once
    bleu = load('bleu')
    rouge = load('rouge')
    bertscore = load("bertscore")

    for model in models_to_grade:
        print(f"\n=== Working on model: {model} ===")

        # Load dataset
        data = load_dataset(f'{res_dir}/{model}_responses.csv')

        # Initialize columns
        for metric in ['BLEU', 'ROUGE2', 'ROUGEL', 'BERTScore']:
            data[f'{model}_{metric}'] = 0.0

        # Prepare for BERTScore batch
        all_preds = data[f'{model}_{response_col}'].fillna("").tolist()
        all_refs = data[gold_col].fillna("").tolist()

        # ---- BERTScore (batched)
        try:
            print("Computing BERTScore...")
            bertscore_result = bertscore.compute(predictions=all_preds, references=all_refs, lang="en", device="cuda")
            data[f'{model}_BERTScore'] = bertscore_result['f1']
        except Exception as e:
            print(f"Error in BERTScore for {model}: {e}")

        # ---- BLEU + ROUGE (per-example loop)
        print("Computing BLEU + ROUGE (per-example)...")
        for index, (pred, ref) in tqdm(enumerate(zip(all_preds, all_refs)), total=len(all_preds), desc=f"Scoring {model} (BLEU/ROUGE)"):
            
            # BLEU
            try:
                bleu_score = bleu.compute(predictions=[pred], references=[[ref]])
                data.at[index, f'{model}_BLEU'] = bleu_score['bleu']
            except Exception:
                data.at[index, f'{model}_BLEU'] = 0.0

            # ROUGE
            try:
                rouge_score = rouge.compute(predictions=[pred], references=[ref])
                data.at[index, f'{model}_ROUGE2'] = rouge_score['rouge2']
                data.at[index, f'{model}_ROUGEL'] = rouge_score['rougeL']
            except Exception:
                data.at[index, f'{model}_ROUGE2'] = 0.0
                data.at[index, f'{model}_ROUGEL'] = 0.0

        print(f"=== Finished model: {model} ===")

        # Save updated dataset
        save_dataset(f'{res_dir}/{model}_responses.csv', data)
        print(f"Results saved to {res_dir}/{model}_responses.csv\n")