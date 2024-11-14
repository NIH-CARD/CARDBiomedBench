import argparse
import json
from scripts.compute_metrics.BleuRougeBert import get_all_model_BLEU_ROUGE_BERT
from scripts.compute_metrics.BioScore import get_all_model_BioScore

def main():
    parser = argparse.ArgumentParser(description="Grade responses on the QA benchmark.")
    parser.add_argument('--res_by_model_dir', type=str, required=True, help='Directory to the responses CSV files')
    parser.add_argument('--models_to_grade', nargs='+', required=True, help='List of models to grade')
    parser.add_argument('--metrics_to_use', nargs='+', required=True, help='List of metrics to compute')
    parser.add_argument('--hyperparams', type=str, required=True, help='Hyperparameters in JSON format')
    parser.add_argument('--bioscore_grading_prompt', type=str, required=False, help='BioScore grading prompt')
    args = parser.parse_args()

    res_dir = args.res_by_model_dir
    models_to_grade = args.models_to_grade
    metrics_to_use = args.metrics_to_use
    hyperparams = json.loads(args.hyperparams)
    bioscore_grading_prompt = args.bioscore_grading_prompt

    if "BioScore" in metrics_to_use:
        print("🔧 Getting BioScore Grades")
        get_all_model_BioScore(res_dir, models_to_grade, hyperparams, bioscore_grading_prompt)
        print("🔧 BioScore Completed")
    if "BLEU_ROUGE_BERT" in metrics_to_use:
        print("🔧 Getting BLEU, ROUGE, and BERTScore")
        get_all_model_BLEU_ROUGE_BERT(res_dir, models_to_grade)
        print("🔧 BLEU, ROUGE, and BERTScore Completed")

if __name__ == "__main__":
    main()