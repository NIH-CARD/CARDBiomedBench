import argparse
from scripts import MODELS_DICT, METRICS_DICT, GRADING_MODEL
from scripts.compute_metrics.BleuRougeBert import get_all_model_BLEU_ROUGE_BERT
from scripts.compute_metrics.BioScore import get_all_model_BioScore

def main():
    parser = argparse.ArgumentParser(description="Grade responses on the QA benchmark.")
    parser.add_argument('--res_dir', type=str, required=True, help='Directory to the responses CSV files')
    args = parser.parse_args()

    res_dir = args.res_dir

    if "BioScore" in METRICS_DICT:
        print("*** Getting BioScore Grades ***")
        get_all_model_BioScore(res_dir, model_dict=MODELS_DICT)
        print("*** BioScore Completed ***")
    if "BLEU_ROUGE_BERT" in METRICS_DICT:
        print("*** Getting BLEU, ROUGE, and BERTScore ***")
        get_all_model_BLEU_ROUGE_BERT(res_dir, model_dict=MODELS_DICT)
        print("*** BLEU, ROUGE, and BERTScore Completed ***")


if __name__ == "__main__":
    main()