import argparse
from scripts import MODELS_DICT, METRICS_DICT, NUM_WORKERS, GRADING_MODEL
from scripts.compute_metrics.compute_metrics_utils import get_all_model_LLMEVAL, get_all_model_BLEU_ROUGE

def main():
    parser = argparse.ArgumentParser(description="Grade responses on the QA benchmark.")
    parser.add_argument('--res_dir', type=str, required=True, help='Directory to the responses CSV files')
    args = parser.parse_args()

    res_dir = args.res_dir

    if "LLMEVAL" in METRICS_DICT:
        get_all_model_LLMEVAL(res_dir, grading_model=GRADING_MODEL, model_dict=MODELS_DICT, max_workers=NUM_WORKERS)
    if "BLEU_ROUGE" in METRICS_DICT:
        get_all_model_BLEU_ROUGE(res_dir, model_dict=MODELS_DICT)


if __name__ == "__main__":
    main()
