import argparse
from scripts import MODELS_DICT, METRICS_DICT, NUM_WORKERS, GRADING_MODEL
from scripts.compute_metrics.compute_metrics_utils import get_all_model_LLMEVAL

def main():
    parser = argparse.ArgumentParser(description="Grade responses on the QA benchmark.")
    parser.add_argument('--local_res_dir', type=str, required=True, help='Directory to the local res CSV files')
    args = parser.parse_args()

    local_res_dir = args.local_res_dir

    if "LLMEVAL" in METRICS_DICT:
        get_all_model_LLMEVAL(local_res_dir, grading_model=GRADING_MODEL, model_dict=MODELS_DICT, max_workers=NUM_WORKERS)


if __name__ == "__main__":
    main()
