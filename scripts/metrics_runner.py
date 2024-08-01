import argparse
from scripts import MODELS_DICT, METRICS_DICT, NUM_WORKERS
from scripts.utils import load_dataset, save_responses
from scripts.compute_metrics.utils import get_all_model_LLMEVAL

def main():
    parser = argparse.ArgumentParser(description="Grade responses on the QA benchmark.")
    parser.add_argument('--local_res_path', type=str, required=True, help='Local path to the results CSV file')
    parser.add_argument('--local_scored_path', type=str, required=True, help='Local path to the scored CSV file')
    args = parser.parse_args()

    local_res_path = args.local_res_path
    local_scored_path = args.local_scored_path

    data = load_dataset(local_res_path)
    if data.empty:
        print("No data to process. Exiting.")
        return

    if "LLMEVAL" in METRICS_DICT:
        data = get_all_model_LLMEVAL(data, grading_model='gpt-4o', model_dict=MODELS_DICT, max_workers=NUM_WORKERS)

    save_responses(local_scored_path, data)
    print(f"Responses scored and saved to {local_scored_path}.")

if __name__ == "__main__":
    main()
