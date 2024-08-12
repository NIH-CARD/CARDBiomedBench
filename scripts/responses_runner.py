import argparse
from scripts import MODELS_DICT, NUM_WORKERS
from scripts.scripts_utils import load_dataset
from scripts.collect_responses.collect_responses_utils import get_all_model_responses

def main():
    parser = argparse.ArgumentParser(description="Get LLM results on a QA benchmark.")
    parser.add_argument('--qa_path', type=str, required=True, help='Path to the QA CSV file')
    parser.add_argument('--res_dir', type=str, required=True, help='Directory to save the response CSV files')
    args = parser.parse_args()

    qa_path = args.qa_path
    res_dir = args.res_dir

    data = load_dataset(qa_path)
    if data.empty:
        print("No data to process. Exiting.")
        return
    
    # TODO DELETE
    data = data[['uuid', 'question', 'answer']]
    data.dropna(inplace=True)
    data = data[:1]
    # TODO DELETE

    data = get_all_model_responses(data, model_dict=MODELS_DICT, max_workers=NUM_WORKERS, res_dir=res_dir)
    print(f"Responses collected and saved to {res_dir}.")

if __name__ == "__main__":
    main()