import argparse
from scripts import MODELS_DICT
from scripts.utils import load_dataset, save_responses
from scripts.collect_responses.utils import get_all_model_responses

def main():
    parser = argparse.ArgumentParser(description="Get LLM results on a QA benchmark.")
    parser.add_argument('--local_qa_path', type=str, required=True, help='Path to the local QA CSV file')
    parser.add_argument('--local_res_path', type=str, required=True, help='Path to save the local res CSV file')
    args = parser.parse_args()

    local_qa_path = args.local_qa_path
    local_res_path = args.local_res_path

    data = load_dataset(local_qa_path)
    if data.empty:
        print("No data to process. Exiting.")
        return
    
    # TODO DELETE
    data = data[['uuid', 'question', 'answer']]
    data.dropna(inplace=True)
    data = data[:10]
    # TODO DELETE

    data = get_all_model_responses(data, MODELS_DICT)
    save_responses(local_res_path, data)
    print(f"Responses collected and saved to {local_res_path}.")

if __name__ == "__main__":
    main()