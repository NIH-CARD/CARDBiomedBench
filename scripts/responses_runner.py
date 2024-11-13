import argparse
import json
from scripts.scripts_utils import load_dataset
from scripts.collect_responses.collect_responses_utils import get_model_responses

def main():
    print("*** *** *** RESPONSES RUNNER *** *** ***")
    parser = argparse.ArgumentParser(description="Get LLM results on a QA benchmark.")
    parser.add_argument('--qa_path', type=str, required=True, help='Path to the QA CSV file')
    parser.add_argument('--res_dir', type=str, required=True, help='Directory to save the response CSV files')
    parser.add_argument('--model_name', type=str, required=True, help="Specify a single model to run")
    parser.add_argument('--hyperparams', type=str, required=True, help='Model hyperparameters as JSON string')
    args = parser.parse_args()

    # Deserialize hyperparameters
    hyperparams = json.loads(args.hyperparams)

    qa_path = args.qa_path
    res_dir = args.res_dir
    model_name = args.model_name

    data = load_dataset(qa_path)
    if data.empty:
        print("No data to process. Exiting.")
        return

    print(f"## Getting model responses on {len(data)} Q/A ##")
    data = get_model_responses(data, model_name=model_name, res_dir=res_dir, hyperparams=hyperparams)
    print(f"## Responses collected and saved to {res_dir}/by_model/ ##")
    print("*** *** *** RESPONSES RUNNER COMPLETED *** *** ***")

if __name__ == "__main__":
    main()