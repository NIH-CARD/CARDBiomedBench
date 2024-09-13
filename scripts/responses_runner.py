import argparse
from scripts import MODELS_DICT, TEMPLATE_SAMPLES
from scripts.scripts_utils import load_dataset, sample_data_by_template
from scripts.collect_responses.collect_responses_utils import get_all_model_responses

def main():
    print("*** *** *** RESPONSES RUNNER *** *** ***")
    parser = argparse.ArgumentParser(description="Get LLM results on a QA benchmark.")
    parser.add_argument('--qa_path', type=str, required=True, help='Path to the QA CSV file')
    parser.add_argument('--res_dir', type=str, required=True, help='Directory to save the response CSV files')
    parser.add_argument('--template', type=bool, default=False, help="Whether to run template-based sampling")
    args = parser.parse_args()

    qa_path = args.qa_path
    res_dir = args.res_dir
    template_flag = args.template

    data = load_dataset(qa_path)
    if data.empty:
        print("No data to process. Exiting.")
        return
    
    if template_flag:
        print(f"## Sampling template file {TEMPLATE_SAMPLES} samples per 'template uuid' ##")
        try:
            data = sample_data_by_template(data, TEMPLATE_SAMPLES)
        except ValueError as e:
            print(f"Error during sampling: {e}")
            return
    
    # TODO DELETE
    data = data[['uuid', 'question', 'answer']]
    data.dropna(inplace=True)
    print(data)
    # TODO DELETE

    print("## Getting model responses")
    data = get_all_model_responses(data, model_dict=MODELS_DICT, res_dir=res_dir)
    print(f"## Responses collected and saved to {res_dir}")

if __name__ == "__main__":
    main()