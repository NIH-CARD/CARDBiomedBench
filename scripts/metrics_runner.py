import argparse
import re
import pandas as pd
from tqdm import tqdm
from scripts import MODELS, METRICS
from scripts.utils import load_dataset, save_responses
from scripts.collect_responses.gpt4o_query import query_gpt4o
from scripts.compute_metrics.prompts import biomedical_grading_prompt

def grade_llm_responses(data: pd.DataFrame, model_list: list, query_col: str='question', answer_col: str='answer', retries: int=5) -> pd.DataFrame:
    for model in model_list:
        data[f'LLMEVAL_{model}'] = 0.0

    total_tasks = len(data) * len(model_list)
    with tqdm(total=total_tasks, desc="Computing LLMEVAL scores") as pbar:
        for index, row in data.iterrows():
            question = row[query_col]
            answer = row[answer_col]
            for model in model_list:
                response = row[f'{model}_response']
                cur_prompt = biomedical_grading_prompt(question, answer, response)
                for attempt in range(retries):
                    try:
                        gpt4_eval = query_gpt4o(cur_prompt)
                        match = re.search(r"[-+]?[0-9]*\.?[0-9]+", gpt4_eval)
                        if match:
                            number = float(match.group(0))
                            if number in [-1, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                                data.at[index, f'LLMEVAL_{model}'] = number
                                break
                    except Exception as e:
                        if attempt == retries - 1:
                            raise ValueError(f"Failed to get a valid LLMEVAL evaluation score for model {model} on index {index} after {retries} retries. Last error: {e}")
                pbar.update(1)
    return data

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

    if "LLMEVAL" in METRICS:
        data = grade_llm_responses(data, MODELS)

    save_responses(local_scored_path, data)
    print(f"Responses scored and saved to {local_scored_path}.")


if __name__ == "__main__":
    main()
