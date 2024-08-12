import argparse
from scripts import MODELS_DICT, METRICS_DICT
from scripts.scripts_utils import load_dataset
from scripts.generate_graphs.boxplot import plot_metric_boxplot
from scripts.generate_graphs.generate_graphs_utils import merge_model_responses

def main():
    parser = argparse.ArgumentParser(description="Create graphs and tables on the benchmark results.")
    parser.add_argument('--res_dir', type=str, required=True, help='Directory to the res CSV files')
    parser.add_argument('--scored_path', type=str, required=True, help='Path to the compiled results file')
    args = parser.parse_args()

    res_dir = args.res_dir
    scored_path = args.scored_path

    merge_model_responses(res_dir, scored_path)
    
    data = load_dataset(scored_path)
    if data.empty:
        print("No data to process. Exiting.")
        return
    
    if "LLMEVAL" in METRICS_DICT:
        plot_metric_boxplot(data, "LLMEVAL", MODELS_DICT, "CARDBench LLMEVAL", "results/")


if __name__ == "__main__":
    main()