import argparse
from scripts import MODELS_DICT, METRICS_DICT
from scripts.scripts_utils import load_dataset
from scripts.generate_graphs.boxplot import plot_metric_boxplot

def main():
    parser = argparse.ArgumentParser(description="Create graphs and tables on the benchmark results.")
    parser.add_argument('--local_scored_path', type=str, required=True, help='Local path to the scored CSV file')
    args = parser.parse_args()

    local_scored_path = args.local_scored_path

    data = load_dataset(local_scored_path)
    if data.empty:
        print("No data to process. Exiting.")
        return
    
    if "LLMEVAL" in METRICS_DICT:
        plot_metric_boxplot(data, "LLMEVAL", MODELS_DICT, "CARDBench LLMEVAL", "results/")


if __name__ == "__main__":
    main()