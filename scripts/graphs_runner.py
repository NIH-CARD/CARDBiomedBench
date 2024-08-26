import argparse
from scripts import MODELS_DICT, METRICS_DICT
from scripts.scripts_utils import load_dataset
from scripts.generate_graphs.boxplot import plot_llmeval_boxplot, plot_metric_boxplot
from scripts.generate_graphs.heatmap import plot_metric_heatmap, plot_idk_heatmap
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
        plot_llmeval_boxplot(data, MODELS_DICT, "CARDBioBench LLMEVAL Boxplot", "results/")
        plot_metric_heatmap(data, "LLMEVAL", MODELS_DICT, "bio_category", "CARDBioBench LLMEVAL Bio Heatmap", "results/")
        plot_idk_heatmap(data, "LLMEVAL", MODELS_DICT, "bio_category", "CARDBioBench IDK Bio Heatmap", "results/")
    if "BLEU_ROUGE" in METRICS_DICT:
        plot_metric_boxplot(data, "bleu", MODELS_DICT, "CARDBioBench BLEU Boxplot", "results/")
        plot_metric_boxplot(data, "rouge1", MODELS_DICT, "CARDBioBench ROUGE1 Boxplot", "results/")
        plot_metric_boxplot(data, "rouge2", MODELS_DICT, "CARDBioBench ROUGE2 Boxplot", "results/")
        plot_metric_boxplot(data, "rougeL", MODELS_DICT, "CARDBioBench ROUGEL Boxplot", "results/")

if __name__ == "__main__":
    main()