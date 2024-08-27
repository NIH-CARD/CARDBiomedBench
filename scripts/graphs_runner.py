import argparse
from scripts import MODELS_DICT, METRICS_DICT
from scripts.scripts_utils import load_dataset
from scripts.generate_graphs.boxplot import plot_llmeval_boxplot, plot_metric_boxplot
from scripts.generate_graphs.heatmap import plot_metric_heatmap, plot_idk_heatmap
from scripts.generate_graphs.table import create_performance_table, style_dataframe
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
    
    metrics_list = []
    if "LLMEVAL" in METRICS_DICT:
        plot_llmeval_boxplot(data, MODELS_DICT, "CARDBioBench LLMEVAL Boxplot", "results/")
        plot_metric_heatmap(data, "LLMEVAL", MODELS_DICT, "bio_category", "CARDBioBench LLMEVAL Bio Heatmap", "results/")
        plot_idk_heatmap(data, "LLMEVAL", MODELS_DICT, "bio_category", "CARDBioBench IDK Bio Heatmap", "results/")
        metrics_list += ["LLMEVAL"]
    if "BLEU_ROUGE_BERT" in METRICS_DICT:
        BRB_list = ['BLEU', 'ROUGE1', 'ROUGE2', 'ROUGEL', 'BERTprecision', 'BERTrecall', 'BERTf1']
        for metric in BRB_list:
            plot_metric_boxplot(data, metric, MODELS_DICT, f"CARDBioBench {metric} Boxplot", "results/")
        metrics_list += BRB_list
    metrics_list = ["LLMEVAL", "BLEU", "ROUGEL", "BERTf1"] # TODO Delete??
    performance_table = create_performance_table(data, metrics_list, MODELS_DICT)
    style_dataframe(performance_table, "CARDBioBench All Metrics", "results/")


if __name__ == "__main__":
    main()