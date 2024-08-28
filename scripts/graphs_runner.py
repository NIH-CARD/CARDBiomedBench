import argparse
from scripts import MODELS_DICT, METRICS_DICT
from scripts.scripts_utils import load_dataset
from scripts.generate_graphs.boxplot import plot_metric_boxplot
from scripts.generate_graphs.heatmap import plot_metric_heatmap, plot_idk_heatmap
from scripts.generate_graphs.table import create_performance_table, style_dataframe
from scripts.generate_graphs.pie import plot_category_pie_chart
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
    
    # Dataset distribution visualizations
    plot_category_pie_chart(data, "bio_category", "Bio Category Pie", "results/")

    # Metric visualizations
    metrics_list = []
    if "BioScore" in METRICS_DICT:
        plot_metric_boxplot(data, "BioScore", MODELS_DICT, "BioScore Boxplot", "results/")
        plot_metric_heatmap(data, "BioScore", MODELS_DICT, "bio_category", "BioScore Bio Heatmap", "results/")
        plot_idk_heatmap(data, "BioScore", MODELS_DICT, "bio_category", "IDK Bio Heatmap", "results/")
        metrics_list += ["BioScore"]
    if "BLEU_ROUGE_BERT" in METRICS_DICT:
        nlp_metrics = ['BLEU', 'ROUGE1', 'ROUGE2', 'ROUGEL', 'BERTScore']
        for metric in nlp_metrics:
            plot_metric_boxplot(data, metric, MODELS_DICT, f"{metric} Boxplot", "results/")
        metrics_list += nlp_metrics
    performance_table = create_performance_table(data, metrics_list, MODELS_DICT)
    style_dataframe(performance_table, "All Metrics", "results/")


if __name__ == "__main__":
    main()