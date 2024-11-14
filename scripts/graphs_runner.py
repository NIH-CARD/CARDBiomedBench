import argparse
from scripts.scripts_utils import load_dataset
from scripts.generate_graphs.boxplot import plot_metric_boxplot
from scripts.generate_graphs.heatmap import plot_heatmap
from scripts.generate_graphs.table import bioscore_performance_table, create_performance_table, style_dataframe
from scripts.generate_graphs.pie import plot_category_pie_chart
from scripts.generate_graphs.histogram import plot_token_histograms
from scripts.generate_graphs.statistics import statistics_txt
from scripts.generate_graphs.scatter import plot_safety_vs_quality
from scripts.generate_graphs.generate_graphs_utils import merge_model_responses, get_model_order, get_token_counts

def main():
    print("*** *** *** GRAPHS RUNNER *** *** ***")
    parser = argparse.ArgumentParser(description="Create graphs and tables on the benchmark results.")
    parser.add_argument('--qa_path', type=str, required=True, help='Path to the QA CSV file')
    parser.add_argument('--res_dir', type=str, required=True, help='Directory to the res CSV files')
    parser.add_argument('--scored_path', type=str, required=True, help='Path to the compiled results file')
    parser.add_argument('--models_to_process', nargs='+', required=True, help='List of models to process')
    parser.add_argument('--metrics_to_use', nargs='+', required=True, help='List of metrics to process')
    args = parser.parse_args()

    qa_path = args.qa_path
    res_dir = args.res_dir
    scored_path = args.scored_path
    models_list = args.models_to_process
    metrics_list = args.metrics_to_use

    merge_model_responses(qa_path, f'{res_dir}by_model', scored_path)
    print("*** Model responses merged ***")
    
    data = load_dataset(scored_path)
    if data.empty:
        print("No data to process. Exiting.")
        return
    
    # Compute and add token count columns for question, answer, and each model_response
    data = get_token_counts(data, models_list)
    print("*** Token counts computed ***")

    # Dataset statistics txt file
    statistics_txt(data, models=models_list, title="statistics", save_path=res_dir)
    print("*** Dataset statistics generated ***")

    # Dataset distribution visualizations
    plot_category_pie_chart(data, category="bio_category", title="Bio Category Donut", save_path=res_dir, color_flag=1)
    print("*** Bio Category Donut Chart Completed ***")
    
    plot_category_pie_chart(data, category="reasoning_category", title="Reasoning Category Donut", save_path=res_dir, color_flag=2)
    print("*** SQL Category Donut Chart Completed ***")
    
    plot_token_histograms(data, text_col="question", color="dodgerblue", title="Question", save_path=res_dir)
    print("*** Question Token Histogram Completed ***")
    
    plot_token_histograms(data, text_col="answer", color="deeppink", title="Answer", save_path=res_dir)
    print("*** Answer Token Histogram Completed ***")

    # Metric visualizations
    if "BioScore" in metrics_list:
        bioscore_model_order = get_model_order(data, "BioScore", models_list)
        plot_safety_vs_quality(data, 'BioScore', models_list, 'Safety Rate vs. Response Quality Rate', res_dir)
        print("*** Safety vs Quality Scatterplot Completed ***")

        plot_metric_boxplot(data, "BioScore", models_list, bioscore_model_order, "BioScore Boxplot", res_dir)
        print("*** BioScore Boxplot Completed ***")
        
        # BioScore Heatmaps
        plot_heatmap(data=data, metric='BioScore', models=models_list, model_order=bioscore_model_order, category='bio_category',
            title='BioScore Bio Heatmap', save_path=res_dir, calculation_type='mean', threshold=5)
        print("*** BioScore Bio Heatmap Completed ***")

        plot_heatmap(data=data, metric='BioScore', models=models_list, model_order=bioscore_model_order, category='reasoning_category',
            title='BioScore Reasoning Heatmap', save_path=res_dir, calculation_type='mean', threshold=5)
        print("*** BioScore Reasoning Heatmap Completed ***")

        # AR Heatmaps
        plot_heatmap(data=data, metric='BioScore', models=models_list, model_order=bioscore_model_order, category='bio_category',
            title='Abstention Rate Bio Heatmap', save_path=res_dir, calculation_type='percentage_idk')
        print("*** Abstention Rate Bio Heatmap Completed ***")

        plot_heatmap(data=data, metric='BioScore', models=models_list, model_order=bioscore_model_order, category='reasoning_category',
            title='Abstention Rate Reasoning Heatmap', save_path=res_dir, calculation_type='percentage_idk')
        print("*** Abstention Rate Reasoning Heatmap Completed ***")

        # Quality Rate Heatmaps
        plot_heatmap(data=data, metric='BioScore', models=models_list, model_order=bioscore_model_order, category='bio_category',
            title='Quality Rate Bio Heatmap', save_path=res_dir, calculation_type='quality_rate', threshold=5)
        print("*** Quality Rate Bio Heatmap Completed ***")

        plot_heatmap(data=data, metric='BioScore', models=models_list, model_order=bioscore_model_order, category='reasoning_category',
            title='Quality Rate Reasoning Heatmap', save_path=res_dir, calculation_type='quality_rate', threshold=5)
        print("*** Quality Rate Reasoning Heatmap Completed ***")

        # Safety Rate Heatmaps
        plot_heatmap(data=data, metric='BioScore', models=models_list, model_order=bioscore_model_order, category='bio_category',
            title='Safety Rate Bio Heatmap', save_path=res_dir, calculation_type='safety_rate', threshold=5)
        print("*** Safety Rate Bio Heatmap Completed ***")

        plot_heatmap(data=data, metric='BioScore', models=models_list, model_order=bioscore_model_order, category='reasoning_category',
            title='Safety Rate Reasoning Heatmap', save_path=res_dir, calculation_type='safety_rate', threshold=5)
        print("*** Safety Rate Reasoning Heatmap Completed ***")

    if "BLEU_ROUGE_BERT" in metrics_list:
        nlp_metrics = ['BLEU', 'ROUGE2', 'ROUGEL', 'BERTScore']
        for metric in nlp_metrics:
            plot_metric_boxplot(data, metric, models_list, bioscore_model_order, f"{metric} Boxplot", res_dir)
        print("*** BLEU, ROUGE, and BERTScore Completed ***")

    # Generate performance tables
    if "BioScore" in metrics_list:
        performance_table = bioscore_performance_table(data, models_list)
        print("*** BioScore Performance Table Created ***")
        style_dataframe(performance_table, "All BioScore Metrics", res_dir)
        print("*** BioScore Table Styled and Saved ***")

    if "BLEU_ROUGE_BERT" in metrics_list:
        performance_table = create_performance_table(data, nlp_metrics, models_list)
        print("*** NLP Performance Table Created ***")
        style_dataframe(performance_table, "All NLP Metrics", res_dir)
        print("*** NLP Table Styled and Saved ***")

    print("*** *** *** GRAPHS RUNNER COMPLETED *** *** ***")

if __name__ == "__main__":
    main()
