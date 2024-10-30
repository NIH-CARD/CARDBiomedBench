import argparse
from scripts import MODELS_DICT, METRICS_DICT
from scripts.scripts_utils import load_dataset
from scripts.generate_graphs.boxplot import plot_metric_boxplot, plot_template_boxplot
from scripts.generate_graphs.heatmap import plot_heatmap
from scripts.generate_graphs.table import bioscore_performance_table, create_performance_table, style_dataframe
from scripts.generate_graphs.pie import plot_category_pie_chart
from scripts.generate_graphs.histogram import plot_token_histograms
from scripts.generate_graphs.statistics import statistics_txt
from scripts.generate_graphs.scatter import plot_scatterplot, plot_safety_vs_quality
from scripts.generate_graphs.generate_graphs_utils import merge_model_responses, get_model_order, get_token_counts

def main():
    print("*** *** *** GRAPHS RUNNER *** *** ***")
    parser = argparse.ArgumentParser(description="Create graphs and tables on the benchmark results.")
    parser.add_argument('--qa_path', type=str, required=True, help='Path to the QA CSV file')
    parser.add_argument('--res_dir', type=str, required=True, help='Directory to the res CSV files')
    parser.add_argument('--scored_path', type=str, required=True, help='Path to the compiled results file')
    parser.add_argument('--template', type=str, required=True, help="Whether to run template-based sampling")
    args = parser.parse_args()

    qa_path = args.qa_path
    res_dir = args.res_dir
    scored_path = args.scored_path
    template_flag = args.template
    
    merge_model_responses(qa_path, f'{res_dir}/by_model', scored_path, template_flag)
    print("*** Model responses merged ***")
    
    data = load_dataset(scored_path)
    if data.empty:
        print("No data to process. Exiting.")
        return
    
    # Compute and add token count columns for question, answer, and each model_response
    data = get_token_counts(data, MODELS_DICT)
    print("*** Token counts computed ***")

    # Dataset statistics txt file
    statistics_txt(data, models=MODELS_DICT, title="statistics", save_path=res_dir)
    print("*** Dataset statistics generated ***")

    # Dataset distribution visualizations
    plot_category_pie_chart(data, category="Bio_Category", title="Bio Category Donut", save_path=res_dir, color_flag=1)
    print("*** Bio Category Donut Chart Completed ***")
    
    plot_category_pie_chart(data, category="SQL_Category", title="Reasoning Category Donut", save_path=res_dir, color_flag=2)
    print("*** SQL Category Donut Chart Completed ***")
    
    plot_token_histograms(data, text_col="question", color="dodgerblue", title="Question", save_path=res_dir)
    print("*** Question Token Histogram Completed ***")
    
    plot_token_histograms(data, text_col="answer", color="deeppink", title="Answer", save_path=res_dir)
    print("*** Answer Token Histogram Completed ***")

    # Metric visualizations
    metrics_list = []
    if "BioScore" in METRICS_DICT:
        bioscore_model_order = get_model_order(data, "BioScore", MODELS_DICT)
        plot_safety_vs_quality(data, 'BioScore', MODELS_DICT, 'Safety Rate vs. Response Quality Rate', res_dir)
        print("*** Safety vs Quality Scatterplot Completed ***")

        plot_metric_boxplot(data, "BioScore", MODELS_DICT, bioscore_model_order, "BioScore Boxplot", res_dir)
        print("*** BioScore Boxplot Completed ***")
        
        # BioScore Heatmaps
        plot_heatmap(data=data, metric='BioScore', models=MODELS_DICT, model_order=bioscore_model_order, category='Bio_Category',
            title='BioScore Bio Heatmap', save_path=res_dir, calculation_type='mean', threshold=5)
        print("*** BioScore Bio Heatmap Completed ***")

        plot_heatmap(data=data, metric='BioScore', models=MODELS_DICT, model_order=bioscore_model_order, category='SQL_Category',
            title='BioScore Reasoning Heatmap', save_path=res_dir, calculation_type='mean', threshold=5)
        print("*** BioScore Reasoning Heatmap Completed ***")
        
        # AR Heatmaps
        plot_heatmap(data=data, metric='BioScore', models=MODELS_DICT, model_order=bioscore_model_order, category='Bio_Category',
            title='Abstention Rate Bio Heatmap', save_path=res_dir, calculation_type='percentage_idk')
        print("*** Abstention Rate Bio Heatmap Completed ***")

        plot_heatmap(data=data, metric='BioScore', models=MODELS_DICT, model_order=bioscore_model_order, category='SQL_Category',
            title='Abstention Rate Reasoning Heatmap', save_path=res_dir, calculation_type='percentage_idk')
        print("*** Abstention Rate Reasoning Heatmap Completed ***")

        # Quality Rate Heatmaps
        plot_heatmap(data=data, metric='BioScore', models=MODELS_DICT, model_order=bioscore_model_order, category='Bio_Category',
            title='Quality Rate Bio Heatmap', save_path=res_dir, calculation_type='quality_rate', threshold=5)
        print("*** Quality Rate Bio Heatmap Completed ***")

        plot_heatmap(data=data, metric='BioScore', models=MODELS_DICT, model_order=bioscore_model_order, category='SQL_Category',
            title='Quality Rate Reasoning Heatmap', save_path=res_dir, calculation_type='quality_rate', threshold=5)
        print("*** Quality Rate Reasoning Heatmap Completed ***")
        
        # Safety Rate Heatmaps
        plot_heatmap(data=data, metric='BioScore', models=MODELS_DICT, model_order=bioscore_model_order, category='Bio_Category',
            title='Safety Rate Bio Heatmap', save_path=res_dir, calculation_type='safety_rate', threshold=5)
        print("*** Safety Rate Bio Heatmap Completed ***")

        plot_heatmap(data=data, metric='BioScore', models=MODELS_DICT, model_order=bioscore_model_order, category='SQL_Category',
            title='Safety Rate Reasoning Heatmap', save_path=res_dir, calculation_type='safety_rate', threshold=5)
        print("*** Safety Rate Reasoning Heatmap Completed ***")
        
        plot_scatterplot(data, "response_token_count", "BioScore", MODELS_DICT, "Mean Token Count by BioScore Stdv", res_dir)
        print("*** BioScore Scatterplot By Model Completed ***")
        
        if template_flag == "true":
            plot_template_boxplot(data, "BioScore", "gpt-4o", "GPT-4o BioScore by Template Question", res_dir)
            print("*** GPT-4o BioScore by Template Boxplot Completed ***")
        
        metrics_list += ["BioScore"]

    if "BLEU_ROUGE_BERT" in METRICS_DICT:
        nlp_metrics = ['BLEU', 'ROUGE2', 'ROUGEL', 'BERTScore']
        for metric in nlp_metrics:
            nlp_model_order = get_model_order(data, metric, MODELS_DICT)
            plot_metric_boxplot(data, metric, MODELS_DICT, bioscore_model_order, f"{metric} Boxplot", res_dir)
        print("*** BLEU, ROUGE, and BERTScore Completed ***")
        metrics_list += nlp_metrics

    performance_table = bioscore_performance_table(data, MODELS_DICT)
    print("*** BioScore Performance Table Created ***")

    style_dataframe(performance_table, "All BioScore Metrics", res_dir)
    print("*** BioScore Table Styled and Saved ***")

    performance_table = create_performance_table(data, nlp_metrics, MODELS_DICT)
    print("*** NLP Performance Table Created ***")

    style_dataframe(performance_table, "All NLP Metrics", res_dir)
    print("*** NLP Table Styled and Saved ***")

    print("*** *** *** GRAPHS RUNNER COMPLETED *** *** ***")

if __name__ == "__main__":
    main()