"""
graphs_runner.py

This script generates graphs and tables based on the benchmark results.
It processes the data, computes metrics, and produces visualizations and performance tables.
"""

import argparse

from scripts.generate_graphs.boxplot import plot_metric_boxplot
from scripts.generate_graphs.generate_graphs_utils import (
    get_model_order,
    get_token_counts,
    merge_model_responses,
)
from scripts.generate_graphs.heatmap import plot_heatmap
from scripts.generate_graphs.histogram import plot_token_histograms
from scripts.generate_graphs.pie import plot_category_pie_chart
from scripts.generate_graphs.scatter import plot_safety_vs_quality
from scripts.generate_graphs.statistics import statistics_txt
from scripts.generate_graphs.table import (
    bioscore_performance_table,
    create_performance_table,
    style_dataframe,
)
from scripts.scripts_utils import load_dataset


def main():
    """
    Main function to generate graphs and tables from benchmark results.
    """
    print("=== GRAPHS RUNNER STARTED ===")
    parser = argparse.ArgumentParser(description="Create graphs and tables on the benchmark results.")
    parser.add_argument('--qa_path', type=str, required=True, 
        help='Path to the QA CSV file'
    )
    parser.add_argument('--res_dir', type=str, required=True, 
        help='Directory containing the result CSV files'
    )
    parser.add_argument('--scored_path', type=str, required=True, 
        help='Path to the compiled results file'
    )
    parser.add_argument('--models_to_process', nargs='+', required=True, 
        help='List of models to process'
    )
    parser.add_argument('--metrics_to_use', nargs='+', required=True, 
        help='List of metrics to process'
    )
    args = parser.parse_args()

    qa_path: str = args.qa_path
    res_dir: str = args.res_dir
    scored_path: str = args.scored_path
    models_list: list = args.models_to_process
    metrics_list: list = args.metrics_to_use

    # Merge model responses
    merge_model_responses(qa_path, f'{res_dir}by_model', scored_path)
    print("Model responses merged successfully.")

    data = load_dataset(scored_path)
    if data.empty:
        print("❌ No data to process. Exiting.")
        return

    # Compute and add token count columns for question, answer, and each model response
    data = get_token_counts(data, models_list)
    print("Token counts computed.")

    # Generate dataset statistics text file
    statistics_txt(data, models=models_list, title="statistics", save_path=res_dir)
    print("Dataset statistics generated.")

    # Generate dataset distribution visualizations
    plot_category_pie_chart(
        data,
        category="bio_category",
        title="Bio Category Distribution",
        save_path=res_dir,
        color_flag=1,
    )
    print("Bio Category Donut Chart created.")

    plot_category_pie_chart(
        data,
        category="reasoning_category",
        title="Reasoning Category Distribution",
        save_path=res_dir,
        color_flag=2,
    )
    print("Reasoning Category Donut Chart created.")

    plot_token_histograms(
        data,
        text_col="question",
        color="dodgerblue",
        title="Question Token Histogram",
        save_path=res_dir,
    )
    print("Question Token Histogram created.")

    plot_token_histograms(
        data,
        text_col="answer",
        color="deeppink",
        title="Answer Token Histogram",
        save_path=res_dir,
    )
    print("Answer Token Histogram created.")

    # Metric visualizations
    if "BioScore" in metrics_list:
        bioscore_model_order = get_model_order(data, "BioScore", models_list)

        plot_safety_vs_quality(
            data,
            metric='BioScore',
            models=models_list,
            title='Safety Rate vs. Response Quality Rate',
            save_path=res_dir,
        )
        print("Safety vs. Quality Scatterplot created.")

        plot_metric_boxplot(
            data,
            metric="BioScore",
            models=models_list,
            model_order=bioscore_model_order,
            title="BioScore Boxplot",
            save_path=res_dir,
        )
        print("BioScore Boxplot created.")

        # BioScore Heatmaps
        plot_heatmap(
            data=data,
            metric='BioScore',
            models=models_list,
            model_order=bioscore_model_order,
            category='bio_category',
            title='BioScore by Bio Category Heatmap',
            save_path=res_dir,
            calculation_type='mean',
            threshold=5,
        )
        print("BioScore Bio Category Heatmap created.")

        plot_heatmap(
            data=data,
            metric='BioScore',
            models=models_list,
            model_order=bioscore_model_order,
            category='reasoning_category',
            title='BioScore by Reasoning Category Heatmap',
            save_path=res_dir,
            calculation_type='mean',
            threshold=5,
        )
        print("BioScore Reasoning Category Heatmap created.")

        # Abstention Rate Heatmaps
        plot_heatmap(
            data=data,
            metric='BioScore',
            models=models_list,
            model_order=bioscore_model_order,
            category='bio_category',
            title='Abstention Rate by Bio Category Heatmap',
            save_path=res_dir,
            calculation_type='percentage_idk',
        )
        print("Abstention Rate Bio Category Heatmap created.")

        plot_heatmap(
            data=data,
            metric='BioScore',
            models=models_list,
            model_order=bioscore_model_order,
            category='reasoning_category',
            title='Abstention Rate by Reasoning Category Heatmap',
            save_path=res_dir,
            calculation_type='percentage_idk',
        )
        print("Abstention Rate Reasoning Category Heatmap created.")

        # Quality Rate Heatmaps
        plot_heatmap(
            data=data,
            metric='BioScore',
            models=models_list,
            model_order=bioscore_model_order,
            category='bio_category',
            title='Quality Rate by Bio Category Heatmap',
            save_path=res_dir,
            calculation_type='quality_rate',
            threshold=5,
        )
        print("Quality Rate Bio Category Heatmap created.")

        plot_heatmap(
            data=data,
            metric='BioScore',
            models=models_list,
            model_order=bioscore_model_order,
            category='reasoning_category',
            title='Quality Rate by Reasoning Category Heatmap',
            save_path=res_dir,
            calculation_type='quality_rate',
            threshold=5,
        )
        print("Quality Rate Reasoning Category Heatmap created.")

        # Safety Rate Heatmaps
        plot_heatmap(
            data=data,
            metric='BioScore',
            models=models_list,
            model_order=bioscore_model_order,
            category='bio_category',
            title='Safety Rate by Bio Category Heatmap',
            save_path=res_dir,
            calculation_type='safety_rate',
            threshold=5,
        )
        print("Safety Rate Bio Category Heatmap created.")

        plot_heatmap(
            data=data,
            metric='BioScore',
            models=models_list,
            model_order=bioscore_model_order,
            category='reasoning_category',
            title='Safety Rate by Reasoning Category Heatmap',
            save_path=res_dir,
            calculation_type='safety_rate',
            threshold=5,
        )
        print("Safety Rate Reasoning Category Heatmap created.")

    if "BLEU_ROUGE_BERT" in metrics_list:
        nlp_metrics = ['BLEU', 'ROUGE2', 'ROUGEL', 'BERTScore']
        for metric in nlp_metrics:
            plot_metric_boxplot(
                data,
                metric=metric,
                models=models_list,
                model_order=bioscore_model_order,
                title=f"{metric} Boxplot",
                save_path=res_dir,
            )
            print(f"{metric} Boxplot created.")

    # Generate performance tables
    if "BioScore" in metrics_list:
        performance_table = bioscore_performance_table(data, models_list)
        print("BioScore Performance Table created.")
        style_dataframe(performance_table, "All BioScore Metrics", res_dir)
        print("BioScore Table styled and saved.")

    if "BLEU_ROUGE_BERT" in metrics_list:
        performance_table = create_performance_table(data, nlp_metrics, models_list)
        print("NLP Performance Table created.")
        style_dataframe(performance_table, "All NLP Metrics", res_dir)
        print("NLP Table styled and saved.")

    print("=== GRAPHS RUNNER COMPLETED ===")


if __name__ == "__main__":
    main()