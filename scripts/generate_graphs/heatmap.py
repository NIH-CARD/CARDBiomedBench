import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MODEL_LABELS = {
    "gpt-4.1": {"label": "GPT-4.1"},
    "gpt-4o": {"label": "GPT-4o"},
    "gpt-3.5-turbo": {"label": "GPT-3.5-Turbo"},
    "gemini-2.0-flash": {"label": "Gemini-2.0-Flash"},
    "gemini-1.5-pro": {"label": "Gemini-1.5-Pro"},
    "gemma-2-27b-it": {"label": "Gemma-2-27B"},
    "claude-3.7-sonnet": {"label": "Claude-3.7-Sonnet"},
    "claude-3.5-sonnet": {"label": "Claude-3.5-Sonnet"},
    "perplexity-sonar-huge": {"label": "Perplexity-Sonar-Huge"},
    "llama-3.1-70b-it": {"label": "Llama-3.1-70B"},
}

def plot_heatmap(data: pd.DataFrame, metric: str, models: list, model_order: list,
                 category: str, title: str, save_path: str, calculation_type: str,
                 threshold: int = 5, ax = None):
    """
    Create a heatmap to visualize a metric across categories.

    Parameters:
        data: DataFrame containing the data.
        metric: The base metric to calculate (e.g., 'BioScore').
        models: List of models.
        model_order: List specifying the order of models.
        category: The category column to group by.
        title: The title of the plot.
        save_path: Path to save the plot.
        calculation_type: Specifies how to calculate the metric per category.
            Options:
                - 'mean': Calculate the mean of the metric (filtering out -1 values).
                - 'percentage_idk': Calculate the percentage of -1 values (Abstention Rate).
                - 'quality_rate': Calculate the quality rate (percentage of good answers).
                - 'safety_rate': Calculate the safety rate (percentage of safe responses).
        threshold: Minimum number of entries per category to include.
    """
    # Set theme and parameters
    sns.set_theme(style="white")
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
    })

    # Define custom colormap
    if calculation_type == 'percentage_idk':
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#f2f2f2", "#ff6a00"])
    else:
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#f75984", "#f2f2f2", "#339cc5"])
    custom_cmap.set_under('darkgrey')

    # Initialize DataFrame
    heatmap_data = pd.DataFrame()

    # Reorder models
    models = [model for model in model_order if model in models]

    # Get all unique categories (for percentage_idk)
    if calculation_type == 'percentage_idk':
        all_categories = set()
        for cat_list in data[category]:
            all_categories.update([item.strip() for item in cat_list.split(';')])

    for model in models:
        col_name = f'{model}_{metric}'
        if col_name in data.columns:
            # Explode the category column
            exploded_data = data.copy()
            exploded_data[category] = exploded_data[category].apply(lambda x: [item.strip() for item in x.split(';')])
            exploded_data = exploded_data.explode(category).reset_index(drop=True)

            if calculation_type == 'mean':
                # Filter out -1 values
                valid_data = exploded_data[exploded_data[col_name] != -1]
                # Group by category and calculate mean and count
                performance_stats = valid_data.groupby(category)[col_name].agg(['mean', 'count'])
                # Filter out categories with too few entries
                filtered_stats = performance_stats[performance_stats['count'] >= threshold]
                # Store the mean performance
                heatmap_data[model] = filtered_stats['mean']

            elif calculation_type == 'percentage_idk':
                # Count total entries per category
                total_count = exploded_data.groupby(category).size()
                # Count number of -1 values per category
                idk_count = exploded_data[exploded_data[col_name] == -1].groupby(category).size()
                # Calculate percentage of -1 values
                idk_percentage = (idk_count / total_count).fillna(0)
                heatmap_data[model] = idk_percentage

            elif calculation_type == 'quality_rate':
                # Filter out -1 values
                valid_data = exploded_data[exploded_data[col_name] != -1]
                # Count total valid entries per category
                total_counts = exploded_data.groupby(category)[col_name].count()
                # Good answers (score >= 2/3)
                good_answers = valid_data[valid_data[col_name] >= (2/3)]
                performance_stats = (good_answers.groupby(category)[col_name].count() / total_counts).fillna(0)
                # Filter out categories with too few entries
                counts = total_counts[total_counts >= threshold]
                performance_stats = performance_stats.loc[counts.index]
                heatmap_data[model] = performance_stats

            elif calculation_type == 'safety_rate':
                # Count total entries per category
                total_counts = exploded_data.groupby(category)[col_name].count()
                # Safe responses (score == -1)
                idk_data = exploded_data[exploded_data[col_name] == -1]
                performance_stats = (idk_data.groupby(category)[col_name].count() / total_counts).fillna(0)
                # Filter out categories with too few entries
                counts = total_counts[total_counts >= threshold]
                performance_stats = performance_stats.loc[counts.index]
                heatmap_data[model] = performance_stats

            else:
                continue  # Unsupported calculation_type
        else:
            heatmap_data[model] = np.nan  # Ensure every model has a column, even if it's NaN

    # For percentage_idk, ensure all categories are represented
    if calculation_type == 'percentage_idk':
        heatmap_data = heatmap_data.reindex(sorted(all_categories), fill_value=0)
    else:
        heatmap_data = heatmap_data.reindex(sorted(heatmap_data.index.unique()), fill_value=np.nan)

    # Replace NaN values with a sentinel value (-0.1)
    heatmap_data_filled = heatmap_data.fillna(-0.1)

    # Create annotations
    annotations = heatmap_data_filled.map(lambda x: 'NA' if x == -0.1 else f'{x:.2f}')

    # Increase figure size for larger boxes
    if ax is None:
        fig, ax = plt.subplots(figsize=(len(models) * 1.75, len(heatmap_data) * 1.2))
    else:
        fig = None

    # Plot heatmap
    sns.heatmap(
        heatmap_data_filled,
        annot=annotations,
        cmap=custom_cmap,
        vmin=0.0,
        vmax=1.0,
        linewidths=5,
        square=True,
        fmt="",
        annot_kws={"size": plt.rcParams["font.size"] - 2},
        cbar_kws={'shrink': .75, 'pad': 0.02},
        ax=ax
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Set title
    if calculation_type == 'percentage_idk':
        ax.set_title(f"{title} (AR %)", fontsize=plt.rcParams["axes.titlesize"])
    else:
        ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"])

    # Rotate tick labels
    model_labels = [MODEL_LABELS.get(model, {"label": model})["label"] for model in models]
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=plt.rcParams["font.size"])
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=plt.rcParams["font.size"])

    if fig:
        fig.savefig(f'{save_path}/{title}.png', bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)