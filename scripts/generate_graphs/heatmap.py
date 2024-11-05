import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_heatmap(data: pd.DataFrame, metric: str, models: dict, model_order: list,
                 category: str, title: str, save_path: str, calculation_type: str,
                 threshold: int = 5):
    """
    Create a heatmap to visualize a metric across categories.

    Parameters:
        data: DataFrame containing the data.
        metric: The base metric to calculate (e.g., 'BioScore').
        models: Dictionary of models.
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
    models = {model: models[model] for model in model_order}

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
    plt.figure(figsize=(len(models) * 1.75, len(heatmap_data) * 1.2))

    # Plot heatmap
    ax = sns.heatmap(
        heatmap_data_filled,
        annot=annotations,
        cmap=custom_cmap,
        vmin=0.0,
        vmax=1.0,
        linewidths=5,
        square=True,
        fmt="",
        annot_kws={"size": 18},
        cbar_kws={'shrink': .75}
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Set title
    if calculation_type == 'percentage_idk':
        plt.title(f"{title} (AR %)", fontsize=18)
    else:
        plt.title(f"{title}", fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}.png')
    plt.close()