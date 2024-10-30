import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_metric_heatmap(data: pd.DataFrame, metric: str, models: dict, model_order: list, category: str, title: str, save_path: str, threshold: int = 5):
    """Create a heatmap to visualize average performance for a single metric across categories, filtering out -1 values and categories with too few valid entries."""
    sns.set_theme(style="white")
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
    })
    
    # Define a custom gradient colormap with three colors
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#f75984", "#f2f2f2", "#339cc5"])
    custom_cmap.set_under('darkgrey')  # Color for 'NA' cells
    
    # Initialize a DataFrame to store average performance per category per model
    heatmap_data = pd.DataFrame()
    
    # Get the new order of models and reorder the models dictionary
    models = {model: models[model] for model in model_order}

    for model in models:
        col_name = f'{model}_{metric}'
        if col_name in data.columns:
            # Explode the category column to handle multiple labels per entry
            exploded_data = data.copy()
            exploded_data[category] = exploded_data[category].apply(lambda x: [item.strip() for item in x.split(';')])
            exploded_data = exploded_data.explode(category).reset_index(drop=True)
            
            # Filter out -1 values
            exploded_data = exploded_data[exploded_data[col_name] != -1]
            
            # Group by category and calculate the number of valid entries and the mean performance
            performance_stats = exploded_data.groupby(category)[col_name].agg(['mean', 'count'])
            
            # Filter out categories with fewer valid entries than the threshold
            filtered_stats = performance_stats[performance_stats['count'] >= threshold]
            
            # Store the mean performance in the heatmap data
            heatmap_data[model] = filtered_stats['mean']
        else:
            heatmap_data[model] = np.nan  # Ensure every model has a column, even if it's NaN
    
    # Ensure that all models have the same set of categories
    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.index.unique()), fill_value=np.nan)
    
    # Replace NaN values with a sentinel value (-0.1)
    heatmap_data_filled = heatmap_data.fillna(-0.1)
    
    # Create custom annotations, replacing sentinel value with 'NA'
    annotations = heatmap_data_filled.map(lambda x: 'NA' if x == -0.1 else f'{x:.2f}')
    
    # Increase figure size for larger boxes
    plt.figure(figsize=(len(models) * 1.75, len(heatmap_data) * 1.2))
    ax = sns.heatmap(
        heatmap_data_filled,
        annot=annotations,
        cmap=custom_cmap,
        vmin=0.0,
        vmax=1.0,
        linewidths=5,
        square=True,
        fmt="",
        annot_kws={"size": 14},
        cbar_kws={'shrink': .75}
    )
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.title(f"{title}", fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}.png')
    plt.close()

def plot_idk_heatmap(data: pd.DataFrame, metric: str, models: dict, model_order: list, category: str, title: str, save_path: str):
    """Create a heatmap to visualize the percentages of -1 values across models."""
    sns.set_theme(style="white")
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
    })

    # Define a custom gradient colormap with two colors
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#f2f2f2", "#ff6a00"])
    
    # Initialize a DataFrame to store percentages of -1 values per category per model
    heatmap_data = pd.DataFrame()

    # Get the new order of models and reorder the models dictionary
    models = {model: models[model] for model in model_order}

    # Get all unique categories to ensure they are present in the heatmap
    all_categories = set()
    for cat_list in data[category]:
        all_categories.update([item.strip() for item in cat_list.split(';')])

    for model in models:
        col_name = f'{model}_{metric}'
        if col_name in data.columns:
            # Explode the category column to handle multiple labels per entry
            exploded_data = data.copy()
            exploded_data[category] = exploded_data[category].apply(lambda x: [item.strip() for item in x.split(';')])
            exploded_data = exploded_data.explode(category).reset_index(drop=True)
            
            # Count the total number of entries for each category
            total_count = exploded_data.groupby(category).size()
            
            # Count the number of -1 values for each category
            idk_count = exploded_data[exploded_data[col_name] == -1].groupby(category).size()
            
            # Calculate the percentage of -1 values
            idk_percentage = (idk_count / total_count)
            heatmap_data[model] = idk_percentage
        else:
            heatmap_data[model] = 0  # Ensure every model has a column, even if it's 0
    
    # Ensure that all categories are represented in the heatmap data
    heatmap_data = heatmap_data.reindex(sorted(all_categories), fill_value=0)

    # Fill empty with 0
    heatmap_data = heatmap_data.fillna(0)

    # Increase figure size for larger boxes
    plt.figure(figsize=(len(models) * 1.75, len(heatmap_data) * 1.2))
    ax = sns.heatmap(heatmap_data, annot=True, cmap=custom_cmap, linewidths=5, square=True, fmt=".2f", annot_kws={"size": 14}, cbar_kws={'shrink': .75}, vmin=0, vmax=1)
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.title(f"{title} (AR %)", fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}.png')
    plt.close()