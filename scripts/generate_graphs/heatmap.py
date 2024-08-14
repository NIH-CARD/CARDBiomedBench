import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_metric_heatmap(data: pd.DataFrame, metric: str, models: dict, category: str, title: str, save_path: str):
    """Create a heatmap to visualize average performance for a single metric across categories, filtering out -1 values."""
    sns.set_theme(style="white")
    
    # Define a custom gradient colormap with three colors
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#f75984", "#f2f2f2", "#339cc5"])
    
    # Initialize a DataFrame to store average performance per category per model
    heatmap_data = pd.DataFrame()
    
    for model in models:
        col_name = f'{model}_{metric}'
        if col_name in data.columns:
            # Explode the category column to handle multiple labels per entry
            exploded_data = data.copy()
            exploded_data[category] = exploded_data[category].apply(lambda x: [item.strip() for item in x.split(';')])
            exploded_data = exploded_data.explode(category).reset_index(drop=True)
            
            # Filter out -1 values
            exploded_data = exploded_data[exploded_data[col_name] != -1]
            
            # Calculate the mean for each category
            avg_performance = exploded_data.groupby(category)[col_name].mean()
            heatmap_data[model] = avg_performance
        else:
            heatmap_data[model] = np.nan  # Ensure every model has a column, even if it's NaN
    
    # Ensure that all models have the same set of categories
    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.index.unique()), fill_value=np.nan)
    
    # Increase figure size for larger boxes
    plt.figure(figsize=(len(models) * 2, len(heatmap_data) * 1.2))
    ax = sns.heatmap(heatmap_data, annot=True, cmap=custom_cmap, linewidths=5, square=True, fmt=".2f", annot_kws={"size": 14}, cbar_kws={'shrink': .75})
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.title(f"{title}", fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}.png')
    plt.close()

def plot_idk_heatmap(data: pd.DataFrame, metric: str, models: dict, category: str, title: str, save_path: str):
    """Create a heatmap to visualize the counts of -1 values across models."""
    sns.set_theme(style="white")

    # Define a custom gradient colormap with two colors
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#f2f2f2", "#fa954d"])
    
    # Initialize a DataFrame to store counts of -1 values per category per model
    heatmap_data = pd.DataFrame()

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
            
            # Count the number of -1 values for each category
            idk_count = exploded_data[exploded_data[col_name] == -1].groupby(category).size()
            heatmap_data[model] = idk_count
        else:
            heatmap_data[model] = 0  # Ensure every model has a column, even if it's 0
    
    # Ensure that all categories are represented in the heatmap data
    heatmap_data = heatmap_data.reindex(sorted(all_categories), fill_value=0)

    # Fill empty with 0
    heatmap_data = heatmap_data.fillna(0)

    # Increase figure size for larger boxes
    plt.figure(figsize=(len(models) * 2, len(heatmap_data) * 1.2))
    ax = sns.heatmap(heatmap_data, annot=True, cmap=custom_cmap, linewidths=5, square=True, fmt="g", annot_kws={"size": 14}, cbar_kws={'shrink': .75}, vmin=0)
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.title(f"{title}", fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}_idk.png')
    plt.close()