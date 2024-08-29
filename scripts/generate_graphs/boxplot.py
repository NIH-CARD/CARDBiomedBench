import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_model_order(data: pd.DataFrame, metric: str, models: dict) -> list:
    """Get the order of models based on the median first, then IDK %, then spread (IQR) of the metric values."""
    model_stats = []
    
    for model in models:
        col_name = f'{model}_{metric}'
        if col_name in data.columns:
            model_data = data[col_name].copy()
            idk_count = (model_data == -1).sum()
            total_count = len(model_data)
            idk_rate = idk_count / total_count
            
            if metric == "BioScore":
                model_data = model_data[model_data != -1]  # Exclude -1 values for BioScore
            
            median_val = model_data.median()
            spread_val = model_data.quantile(0.75) - model_data.quantile(0.25)  # IQR for spread
            
            model_stats.append((model, median_val, idk_rate, spread_val))
    
    # Sort by median (descending), then by IDK rate (ascending), then by spread (ascending)
    model_stats_sorted = sorted(model_stats, key=lambda x: (-x[1], x[2], x[3]))

    # Extract the sorted model names
    sorted_models = [model[0] for model in model_stats_sorted]
    
    return sorted_models

def plot_metric_boxplot(data: pd.DataFrame, metric: str, models: dict, title: str, save_path: str):
    """Create a box and whisker plot to visualize performance for the BioScore metric, handling -1 values separately."""
    colors = ['#ADD8E6', '#FFB6C1', '#DDA0DD', '#87CEEB', '#FF69B4', '#BA55D3']
    sns.set_style("whitegrid")
    sns.set_context("talk")

    plt.figure(figsize=(20, 10))
    plt.axhline(y=0, color='k', linestyle=':', linewidth=2)

    melted_data = pd.DataFrame()
    idk_counts = {}

    # Get the new order of models and reorder the models dictionary
    ordered_model_names = get_model_order(data, metric, models)
    models = {model: models[model] for model in ordered_model_names}

    for model in models:
        col_name = f'{model}_{metric}'
        if col_name in data.columns:
            model_data = data[[col_name]].copy()
            model_data['Model'] = model
            model_data.rename(columns={col_name: metric}, inplace=True)
            
            if metric == "BioScore":
                idk_count = (model_data[metric] == -1).sum()
                idk_counts[model] = idk_count
                model_data = model_data[model_data[metric] != -1]

            melted_data = pd.concat([melted_data, model_data], axis=0)
        else:
            idk_counts[model] = 0  # Ensure every model has a count entry
            melted_data = pd.concat([melted_data, pd.DataFrame({metric: [np.nan], 'Model': model})], axis=0)
        
    ax = plt.gca()
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=0, fontsize=20, ha='center')
    
    # Plot the boxplot for metric
    sns.boxplot(
        x='Model', 
        y=metric, 
        data=melted_data, 
        palette=colors[:len(models)], 
        linewidth=2,
        hue='Model',
        dodge=False,
        ax=ax,
        legend=False
    )
    
    # Plot the stripplot for the metric
    sns.stripplot(
        x='Model', 
        y=metric, 
        data=melted_data, 
        jitter=True, 
        size=8, 
        edgecolor='black', 
        color='white', 
        linewidth=1, 
        alpha=0.3,
        ax=ax
    )
    
    if metric == "BioScore":
        for model in models:
            count = idk_counts.get(model, 0)
            model_index = list(models.keys()).index(model)
            percent = (count / len(data)) * 100
            plt.text(model_index, -0.125, f'({percent:.2f}%)', ha='center', va='center', fontsize=16, color='red')
        
    yticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    yticks = [yt / 3.0 for yt in yticks]
    
    plt.ylim(-0.05, max(yticks) + 0.05)
    plt.yticks(yticks, [f'{yt:.2f}' for yt in yticks])

    plt.xlabel("Model", fontsize=22, fontweight='bold')
    plt.ylabel(metric, fontsize=22, fontweight='bold')
    plt.title(f"{title} (n = {len(data)})", fontsize=24)

    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}')
    plt.close()