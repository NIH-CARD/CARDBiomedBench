import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.generate_graphs.wandb_logging import wandb_boxplot


def plot_metric_boxplot(data: pd.DataFrame, metric: str, models: dict, title: str, save_path: str):
    """Create a box and whisker plot to visualize performance for a single metric."""
    colors = ['#ADD8E6', '#FFB6C1', '#DDA0DD', '#87CEEB', '#FF69B4', '#BA55D3']
    sns.set_style("whitegrid")
    sns.set_context("talk")

    plt.figure(figsize=(20, 10))
    plt.axhline(y=0, color='k', linestyle=':', linewidth=2)

    melted_data = pd.DataFrame()
    idk_data = pd.DataFrame()
    idk_counts = {}

    for model in models:
        col_name = f'{model}_{metric}'
        if col_name in data.columns:
            model_data = data[[col_name]].copy()
            model_data['Model'] = model
            model_data.rename(columns={col_name: metric}, inplace=True)
            
            if metric == 'LLMEVAL':
                idk_count = (model_data[metric] == -1).sum()
                idk_counts[model] = idk_count
                
                # Separate -1 values for the strip plot
                idk_model_data = model_data[model_data[metric] == -1]
                idk_data = pd.concat([idk_data, idk_model_data], axis=0)
                
                # Exclude -1 values for the box plot
                model_data = model_data[model_data[metric] != -1]
            else:
                idk_counts[model] = 0
                
            melted_data = pd.concat([melted_data, model_data], axis=0)
        else:
            idk_counts[model] = 0  # Ensure every model has a count entry
            melted_data = pd.concat([melted_data, pd.DataFrame({metric: [np.nan], 'Model': model})], axis=0)
    # Set the x-axis ticks and labels before plotting
    ax = plt.gca()
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models.keys(), rotation=0, fontsize=20, ha='center')
    
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
    
    sns.stripplot(
        x='Model', 
        y=metric, 
        data=melted_data, 
        jitter=.35, 
        size=8, 
        edgecolor='black', 
        color='white', 
        linewidth=1, 
        alpha=0.3,
        ax=ax
    )
    
    # Plot the -1 values as dots in the -1 zone
    sns.stripplot(
        x='Model', 
        y=metric, 
        data=idk_data, 
        jitter=.35, 
        size=8, 
        edgecolor='black', 
        color='red', 
        linewidth=1, 
        alpha=0.7,
        ax=ax
    )

    for model in models:
        if metric == 'LLMEVAL':
            count = idk_counts.get(model, 0)
            model_index = list(models.keys()).index(model)
            plt.text(model_index, -1.2, f'(IDK={count})', ha='center', va='center', fontsize=16, color='red')

    plt.yticks(list(plt.yticks()[0]) + [-1], fontsize=20)
    plt.xlabel("Model", fontsize=22, fontweight='bold')
    plt.ylabel(metric, fontsize=22, fontweight='bold')
    plt.title(f"{title} (n = {len(data)})", fontsize=24)
    
    if metric == "LLMEVAL":
        plt.ylim(-1.5, 3.0)
        plt.yticks(np.arange(-1, 3.5, 0.5))

    plt.tight_layout()

    # Also log to wandb
    wandb_boxplot(data,plt)
    
    
    plt.savefig(f'{save_path}/{title}')
    plt.close()
