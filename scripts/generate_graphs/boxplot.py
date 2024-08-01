import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_metric_boxplot(data: pd.DataFrame, metric: str, models: dict, title: str, save_path: str):
    """Create a box and whisker plot to visualize performance for a single metric."""
    colors = ['#ADD8E6', '#FFB6C1', '#DDA0DD', '#87CEEB', '#FF69B4', '#BA55D3']
    sns.set_style("whitegrid")
    sns.set_context("talk")

    plt.figure(figsize=(20, 10))
    plt.axhline(y=0, color='k', linestyle=':', linewidth=2)

    melted_data = pd.DataFrame()
    for model in models:
        col_name = f'{model}_{metric}'
        if col_name in data.columns:
            model_data = data[[col_name]].copy()
            model_data['Model'] = model
            model_data.rename(columns={col_name: metric}, inplace=True)
            
            if metric == 'LLMEVAL':
                model_data = model_data[model_data[metric] != -1]

            ci_lower, ci_upper = np.percentile(model_data[metric], [2.5, 97.5])
            model_data = model_data[(model_data[metric] >= ci_lower) & (model_data[metric] <= ci_upper)]
            
            melted_data = pd.concat([melted_data, model_data], axis=0)
    
   # melted_data['Model'] = melted_data['Model'].map(MODEL_NAME_MAPPING)

    ax = sns.boxplot(
        x='Model', 
        y=metric, 
        data=melted_data, 
        palette=colors, 
        linewidth=2,
        hue='Model',
        dodge=False,
        legend=False
    )
    
    if ax.legend_ is not None:
        ax.legend_.remove()
    
    ax = sns.stripplot(
        x='Model', 
        y=metric, 
        data=melted_data, 
        jitter=.35, 
        size=8, 
        edgecolor='black', 
        color='white', 
        linewidth=1, 
        alpha=0.3
    )
    
    plt.xticks(rotation=0, fontsize=20, ha='center')
    plt.yticks(fontsize=20)
    plt.xlabel("Model", fontsize=22, fontweight='bold')
    plt.ylabel(metric, fontsize=22, fontweight='bold')
    plt.title(title, fontsize=24)
    
    if metric == "LLMEVAL":
        plt.ylim(0, 3.0)
        plt.yticks(np.arange(0, 3.5, 0.5))

    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}')
    plt.close()