import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_metric_boxplot(data: pd.DataFrame, metric: str, models: dict, model_order: list, title: str, save_path: str):
    """Create a box and whisker plot to visualize performance for the BioScore metric, handling -1 values separately."""
    colors = ['#ADD8E6', '#FFB6C1', '#DDA0DD', '#87CEEB', '#FF69B4', '#BA55D3']
    sns.set_style("whitegrid")
    sns.set_context("talk")

    plt.figure(figsize=(20, 10))
    plt.axhline(y=0, color='k', linestyle=':', linewidth=2)

    melted_data = pd.DataFrame()
    idk_counts = {}

    # Get the new order of models and reorder the models dictionary
    models = {model: models[model] for model in model_order}

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
    
    # Custom properties for the median line
    medianprops = {'color': 'black', 'linewidth': 3}

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
        medianprops=medianprops,  # Change median line color
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

def plot_template_boxplot(data: pd.DataFrame, metric: str, model: str, title: str, save_path: str):
    """Create box and whisker plots to visualize performance on each template_uuid, handling -1 values for BioScore separately."""
    sns.set_style("whitegrid")
    sns.set_context("talk")

    col_name = f'{model}_{metric}'
    if col_name not in data.columns:
        raise ValueError(f"Column {col_name} not found in the data.")
    
    melted_data = data[[col_name, 'uuid', 'template uuid']].copy()
    melted_data.rename(columns={col_name: metric}, inplace=True)
    
    idk_counts = {}
    for template_uuid in melted_data['template uuid'].unique():
        template_data = melted_data[melted_data['template uuid'] == template_uuid]
        idk_count = (template_data[metric] == -1).sum()
        idk_counts[template_uuid] = idk_count
        melted_data = melted_data[~((melted_data['template uuid'] == template_uuid) & (melted_data[metric] == -1))]
    
    plt.figure(figsize=(20, 10))
    plt.axhline(y=0, color='k', linestyle=':', linewidth=2)
    
    sns.boxplot(
        x='template uuid', 
        y=metric, 
        data=melted_data, 
        palette='Set2',
        hue='template uuid',
        linewidth=2
    )
    
    total_counts = melted_data.groupby('template uuid').size() + pd.Series(idk_counts)
    for template_uuid in melted_data['template uuid'].unique():
        model_index = list(melted_data['template uuid'].unique()).index(template_uuid)
        count = idk_counts.get(template_uuid, 0)
        total = total_counts.get(template_uuid, len(data))
        percent = (count / total) * 100
        plt.text(model_index, -0.125, f'({percent:.2f}%)', ha='center', va='center', fontsize=16, color='red')
    
    yticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    yticks = [yt / 3.0 for yt in yticks]
    
    plt.ylim(-0.05, max(yticks) + 0.05)
    plt.yticks(yticks, [f'{yt:.2f}' for yt in yticks])

    plt.xlabel("Template UUID", fontsize=22, fontweight='bold')
    plt.ylabel(metric, fontsize=22, fontweight='bold')
    plt.title(f"{title} (n = {len(melted_data)})", fontsize=24)
    
    plt.xticks()
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/{model}_{metric}_template.png')
    plt.close()