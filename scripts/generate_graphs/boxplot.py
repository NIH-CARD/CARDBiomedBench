import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_metric_boxplot(data: pd.DataFrame, metric: str, models: list, model_order: list, title: str, save_path: str):
    """
    Create a box and whisker plot to visualize performance for the specified metric,
    handling -1 values separately (e.g., for BioScore).
    """
    colors = ['#ADD8E6', '#FFB6C1', '#DDA0DD', '#87CEEB', '#FF69B4', '#BA55D3', '#CECECD']
    sns.set_style("whitegrid")
    sns.set_context("talk")
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
    })

    plt.figure(figsize=(20, 12))
    plt.axhline(y=0, color='k', linestyle=':', linewidth=2)

    melted_data = pd.DataFrame()
    idk_counts = {}

    # Reorder the models list according to model_order
    models = [model for model in model_order if model in models]

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
            else:
                idk_counts[model] = 0  # No IDK values for other metrics

            melted_data = pd.concat([melted_data, model_data], axis=0)
        else:
            idk_counts[model] = 0  # Ensure every model has a count entry
            # Append NaN values for models without data
            melted_data = pd.concat([melted_data, pd.DataFrame({metric: [np.nan], 'Model': model})], axis=0)
    
    ax = plt.gca()
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=0, fontsize=20, ha='center')
    
    # Custom properties for the median line
    medianprops = {'color': 'black', 'linewidth': 3}

    # Plot the boxplot for the metric
    sns.boxplot(
        x='Model', 
        y=metric, 
        data=melted_data, 
        palette=colors[:len(models)], 
        linewidth=2,
        hue='Model',
        dodge=False,
        ax=ax,
        medianprops=medianprops,
        legend=False
    )
    
    if metric == "BioScore":
        for model in models:
            count = idk_counts.get(model, 0)
            model_index = models.index(model)
            percent = (count / len(data)) * 100
            plt.text(model_index, -0.12, f'({percent:.2f}%)', ha='center', va='center', fontsize=20, color='red')
        
    # Adjust y-axis ticks
    yticks = [yt / 3.0 for yt in range(0, 4)]
    plt.ylim(-0.05, max(yticks) + 0.05)
    plt.yticks(yticks, [f'{yt:.2f}' for yt in yticks], fontsize=20)

    plt.xlabel("Model", fontsize=26, fontweight='bold', labelpad=30)
    plt.ylabel(metric, fontsize=26, fontweight='bold')
    plt.title(f"{title}", fontsize=28)

    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}.png')
    plt.close()

def plot_template_boxplot(data: pd.DataFrame, metric: str, model: str, title: str, save_path: str):
    """
    Create box and whisker plots to visualize performance on each template_uuid,
    handling -1 values for BioScore separately.
    """
    sns.set_style("whitegrid")
    sns.set_context("talk")

    col_name = f'{model}_{metric}'
    if col_name not in data.columns:
        raise ValueError(f"Column {col_name} not found in the data.")
    
    melted_data = data[[col_name, 'uuid', 'template uuid']].copy()
    melted_data.rename(columns={col_name: metric}, inplace=True)
    
    idk_counts = {}
    # Remove -1 values for BioScore and calculate IDK counts
    if metric == "BioScore":
        for template_uuid in melted_data['template uuid'].unique():
            template_data = melted_data[melted_data['template uuid'] == template_uuid]
            idk_count = (template_data[metric] == -1).sum()
            idk_counts[template_uuid] = idk_count
            # Exclude -1 values
            melted_data = melted_data[~((melted_data['template uuid'] == template_uuid) & (melted_data[metric] == -1))]
    else:
        # No IDK values for other metrics
        idk_counts = {template_uuid: 0 for template_uuid in melted_data['template uuid'].unique()}
    
    plt.figure(figsize=(20, 10))
    plt.axhline(y=0, color='k', linestyle=':', linewidth=2)
    
    sns.boxplot(
        x='template uuid', 
        y=metric, 
        data=melted_data, 
        palette='Set2',
        linewidth=2
    )
    
    total_counts = melted_data.groupby('template uuid').size() + pd.Series(idk_counts)
    for template_uuid in melted_data['template uuid'].unique():
        model_index = list(melted_data['template uuid'].unique()).index(template_uuid)
        count = idk_counts.get(template_uuid, 0)
        total = total_counts.get(template_uuid, len(data))
        percent = (count / total) * 100 if total > 0 else 0
        plt.text(model_index, -0.125, f'({percent:.2f}%)', ha='center', va='center', fontsize=16, color='red')
    
    # Adjust y-axis ticks
    yticks = [yt / 3.0 for yt in range(0, 4)]
    plt.ylim(-0.05, max(yticks) + 0.05)
    plt.yticks(yticks, [f'{yt:.2f}' for yt in yticks])
    
    plt.xlabel("Template UUID", fontsize=22, fontweight='bold')
    plt.ylabel(metric, fontsize=22, fontweight='bold')
    plt.title(f"{title} (n = {len(melted_data)})", fontsize=24)
    
    plt.xticks()
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/{model}_{metric}_template.png')
    plt.close()