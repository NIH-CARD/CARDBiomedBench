import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_scatterplot(data: pd.DataFrame, x_metric: str, y_metric: str, models: dict, title: str, save_path: str):
    """Create a scatterplot to visualize the relationship between the stdv of one metrics to another's mean for each model."""
    colors = ['#ADD8E6', '#FFB6C1', '#DDA0DD', '#87CEEB', '#FF69B4', '#BA55D3']
    sns.set_style("whitegrid")
    sns.set_context("talk")

    plt.figure(figsize=(20, 10))
    
    for i, (model, color) in enumerate(zip(models.keys(), colors)):
        x_col = f'{model}_{x_metric}'
        y_col = f'{model}_{y_metric}'
        if x_col in data.columns and y_col in data.columns:
            
            # Filter out -1 values
            x_filtered = data[x_col][data[x_col] != -1]
            y_filtered = data[y_col][data[y_col] != -1]

            # Compute stdvs
            x_stdv = x_filtered.std()
            y_stdv = y_filtered.std()
            
            plt.scatter(
                x_stdv, 
                y_stdv, 
                s=300,
                label=model, 
                color=color, 
                edgecolor='black',
                alpha=0.7
            )
        
    plt.axhline(y=0, color='k', linestyle=':', linewidth=2)
    plt.axvline(x=0, color='k', linestyle=':', linewidth=2)
    
    plt.xlabel(f'{x_metric} Mean', fontsize=22, fontweight='bold')
    plt.ylabel(f'{y_metric} Stdv', fontsize=22, fontweight='bold')
    plt.title(f"{title} (n = {len(data)})", fontsize=24)
    
    plt.legend(title="Model", fontsize=18, title_fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}_scatter.png')
    plt.close()
