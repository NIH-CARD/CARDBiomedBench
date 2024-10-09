import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_abstention_vs_bioscore(data: pd.DataFrame, metric: str, models: dict, title: str, save_path: str):
    """Plot mean BioScore against abstention rate for each model with legend and pastel colors."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    mean_bioscores = []
    abstention_rates = []
    model_names = []

    # Collect data for each model
    for model in models:
        col_name = f'{model}_{metric}'
        if col_name in data.columns:
            model_data = data[col_name]
            total_count = len(model_data)
            idk_count = (model_data == -1).sum()
            abstention_rate = idk_count / total_count
            valid_data = model_data[model_data != -1]
            if not valid_data.empty:
                mean_bioscore = valid_data.mean()
            else:
                mean_bioscore = np.nan
            mean_bioscores.append(mean_bioscore)
            abstention_rates.append(abstention_rate)
            model_names.append(model)
        else:
            # Handle missing data
            mean_bioscores.append(np.nan)
            abstention_rates.append(np.nan)
            model_names.append(model)
            print(f'Warning: Column {col_name} not found in data.')

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Use pastel color palette
    colors = sns.color_palette('pastel', n_colors=len(model_names))

    # Create scatter plot with legend labels
    for i, model in enumerate(model_names):
        x = mean_bioscores[i]
        y = abstention_rates[i]
        if np.isnan(x) or np.isnan(y):
            continue  # Skip models with missing data
        plt.scatter(x, y, s=150, color=colors[i], edgecolors='k', label=model)

    # Set axes limits and flip y-axis
    plt.xlim(0.0, 1.0)
    plt.ylim(1.0, 0.0)

    # Label axes
    plt.xlabel("Mean BioScore", fontsize=18, fontweight='bold')
    plt.ylabel("Abstention Rate", fontsize=18, fontweight='bold')

    # Add title
    plt.title(title, fontsize=20, fontweight='bold')

    # Add legend
    plt.legend(title='Models', fontsize=12, title_fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}.png')
    plt.close()

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
