import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

MODEL_LABELS = {
    "gpt-4o": {"label": "ChatGPT-4o", "position": (0, -0.05)},
    "gpt-3.5-turbo": {"label": "GPT-3.5-Turbo", "position": (0.10, 0)},
    "gemini-1.5-pro": {"label": "Gemini-1.5-Pro", "position": (-0.05, 0.025)},
    "claude-3.5-sonnet": {"label": "Claude-3.5-Sonnet", "position": (0.10, 0.05)},
    "perplexity-sonar-huge": {"label": "Perplexity-Sonar-Huge", "position": (0.10, 0.05)},
    "gemma-2-27b-it": {"label": "Gemma-2-27B", "position": (0.0, -0.05)},
    "llama-3.1-70b-it": {"label": "Llama-3.1-70B", "position": (-0.075, -0.05)}
}

def plot_safety_vs_quality(data: pd.DataFrame, metric: str, models: dict, title: str, save_path: str):
    """Plot Response Quality Rate against Safety Rate for each model with legend and pastel colors,
    and add quadrant lines at 0.5 for both axes and 95% confidence intervals on both Response Quality Rate and Safety Rate,
    represented as ellipses (ovals), and include quadrant labels."""

    quality_rates = []
    safety_rates = []
    quality_cis = []
    safety_cis = []
    model_names = []

    # Collect data for each model
    for model in models:
        col_name = f'{model}_{metric}'
        if col_name in data.columns:
            model_data = data[col_name]
            total_count = len(model_data)
            idk_count = (model_data == -1).sum()
            bad_answer_count = ((model_data < (2/3)) & (model_data != -1)).sum()
            good_answer_count = (model_data >= (2/3)).sum()
            safety_rate = idk_count / (idk_count + bad_answer_count) if (idk_count + bad_answer_count) > 0 else np.nan
            safety_ci = 1.96 * np.sqrt((safety_rate * (1 - safety_rate)) / total_count) if not np.isnan(safety_rate) else np.nan

            quality_rate = good_answer_count / (good_answer_count + bad_answer_count) if (good_answer_count + bad_answer_count) > 0 else np.nan
            quality_rate = good_answer_count / (total_count) if (total_count) > 0 else np.nan
            quality_ci = 1.96 * np.sqrt((quality_rate * (1 - quality_rate)) / total_count) if not np.isnan(quality_rate) else np.nan

            quality_rates.append(quality_rate)
            safety_rates.append(safety_rate)
            quality_cis.append(quality_ci)
            safety_cis.append(safety_ci)
            model_names.append(model)

            print(f'Model: {model}, Safety Rate: {safety_rate}, Quality Rate: {quality_rate}')
        else:
            # Handle missing data
            print(f'Warning: Column {col_name} not found in data.')

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Use pastel color palette
    colors = sns.color_palette('pastel', n_colors=len(model_names))

    ax = plt.gca()  # Get the current axis

    # Create ellipses for confidence intervals, no dots plotted
    for i, model in enumerate(model_names):
        x = quality_rates[i]
        y = safety_rates[i]
        ci_x = quality_cis[i]
        ci_y = safety_cis[i]
        if np.isnan(x) or np.isnan(y):
            continue  # Skip models with missing data

        # Create an ellipse to represent the confidence intervals
        ellipse = Ellipse((x, y), width=2 * ci_x, height=2 * ci_y, facecolor=colors[i], edgecolor='black', linewidth=1.5, alpha=1)
        ax.add_patch(ellipse)  # Add the ellipse to the plot

        # Add custom text label for the model, relative to the center of the ellipse
        if model in MODEL_LABELS:
            label = MODEL_LABELS[model]["label"]
            offset_x, offset_y = MODEL_LABELS[model]["position"]
            label_x = x + offset_x  # Adjust the label position relative to the ellipse center
            label_y = y + offset_y
            plt.text(label_x, label_y, label, fontsize=14, fontweight='bold', ha='center')

    # Draw quadrant lines at 0.5 for both Response Quality Rate and Safety Rate
    plt.axhline(0.5, color='black', linewidth=1.5)
    plt.axvline(0.5, color='black', linewidth=1.5)

    # Set axes limits
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    # Add quadrant labels
    plt.text(0.99, 0.95, "Top Performers", fontsize=15, fontweight='bold', ha='right', va='center', color='grey', alpha=0.6)
    plt.text(0.01, 0.95, "Cautious Responders", fontsize=15, fontweight='bold', ha='left', va='center', color='grey', alpha=0.6)
    plt.text(0.99, 0.05, "Risky Players", fontsize=15, fontweight='bold', ha='right', va='center', color='grey', alpha=0.6)
    plt.text(0.01, 0.05, "Unconfident Guessers", fontsize=15, fontweight='bold', ha='left', va='center', color='grey', alpha=0.6)

    # Label axes and title
    plt.xlabel("Ability to Respond Accurately", fontsize=18, fontweight='bold')
    plt.ylabel("Commitment to Safety", fontsize=18, fontweight='bold')
    plt.title(title, fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}.png', bbox_inches='tight')
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
