import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import AnnotationBbox, TextArea, HPacker


MODEL_LABELS = {
    "gpt-5": {"label": "OpenAI-GPT-5", "position": (-0.010, +0.030)},
    "gpt-5-mini": {"label": "OpenAI-GPT-5-Mini", "position": (0.00, +0.030)},
    "o3": {"label": "OpenAI-o3", "position": (-0.010, -0.040)},
    "o3-mini": {"label": "OpenAI-o3-mini", "position": (-0.010, +0.030)},
    "gpt-4.1": {"label": "OpenAI-GPT-4.1", "position": (+0.060, +0.030)},
    "gpt-4o": {"label": "OpenAI-GPT-4o", "position": (0.00, -0.040)},
    "gpt-3.5-turbo": {"label": "OpenAI-GPT-3.5-Turbo", "position": (+0.050, -0.050)},
    "gemini-2.5-pro": {"label": "Gemini-2.5-Pro", "position": (0.000, +0.030)},
    "gemini-2.5-flash": {"label": "Gemini-2.5-Flash", "position": (0.000, +0.030)},
    "gemini-2.0-flash": {"label": "Gemini-2.0-Flash", "position": (-0.085, -0.002)},
    "gemini-1.5-pro": {"label": "Gemini-1.5-Pro", "position": (-0.080, -0.025)},
    "gemma-2-27b-it": {"label": "Gemma-2-27B", "position": (0.00, -0.040)},
    "claude-4.1-opus": {"label": "Claude-4.1-Opus", "position": (0.00, -0.040)},
    "claude-4.0-sonnet": {"label": "Claude-4.0-Sonnet", "position": (0.00, +0.060)},
    "claude-3.7-sonnet": {"label": "Claude-3.7-Sonnet", "position": (0.00, -0.040)},
    "claude-3.5-sonnet": {"label": "Claude-3.5-Sonnet", "position": (-0.100, +0.010)},
    "perplexity-sonar-huge": {"label": "Perplexity-Sonar-Huge", "position": (0.020, +0.030)},
    "llama-3.1-70b-it": {"label": "Llama-3.1-70B", "position":(-0.085, -0.002)},
}

def small_after_dash(label: str, main_size=14, small_size=12, weight='bold'):
    if "-" not in label:
        return TextArea(label, textprops=dict(fontsize=main_size, fontweight=weight))
    head, tail = label.split("-", 1)
    box_main  = TextArea(head, textprops=dict(fontsize=main_size, fontweight=weight))
    box_small = TextArea("-" + tail, textprops=dict(fontsize=small_size, fontweight=weight))
    return HPacker(children=[box_main, box_small], align="center", pad=0, sep=0)

def plot_safety_vs_quality(data: pd.DataFrame, metric: str, models: list, title: str, save_path: str):
    """Plot Response Quality Rate against Safety Rate for each model with legend and pastel colors,
    and add quadrant lines at 0.5 for both axes and 95% confidence intervals on both Response Quality Rate and Safety Rate,
    represented as ellipses (ovals), and include quadrant labels."""
    
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
    })

    quality_rates = []
    safety_rates = []
    quality_cis = []
    safety_cis = []
    model_names = []

    # Collect data for each model
    for model in models:
        if model in ["gpt-3.5-turbo", "gpt-5-mini"]: continue # skip these models
        col_name = f'{model}_{metric}'
        if col_name in data.columns:
            model_data = data[col_name]
            total_count = len(model_data)
            idk_count = (model_data == -1).sum()
            bad_answer_count = ((model_data < (2/3)) & (model_data != -1)).sum()
            good_answer_count = (model_data >= (2/3)).sum()
            
            safety_rate = idk_count / (idk_count + bad_answer_count) if (idk_count + bad_answer_count) > 0 else np.nan
            safety_ci = 1.96 * np.sqrt((safety_rate * (1 - safety_rate)) / total_count) if not np.isnan(safety_rate) else np.nan
            quality_rate = good_answer_count / (total_count) if (total_count) > 0 else np.nan
            quality_ci = 1.96 * np.sqrt((quality_rate * (1 - quality_rate)) / total_count) if not np.isnan(quality_rate) else np.nan

            quality_rates.append(quality_rate)
            safety_rates.append(safety_rate)
            quality_cis.append(quality_ci)
            safety_cis.append(safety_ci)
            model_names.append(model)
        else:
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
        ellipse = Ellipse((x, y), width=2 * ci_x, height=2 * ci_y, facecolor='white', edgecolor="#3587CD", linewidth=1.75, alpha=.9, zorder=2)
        ax.add_patch(ellipse)

        # Add custom text label for the model, relative to the center of the ellipse
        if model in MODEL_LABELS:
            label = MODEL_LABELS[model]["label"]
            offset_x, offset_y = MODEL_LABELS[model]["position"]
            label_x = x + offset_x  # Adjust the label position relative to the ellipse center
            label_y = y + offset_y
            main_size, small_size = 11, 10
            packed = small_after_dash(label, main_size=main_size, small_size=small_size)
            ab = AnnotationBbox(
                packed, (label_x, label_y),
                xycoords='data',
                frameon=False,
                box_alignment=(0.5, 0.5),
                zorder=3
            )
            ax.add_artist(ab)
            if model in ["claude-4.0-sonnet"]:
                ax.annotate(
                    '', xy=(x, y), xytext=(label_x, label_y),
                    arrowprops=dict(
                        arrowstyle="-",  # simple line
                        color="black",
                        lw=1,
                        shrinkA=5, shrinkB=5
                    ),
                    zorder=1
                )

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

    ax.annotate(
        '', xy=(1.05, -0.15), xytext=(-0.1, -0.15),
        arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='-|>', lw=1.75),
        annotation_clip=False
    )
    ax.annotate(
        '', xy=(-0.1, 1.05), xytext=(-0.1, -0.15),
        arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='-|>', lw=1.75),
        annotation_clip=False
    )

    plt.tight_layout()
    plt.savefig(f'{save_path}/figures/figure3.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.savefig(f'{save_path}/figures/figure3.eps', format='eps', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

if __name__ == "__main__":
    data = pd.read_csv("compiled.csv")
    models_list = list(MODEL_LABELS.keys())
    plot_safety_vs_quality(data, metric='BioScore', models=models_list, title='Safety Rate vs. Response Quality Rate', save_path=".")