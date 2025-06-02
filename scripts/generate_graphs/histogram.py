import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt

def plot_token_histograms(data: pd.DataFrame, text_col: str, color: str, title: str, save_path: str, ax: plt.Axes = None):
    """Create a histogram showing token counts for a text column, with optional subplot support."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20
    })

    token_col = f'{text_col}_token_count'

    # IQR filtering to remove outliers
    Q1 = data[token_col].quantile(0.25)
    Q3 = data[token_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[token_col] >= lower_bound) & (data[token_col] <= upper_bound)]

    median_token_count = filtered_data[token_col].median()
    max_token_count = filtered_data[token_col].max()
    step_size = math.ceil(max_token_count / 10 / 10) * 10

    # Use provided ax or create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

    sns.histplot(
        filtered_data[token_col],
        color=color,
        kde=True,
        stat='percent',
        ax=ax
    )

    ax.axvline(median_token_count, color='black', linestyle='--', label=f'Median: {median_token_count:.0f} (IQR={IQR:.0f})')
    ax.set_title(f'{title} Token Count Distribution')
    ax.set_xlabel(f'{title} Token Count')
    ax.set_ylabel('Frequency (%)')

    ax.set_xticks(range(0, max_token_count + step_size, step_size))
    ax.set_ylim(0, 15)
    ax.set_yticks(range(0, 20, 5))
    ax.grid(True, which='major', axis='both', linestyle='-', alpha=0.7)

    # ax.legend()

    if save_path and ax is None:
        fig.tight_layout()
        fig.savefig(f'{save_path}/{title}_Token_Histogram.png', bbox_inches="tight", pad_inches=0)
        plt.close(fig)