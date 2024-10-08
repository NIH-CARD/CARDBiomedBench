import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt

def plot_token_histograms(data: pd.DataFrame, text_col: str, color: str, title: str, save_path: str):
    """Create a histogram to visualize token counts for a given text column, showing frequency as a percentage."""
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    token_col = f'{text_col}_token_count'

    # Print median and sum token count
    median_token_count = data[token_col].median()

    # Determine the maximum token count and calculate the step size
    max_token_count = data[token_col].max()
    step_size = math.ceil(max_token_count / 10 / 10) * 10

    # Plot the histogram with percentage on y-axis
    plt.figure(figsize=(10, 4), dpi=100)
    sns.histplot(data[token_col], color=color, kde=True, stat='percent')
    
    # Add a vertical dashed line at the median
    plt.axvline(median_token_count, color='black', linestyle='--', label=f'Median: {median_token_count:.0f}')
    
    # Update axis labels and title
    plt.title(f'{title} Token Count Distribution')
    plt.xlabel(f'{title} Token Count')
    plt.ylabel('Frequency (%)')

    # Set x-axis ticks to start at 0 with dynamic step size
    plt.xticks(range(0, max_token_count + step_size, step_size))
    
    # Set y-axis limits and ticks
    plt.ylim(0, 15)
    plt.yticks(range(0, 20, 5))
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}_Token_Histogram.png')
    plt.close()