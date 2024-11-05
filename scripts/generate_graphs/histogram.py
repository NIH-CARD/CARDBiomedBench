import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt

def plot_token_histograms(data: pd.DataFrame, text_col: str, color: str, title: str, save_path: str):
    """Create a histogram to visualize token counts for a given text column, showing frequency as a percentage, and filter outliers."""
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

    # Calculate Q1 (25th percentile), Q3 (75th percentile), and Interquartile range
    Q1 = data[token_col].quantile(0.25)
    Q3 = data[token_col].quantile(0.75)
    IQR = Q3 - Q1

    # Define lower and upper bounds for filtering outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the data to exclude outliers
    filtered_data = data[(data[token_col] >= lower_bound) & (data[token_col] <= upper_bound)]

    # Print median and sum token count of the filtered data
    median_token_count = filtered_data[token_col].median()

    # Determine the maximum token count and calculate the step size
    max_token_count = filtered_data[token_col].max()
    step_size = math.ceil(max_token_count / 10 / 10) * 10

    # Plot the histogram with percentage on y-axis
    plt.figure(figsize=(10, 4), dpi=100)
    sns.histplot(filtered_data[token_col], color=color, kde=True, stat='percent')
    
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