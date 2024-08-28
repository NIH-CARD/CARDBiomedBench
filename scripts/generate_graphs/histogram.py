import pandas as pd
import seaborn as sns
import tiktoken
import math
import matplotlib.pyplot as plt

def count_tokens_tiktoken(string: str, model: str = "gpt-4o") -> int:
    """Returns the number of tokens in a text string using the appropriate encoding for a given model."""
    try:
        # Get the encoding for the specified model
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # If the model is not recognized, default to cl100k_base encoding
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Encode the string and return the number of tokens
    num_tokens = len(encoding.encode(string))
    return num_tokens

def plot_token_histograms(data: pd.DataFrame, text_col: str, color: str, title: str, save_path: str):
    """Create a histogram to visualize token counts for a given text column, showing frequency as a percentage."""
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Compute token counts for the given column
    data['token_count'] = data[text_col].apply(lambda x: count_tokens_tiktoken(x))
    
    # Determine the maximum token count and calculate the step size
    max_token_count = data['token_count'].max()
    step_size = math.ceil(max_token_count / 10 / 10) * 10

    # Plot the histogram with percentage on y-axis
    plt.figure(figsize=(10, 4), dpi=100)
    sns.histplot(data['token_count'], color=color, kde=True, stat='percent')
    
    plt.title(f'{title} Token Counts')
    plt.xlabel('Token Count')
    plt.ylabel('Percentage')

    # Set x-axis ticks to start at 0 with dynamic step size
    plt.xticks(range(0, max_token_count + step_size, step_size))
    
    # Set y-axis limits and ticks
    plt.ylim(0, 50)
    plt.yticks(range(0, 60, 10))
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'{save_path}/{title} Histogram.png')
    plt.close()