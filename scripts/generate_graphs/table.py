import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

def create_performance_table(data: pd.DataFrame, metrics: list, models: dict) -> pd.DataFrame:
    """Create a table with columns: model | metric1 mean, metric1 95% CI | metric2 mean, metric2 95% CI ..."""
    
    performance_rows = []
    
    for model in models:
        row = {'Model': model}
        
        for metric in metrics:
            col_name = f'{model}_{metric}'
            
            if col_name in data.columns:
                # Filter out -1 values
                metric_data = data[data[col_name] != -1][col_name]
                
                # Calculate mean and 95% confidence interval
                mean_val = metric_data.mean()
                ci_low, ci_high = stats.t.interval(0.95, len(metric_data)-1, loc=mean_val, scale=stats.sem(metric_data))
                
                # Store the values in the row
                row[f'{metric} Mean'] = mean_val
                row[f'{metric} 95% CI'] = f'{ci_low:.2f}, {ci_high:.2f}'
            else:
                row[f'{metric} Mean'] = 'N/A'
                row[f'{metric} 95% CI'] = 'N/A'
        
        performance_rows.append(row)
    
    # Convert the list of rows into a DataFrame
    performance_table = pd.DataFrame(performance_rows)
    
    return performance_table

def style_dataframe(df: pd.DataFrame, title: str, save_path: str):
    # Set up the aesthetics for the plot
    sns.set_theme(style="whitegrid", font="DejaVu Sans")  # Use Seaborn to set a good font and style

    # Format the DataFrame
    styled_df = df.style.format("{:.4f}") \
                         .set_table_styles([{
                             'selector': 'th',
                             'props': [('font-weight', 'bold')]
                         }, {
                             'selector': 'td',
                             'props': [('font-family', 'DejaVu Sans')]
                         }])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(len(df.columns) * 2, len(df) * 0.25 + 1))  # Adjust size based on number of rows

    # Hide axes
    ax.axis('off')
    ax.axis('tight')

    # Render the styled DataFrame as a table in Matplotlib
    table = ax.table(cellText=styled_df.data.map(lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else x).values,
                     colLabels=styled_df.columns,
                     cellLoc='center',
                     loc='center')

    # Set font size and alignment
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Adjusted to a smaller font size
    table.scale(1, 1)

    # Bold the headers
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header
            cell.set_text_props(weight='bold', fontsize=10)

    # Set title
    plt.title(title, weight='bold', fontsize=14, fontname='DejaVu Sans')

    # Save the figure as a PNG
    plt.savefig(f'{save_path}/{title}.png', bbox_inches='tight', dpi=300)
    plt.close()