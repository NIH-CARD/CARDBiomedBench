import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

def create_performance_table(data: pd.DataFrame, metrics: list, models: dict) -> pd.DataFrame:
    """Create a table with columns: model | metric1 (mean ± 95% CI) | metric2 (mean ± 95% CI) | AR (± 95% CI) ..."""
    
    performance_rows = []
    
    for model in models:
        row = {'Model': model}
        
        for metric in metrics:
            col_name = f'{model}_{metric}'
            
            if col_name in data.columns:
                # Filter out -1 values for the metric
                metric_data = data[data[col_name] != -1][col_name]
                
                # Calculate mean and 95% confidence interval for the metric
                mean_val = metric_data.mean()
                ci_low, ci_high = stats.t.interval(0.95, len(metric_data)-1, loc=mean_val, scale=stats.sem(metric_data))
                
                # Combine mean and 95% CI into one string
                row[f'{metric} (95% CI)'] = f'{mean_val:.2f} ({ci_low:.2f}, {ci_high:.2f})'
                
            else:
                row[f'{metric} (95% CI)'] = 'N/A'
        
        # Calculate AR (Abstention Rate) and its CI
        bio_col_name = f'{model}_BioScore'
        if bio_col_name in data.columns:
            total_count = len(data[bio_col_name])
            ar_count = (data[bio_col_name] == -1).sum()
            ar_rate = ar_count / total_count
            
            # Calculate AR 95% confidence interval using binomial proportion CI
            ci_low, ci_high = stats.binom.interval(0.95, total_count, ar_rate, loc=0)
            ar_ci_low = ci_low / total_count
            ar_ci_high = ci_high / total_count
            
            # Combine AR rate and 95% CI into one string
            row['AR (95% CI)'] = f'{ar_rate:.2f} ({ar_ci_low:.2f}, {ar_ci_high:.2f})'
        else:
            row['AR (95% CI)'] = 'N/A'
        performance_rows.append(row)
    
    # Convert the list of rows into a DataFrame
    performance_table = pd.DataFrame(performance_rows)
    
    return performance_table

def style_dataframe(df: pd.DataFrame, title: str, save_path: str):
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

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
    fig, ax = plt.subplots(figsize=(len(df.columns) * 2.5, len(df) * 0.25 + 0.5))  # Adjust size based on number of rows

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