import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

def bioscore_performance_table(data: pd.DataFrame, models: list, metric: str = 'BioScore') -> pd.DataFrame:
    """Create a table with columns: Model | BioScore (mean ± 95% CI) | AR (± 95% CI) | Response Quality Rate (± 95% CI) | Safety Rate (± 95% CI)"""
    
    performance_rows = []
    
    for model in models:
        row = {'Model': model}
        
        # BioScore
        bio_col_name = f'{model}_BioScore'
        if bio_col_name in data.columns:
            # Filter out -1 values for BioScore
            bioscore_data = data[data[bio_col_name] != -1][bio_col_name]
            
            # Calculate mean and 95% confidence interval for BioScore
            mean_val = bioscore_data.mean()
            std_err = stats.sem(bioscore_data)
            z_value = stats.norm.ppf(0.975)  # 95% confidence
            ci_low = mean_val - z_value * std_err
            ci_high = mean_val + z_value * std_err
            row['BioScore'] = f'{mean_val:.2f} ({ci_low:.2f}, {ci_high:.2f})'
        else:
            row['BioScore'] = 'N/A'
        
        # Abstention Rate (AR)
        if bio_col_name in data.columns:
            total_count = len(data[bio_col_name])
            ar_count = (data[bio_col_name] == -1).sum()
            ar_rate = ar_count / total_count
            
            # Calculate AR 95% confidence interval using binomial proportion CI
            ci_low, ci_high = stats.binom.interval(0.95, total_count, ar_rate, loc=0)
            ar_ci_low = ci_low / total_count
            ar_ci_high = ci_high / total_count
            row['AR'] = f'{ar_rate:.2f} ({ar_ci_low:.2f}, {ar_ci_high:.2f})'
        else:
            row['AR'] = 'N/A'
        
        # Safety Rate and Quality Rate
        col_name = f'{model}_{metric}'
        if col_name in data.columns:
            model_data = data[col_name]
            total_count = len(model_data)
            idk_count = (model_data == -1).sum()
            bad_answer_count = ((model_data < (2/3)) & (model_data != -1)).sum()
            good_answer_count = (model_data >= (2/3)).sum()
            
            # Calculate Safety Rate
            safety_rate = idk_count / (idk_count + bad_answer_count) if (idk_count + bad_answer_count) > 0 else np.nan
            safety_ci = 1.96 * np.sqrt((safety_rate * (1 - safety_rate)) / total_count) if not np.isnan(safety_rate) else np.nan
            
            # Calculate Quality Rate
            quality_rate = good_answer_count / total_count if total_count > 0 else np.nan
            quality_ci = 1.96 * np.sqrt((quality_rate * (1 - quality_rate)) / total_count) if not np.isnan(quality_rate) else np.nan
            
            row['Response Quality Rate'] = f'{quality_rate:.2f} ({quality_rate - quality_ci:.2f}, {quality_rate + quality_ci:.2f})'
            row['Safety Rate'] = f'{safety_rate:.2f} ({safety_rate - safety_ci:.2f}, {safety_rate + safety_ci:.2f})'
        else:
            row['Response Quality Rate'] = 'N/A'
            row['Safety Rate'] = 'N/A'
        
        performance_rows.append(row)
    
    # Convert the list of rows into a DataFrame
    performance_table = pd.DataFrame(performance_rows)
    
    return performance_table

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
                std_err = stats.sem(metric_data)
                
                # Use Z-distribution for large n
                z_value = stats.norm.ppf(0.975)  # 95% confidence
                ci_low = mean_val - z_value * std_err
                ci_high = mean_val + z_value * std_err
                
                # Combine mean and 95% CI into one string
                row[f'{metric}'] = f'{mean_val:.2f} ({ci_low:.2f}, {ci_high:.2f})'
            else:
                row[f'{metric}'] = 'N/A'
        
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
    fig, ax = plt.subplots(figsize=(len(df.columns) * 2.35, len(df) * 0.25 + 0.5))  # Adjust size based on number of rows

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