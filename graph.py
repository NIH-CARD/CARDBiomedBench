import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

def plot_auto_vs_manual_correlation(
    df: pd.DataFrame,
    auto_metrics: list,
    manual_metric: str,
    save_dir: str,
    normalize_abstain: bool = True
):
    """
    Computes Spearman correlations between automatic metrics and manual grading,
    and creates both a barplot of correlations and scatter plots with regression lines.

    Parameters:
    - df: DataFrame containing metric columns
    - auto_metrics: List of automatic metric column names (e.g., ['BLEU', 'ROUGE2', ...])
    - manual_metric: Name of the manual grading column (e.g., 'SMEScore')
    - save_dir: Directory to save the plots
    - normalize_abstain: If True, treats -1 values as 0
    """
    os.makedirs(save_dir, exist_ok=True)

    # Drop missing values
    df_clean = df[auto_metrics + [manual_metric]].dropna()

    # Normalize abstain values
    if normalize_abstain:
        df_clean[manual_metric] = df_clean[manual_metric].replace(-1, 0)
        if "BioScore" in auto_metrics:
            df_clean["BioScore"] = df_clean["BioScore"].replace(-1, 0)

    # Style setup
    sns.set_style("whitegrid")
    sns.set_context("talk")
    plt.rcParams.update({'font.family': 'DejaVu Sans'})

    # Compute Spearman correlations
    correlations = {
        metric: spearmanr(df_clean[metric], df_clean[manual_metric]).correlation
        for metric in auto_metrics
    }

    # Barplot of correlations
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(correlations.keys()), y=list(correlations.values()), palette='pastel', linewidth=2)
    plt.title("Metric Spearman Correlation with SME Grading", fontsize=20, weight='bold')
    plt.ylabel("Spearman Correlation", fontsize=16)
    plt.ylim(-0.2, 1.01)
    plt.axhline(0, color="gray", linestyle="--")
    plt.xticks(rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correlation_barplot.png")
    plt.close()

    # Scatter plots
    for metric in auto_metrics:
        sns.lmplot(
            x=metric, y=manual_metric, data=df_clean,
            height=5, aspect=1.3,
            scatter_kws={"s": 50, "alpha": 0.7},
            line_kws={"color": "black", "linewidth": 2}
        )
        plt.ylim(-0.1, 1.1)
        # plt.xlim(-0.1, 1.1)
        plt.xlabel(metric, fontsize=14)
        plt.ylabel(manual_metric, fontsize=14)
        plt.title(f"{metric} vs. {manual_metric}", fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{metric}_vs_{manual_metric}.png", bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

df = pd.read_csv("data/CARDBiomedBench_handgraded.csv")
auto_metrics = ['BioScore', 'BLEU', 'ROUGE2', 'ROUGEL', 'BERTScore']
manual_metric = 'SMEScore'
plot_auto_vs_manual_correlation(df, auto_metrics, manual_metric, save_dir="results")