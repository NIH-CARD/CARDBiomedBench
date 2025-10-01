import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scripts.generate_graphs.pie import plot_category_pie_chart
from scripts.generate_graphs.histogram import plot_token_histograms
from scripts.generate_graphs.heatmap import plot_heatmap

def create_four_panel_distribution_figure(data, save_path: str):
    """
    Create a 2x2 figure with:
    - Bio Category Pie Chart
    - Reasoning Category Pie Chart
    - Question Token Histogram (spanning bottom left)
    - Answer Token Histogram (spanning bottom right)
    """
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18
    })
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.8])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Subplot labels
    ax1.text(-1.045, 1.1, "(A)", transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')
    ax2.text(1.70, 1.1, "(B)", transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')
    ax3.text(-0.1125, -0.15, "(C)", transform=ax3.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')
    ax4.text(.975, -0.15, "(D)", transform=ax4.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')

    # Donut charts
    plot_category_pie_chart(
        data,
        category="bio_category",
        title="Bio Category Distribution",
        ax=ax1,
        color_flag=1,
        save_path=None
    )
    plot_category_pie_chart(
        data,
        category="reasoning_category",
        title="Reasoning Category Distribution",
        ax=ax2,
        color_flag=2,
        save_path=None
    )

    # Token histograms
    plot_token_histograms(
        data,
        text_col="question",
        color="dodgerblue",
        title="Question",
        ax=ax3,
        save_path=None
    )
    plot_token_histograms(
        data,
        text_col="answer",
        color="deeppink",
        title="Answer",
        ax=ax4,
        save_path=None
    )

    plt.tight_layout()
    fig.savefig(f"{save_path}/figures/figure2.png", bbox_inches='tight', pad_inches=0, dpi=300)
    fig.savefig(f"{save_path}/figures/figure2.eps", bbox_inches='tight', pad_inches=0, format='eps', dpi=300)

def create_two_panel_heatmap_figure(
    data,
    save_path: str,
    models_list: list,
    model_order: list,
    width_per_model: float = 0.9,
    height_per_category: float = 0.9,
    cbar_frac: float = 0.06,
    wspace: float = 0.00,
    fig_width: float | None = None,
    fig_height: float | None = None,
):
    # Count columns and rows for sizing
    n_models = sum(1 for m in model_order if m in models_list)

    cats = set()
    for cat_list in data['bio_category']:
        if pd.isna(cat_list):
            continue
        cats.update([s.strip() for s in str(cat_list).split(';')])
    n_rows = len(cats) if cats else 10

    # Compute figure size if not provided
    heatmap_width = n_models * width_per_model
    cbar_width = max(0.8, heatmap_width * cbar_frac)

    if fig_width is None:
        # two heatmaps + two cbar columns + a little padding
        fig_width = 2 * heatmap_width + 2 * cbar_width + 1.0
    if fig_height is None:
        fig_height = max(6.5, n_rows * height_per_category)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[heatmap_width, heatmap_width, cbar_width],
        wspace=wspace
    )

    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])
    cax2 = fig.add_subplot(gs[0, 2])
    plt.subplots_adjust(wspace=0.0)

    # Subplot labels
    ax1.text(0.00, 1.01, "(A)", transform=ax1.transAxes, fontsize=20, fontweight='bold')
    ax2.text(0.96,  1.01, "(B)", transform=ax2.transAxes, fontsize=20, fontweight='bold')

    # Left heatmap: no legend, but keep its reserved cbar slot
    plot_heatmap(
        data=data,
        metric='BioScore',
        models=models_list,
        model_order=model_order,
        category='bio_category',
        title='Quality Rate by Bio Category',
        save_path=None,
        calculation_type='quality_rate',
        threshold=5,
        ax=ax1,
        include_legend=False,
    )

    # Right heatmap: legend on; hide y-ticks to reduce middle clutter
    plot_heatmap(
        data=data,
        metric='BioScore',
        models=models_list,
        model_order=model_order,
        category='bio_category',
        title='Safety Rate by Bio Category',
        save_path=None,
        calculation_type='safety_rate',
        threshold=5,
        ax=ax2,
        include_yticks=False,
        include_legend=True,
        cbar_ax=cax2,
    )

    fig.savefig(f"{save_path}/figures/figure4.png", bbox_inches='tight', pad_inches=0, dpi=300)
    fig.savefig(f"{save_path}/figures/figure4.eps", bbox_inches='tight', pad_inches=0, format='eps', dpi=300)
