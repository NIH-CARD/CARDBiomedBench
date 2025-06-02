import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scripts.generate_graphs.pie import plot_category_pie_chart
from scripts.generate_graphs.histogram import plot_token_histograms

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
    fig.savefig(f"{save_path}/figures/figure2.png", dpi=300, bbox_inches='tight', pad_inches=0)
    fig.savefig(f"{save_path}/figures/figure2.eps", format='eps', dpi=300, bbox_inches='tight', pad_inches=0)