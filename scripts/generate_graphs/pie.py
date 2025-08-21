import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_category_pie_chart(data: pd.DataFrame, category: str, title: str, save_path: str, color_flag: int, ax: plt.Axes = None):
    """Create a (donut) pie chart to visualize the distribution of categories in the dataset."""

    # # Set the font for consistent style
    # plt.rcParams.update({
    #     'font.family': 'DejaVu Sans',
    #     'font.size': 16,
    #     'axes.titlesize': 16,
    #     'axes.labelsize': 16
    # })

    # Prepare category labels
    exploded_data = data.copy()
    exploded_data[category] = exploded_data[category].apply(lambda x: [item.strip() for item in x.split(';')])
    exploded_data = exploded_data.explode(category).reset_index(drop=True)
    category_counts = exploded_data[category].value_counts()

    labels = category_counts.index.tolist()
    sizes = category_counts.values

    # Choose color palette
    if color_flag == 1:
        colors = sns.color_palette("Pastel1", len(labels))
    elif color_flag == 2:
        colors = sns.color_palette("Pastel2", len(labels))
    else:
        colors = sns.color_palette("pastel", len(labels))

    explode = [0.1] * len(labels)

    # Create figure/axes if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the donut chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        labels=labels,
        autopct='%1.1f%%',
        startangle=0,
        pctdistance=0.85,
        explode=explode,
        labeldistance=1.1
    )

    # Manually apply font settings to autotexts based on rcParams
    fontsize = plt.rcParams.get('font.size')
    fontfamily = plt.rcParams.get('font.family')[0]
    for autotext in autotexts:
        autotext.set_fontsize(fontsize)
        autotext.set_fontname(fontfamily)

    # Draw white circle for donut
    centre_circle = plt.Circle((0, 0), 0.75, fc='white')
    ax.add_artist(centre_circle)
    ax.set_title(title)
    ax.set_aspect('equal')

    if save_path and ax is None:
        fig.tight_layout()
        fig.savefig(f'{save_path}/{title}.png', bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)