import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_category_pie_chart(data: pd.DataFrame, category: str, title: str, save_path: str):
    """Create a pie chart to visualize the distribution of categories in the dataset."""
    
    # Explode the category column to handle multiple labels per entry
    exploded_data = data.copy()
    exploded_data[category] = exploded_data[category].apply(lambda x: [item.strip() for item in x.split(';')])
    exploded_data = exploded_data.explode(category).reset_index(drop=True)
    
    # Calculate the number of occurrences of each category
    category_counts = exploded_data[category].value_counts()
    
    # Labels for the pie chart
    labels = category_counts.index.tolist()
    
    # Sizes for each slice (proportional to the number of occurrences)
    sizes = category_counts.values
    
    # Define colors for the pie chart
    colors = sns.color_palette("pastel")[0:len(labels)]
    
    # Define explode values (to slightly separate each slice)
    explode = [0.1] * len(labels)
    
    # Plot the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=0, pctdistance=0.85, explode=explode, labeldistance=1.1)
    
    # Draw circle to make it a donut chart
    centre_circle = plt.Circle((0, 0), 0.75, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.gca().set_aspect('equal')
    
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}.png')
    plt.close()