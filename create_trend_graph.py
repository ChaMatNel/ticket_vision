import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

def create_trend_graph(df, file_path):

    # Define the desired order for the seat_location
    seat_location_order = ["upper left", "upper center", "upper right", 
                        "middle left", "middle center", "middle right", 
                        "lower left", "lower center", "lower right"]

    # Set the theme with a white background
    sns.set_theme(style="whitegrid")  # This will set the background to white with gridlines

    # Create a custom color palette from green to dark red
    palette = sns.color_palette("RdYlGn_r", as_cmap=True)

    # Create a FacetGrid with independent y-axes for each plot and individual x-axes
    g = sns.FacetGrid(df, col="seat_location", col_wrap=3, height=4, aspect=1.75, sharex=False, sharey=False,
                    col_order=seat_location_order)  # Specify the desired order

    # Apply stripplot to each facet with hue for color scaling based on "deal_score"
    g.map(
        sns.stripplot, "days_to_game", "price", hue="deal_score", data=df, 
        zorder=1, palette=palette
    )

    # Add a regression line to each subplot
    g.map(
        sns.regplot, "days_to_game", "price", data=df,  
        scatter=False, truncate=False, order=2, color=".2"
    )

    # Set titles and labels
    g.set_titles("{col_name}")
    g.set_axis_labels("Days to Game", "Price")

    # Reverse the x-axis for each subplot
    for ax in g.axes.flat:
        ax.set_xlim(ax.get_xlim()[::-1])  # Reverse the x-axis
        ax.set_facecolor("white")  # Ensure each facet's background is white

    # Adjust layout
    plt.tight_layout()

    # Save the plot with a transparent background
    plt.savefig(file_path, transparent=False)