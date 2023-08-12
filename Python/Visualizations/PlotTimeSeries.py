import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import textwrap

sns.set(style="white",
        font="Arial",
        context="paper")

def PlotTimeSeries(dataframe,
                   numeric_variable,
                   time_variable,
                   group_variable=None,
                   # Line formatting arguments
                   line_color="#3269a8",
                   line_alpha=0.8,
                   color_palette="Set2",
                   markers="o",
                   # Text formatting arguments
                   title_for_plot=None,
                   subtitle_for_plot=None,
                   caption_for_plot=None,
                   data_source_for_plot=None,
                   x_indent=-0.127,
                   title_y_indent=1.125,
                   subtitle_y_indent=1.05,
                   caption_y_indent=-0.3,
                   # Plot formatting arguments
                   figure_size=(8, 5)):
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Use Seaborn to create a line plot
    if group_variable is None:
        sns.lineplot(
            data=dataframe,
            x=time_variable,
            y=numeric_variable,
            color=line_color,
            alpha=line_alpha,
            marker=markers,
            ax=ax
        )
    else:
        sns.lineplot(
            data=dataframe,
            x=time_variable,
            y=numeric_variable,
            hue=group_variable,
            palette=color_palette,
            alpha=line_alpha,
            marker=markers,
            ax=ax
        )
    
    # Remove top and right spines, and set bottom and left spines to gray
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#666666')
    ax.spines['left'].set_color('#666666')

    # Format tick labels to be Arial, size 9, and color #666666
    ax.tick_params(
        which='major',
        labelsize=9,
        color='#666666'
    )
    
    # Set the title with Arial font, size 14, and color #262626 at the top of the plot
    ax.text(
        x=x_indent,
        y=title_y_indent,
        s=title_for_plot,
        fontname="Arial",
        fontsize=14,
        color="#262626",
        transform=ax.transAxes
    )
    
    # Word wrap the subtitle without splitting words
    if subtitle_for_plot != None:   
        subtitle_for_plot = textwrap.fill(subtitle_for_plot, 100, break_long_words=False)
    # Set the subtitle with Arial font, size 11, and color #666666
    ax.text(
        x=x_indent,
        y=subtitle_y_indent,
        s=subtitle_for_plot,
        fontname="Arial",
        fontsize=11,
        color="#666666",
        transform=ax.transAxes
    )
    
    # Move the y-axis label to the top of the y-axis, and set the font to Arial, size 9, and color #666666
    ax.yaxis.set_label_coords(-0.1, 0.84)
    ax.yaxis.set_label_text(
        numeric_variable,
        fontname="Arial",
        fontsize=10,
        color="#666666"
    )
    
    # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
    ax.xaxis.set_label_coords(0.9, -0.1)
    ax.xaxis.set_label_text(
        time_variable,
        fontname="Arial",
        fontsize=10,
        color="#666666"
    )
    
    # Add a word-wrapped caption if one is provided
    if caption_for_plot != None or data_source_for_plot != None:
        # Create starting point for caption
        wrapped_caption = ""
        
        # Add the caption to the plot, if one is provided
        if caption_for_plot != None:
            # Word wrap the caption without splitting words
            wrapped_caption = textwrap.fill(caption_for_plot, 110, break_long_words=False)
            
        # Add the data source to the caption, if one is provided
        if data_source_for_plot != None:
            wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
        
        # Add the caption to the plot
        ax.text(
            x=x_indent,
            y=caption_y_indent,
            s=wrapped_caption,
            fontname="Arial",
            fontsize=8,
            color="#666666",
            transform=ax.transAxes
        )

    # Show plot
    plt.show()
    

# # Test the function
# time_series = pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')
# PlotTimeSeries(
#     dataframe=time_series,
#     numeric_variable='Consumption',
#     time_variable='Date',
#     title_for_plot='Daily Electricity Consumption in Germany',
#     data_source_for_plot='https://raw.githubusercontent.com/jenfly',
# )
