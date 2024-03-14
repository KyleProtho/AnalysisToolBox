# Load packages
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def PlotDotPlot(dataframe,
                categorical_column_name,
                value_column_name,
                group_column_name,
                # Dot formatting arguments
                group_1_color="#4257f5",
                group_2_color="#ccd2ff",
                dot_size=20,
                dot_alpha=1,
                zero_line_group=None,
                # Connecting line formatting arguments
                connect_dots=True,
                connect_line_color="#666666",
                connect_line_alpha=0.4,
                connect_line_width=1.0,
                connect_line_style='dashed',
                # Connecting line label formatting arguments
                dict_of_connect_line_labels=None,
                connect_line_label_fontsize=11,
                connect_line_label_fontsize_fontweight='bold',
                connect_line_label_color="#262626",
                show_connect_line_labels_in_margin=False,
                connect_line_label_margin=0.5,
                # Plot formatting arguments
                display_order_list=None,
                figure_size=(10, 6),
                show_x_axis=False,
                # Data label arguments
                show_data_labels=True,
                decimal_places_for_data_label=1,
                data_label_fontsize=11,
                data_label_fontweight='bold',
                data_label_color="#262626",
                # Text formatting arguments
                title_for_plot=None,
                subtitle_for_plot=None,
                caption_for_plot=None,
                data_source_for_plot=None,
                title_y_indent=1.15,
                subtitle_y_indent=1.1,
                caption_y_indent=-0.15,
                # Plot saving arguments
                filepath_to_save_plot=None):
    """
    Plot a dot plot with optional connecting lines.

    Args:
        dataframe (pd.DataFrame): The input dataframe.
        categorical_column_name (str): The name of the categorical column in the dataframe.
        value_column_name (str): The name of the value column in the dataframe.
        group_column_name (str): The name of the group column in the dataframe.
        group_1_color (str, optional): The color for group 1 dots. Defaults to "#4257f5".
        group_2_color (str, optional): The color for group 2 dots. Defaults to "#ccd2ff".
        dot_size (float, optional): The size of the dots in the plot. Defaults to 2.
        dot_alpha (float, optional): The transparency of the dots in the plot. Defaults to 1.
        connect_dots (bool, optional): Whether to connect the dots with lines. Defaults to True.
        connect_line_color (str, optional): The color of the connecting lines. Defaults to "#666666".
        connect_line_alpha (float, optional): The transparency of the connecting lines. Defaults to 0.4.
        connect_line_width (float, optional): The width of the connecting lines. Defaults to 1.0.
        connect_line_style (str, optional): The style of the connecting lines. Defaults to 'dashed'.
        dict_of_connect_line_labels (dict, optional): A dictionary of labels for the connecting lines. 
            The keys should be the categories in the dataframe and the values should be the labels. 
            Defaults to None.
        connect_line_label_fontsize (int, optional): The fontsize of the connecting line labels. Defaults to 11.
        connect_line_label_fontsize_fontweight (str, optional): The fontweight of the connecting line labels. 
            Defaults to 'bold'.
        connect_line_label_color (str, optional): The color of the connecting line labels. Defaults to "#262626".
        zero_line_group (str, optional): The group to use as the zero line reference. 
            The values in the value column will be relative to this group. Defaults to None.
        display_order_list (list, optional): The order in which the categories should be displayed. 
            If not provided, the categories will be sorted by the difference between the two groups. 
            Defaults to None.
        figure_size (tuple, optional): The size of the plot figure. Defaults to (10, 6).
        show_legend (bool, optional): Whether to show the legend. Defaults to True.
        show_data_labels (bool, optional): Whether to show data labels on the dots. Defaults to True.
        show_connect_line_labels_in_margin (bool, optional): Whether to show the data labels in the margin to the right of the plot.
        decimal_places_for_data_label (int, optional): The number of decimal places to round the data labels. 
            Defaults to 1.
        data_label_fontsize (int, optional): The fontsize of the data labels. Defaults to 11.
        data_label_fontweight (str, optional): The fontweight of the data labels. Defaults to 'bold'.
        data_label_color (str, optional): The color of the data labels. Defaults to "#262626".
        title_for_plot (str, optional): The title for the plot. Defaults to None.
        subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to None.
        caption_for_plot (str, optional): The caption for the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
        title_y_indent (float, optional): The y-axis indentation for the title. Defaults to 1.15.
        subtitle_y_indent (float, optional): The y-axis indentation for the subtitle. Defaults to 1.1.
        caption_y_indent (float, optional): The y-axis indentation for the caption. Defaults to -0.15.
        filepath_to_save_plot (str, optional): The filepath to save the plot. Defaults to None.

    Returns:
        None
    """
    # Ensure that the group column only has two unique values
    if len(dataframe[group_column_name].unique()) != 2:
        raise ValueError("group_column_name must have exactly two unique values.")
    
    # Ensure that the zero_line_group column is in the group column
    if zero_line_group != None:
        if zero_line_group not in dataframe[group_column_name].unique():
            raise ValueError("zero_line_group must a value in the " + group_column_name + " column.")
        
    # Ensure that each row is a unique combination of the categorical and group columns
    if len(dataframe[[categorical_column_name, group_column_name]].drop_duplicates()) != len(dataframe):
        raise ValueError("Each row in the dataframe must be a unique combination of the categorical and group columns.")
    
    # Ensure that connect_line_label_margin is between 0 and 1
    if connect_line_label_margin < 0 or connect_line_label_margin > 1:
        raise ValueError("connect_line_label_margin must be between 0 and 1.")
    
    # Drop missing values from the dataframe and across the columns
    dataframe = dataframe.dropna(subset=[categorical_column_name, value_column_name, group_column_name])
    
    # Make a copy of the value column, add ' - Original' to the name
    dataframe[value_column_name + '- Original'] = dataframe[value_column_name]
    
    # If zero_line_group is provided
    if zero_line_group != None:
        # Sort the dataframe so that it is the first group
        dataframe[group_column_name] = pd.Categorical(dataframe[group_column_name], categories=[zero_line_group] + [x for x in dataframe[group_column_name].unique() if x != zero_line_group])
        dataframe = dataframe.sort_values(group_column_name)
          
        # Left join the original values of the zero_line_group to the dataframe
        df_temp = pd.merge(
            dataframe,
            dataframe[dataframe[group_column_name] == zero_line_group][[categorical_column_name, value_column_name]],
            on=categorical_column_name,
            how='left',
            suffixes=('', '_original')
        )
        
        # Reset the values of the other group to be the difference between the original and zero_line_group values
        df_temp[value_column_name] = df_temp[value_column_name] - df_temp[value_column_name + '_original']
        
        # Set the values for the zero_line_group to zero
        df_temp.loc[df_temp[group_column_name] == zero_line_group, value_column_name] = 0
        
        # Set the dataframe to the new dataframe
        dataframe = df_temp
    
    # If display_order_list is provided, check that it contains all of the categories in the dataframe
    if display_order_list != None:
        if not set(display_order_list).issubset(set(dataframe[categorical_column_name].unique())):
            raise ValueError("display_order_list must contain all of the categories in the dataframe.")
    else:
        # If display_order_list is not provided, create one from the dataframe using the difference between the two groups
        df_temp = pd.merge(
            dataframe[dataframe[group_column_name] == dataframe[group_column_name].unique()[0]][[categorical_column_name, value_column_name]],
            dataframe[dataframe[group_column_name] == dataframe[group_column_name].unique()[1]][[categorical_column_name, value_column_name]],
            on=categorical_column_name,
            how='outer',
            suffixes=('_1', '_2')
        )
        df_temp['difference'] = df_temp[value_column_name + '_1'] - df_temp[value_column_name + '_2']
        df_temp = df_temp.sort_values('difference', ascending=False)
        display_order_list = df_temp[categorical_column_name].unique()
    
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=figure_size)
    
    # If connect_dots is True, create horizontal lines connecting the dots
    if connect_dots == True:
        for i in range(len(display_order_list)):
            # Get the x and y coordinates for the dots
            x_coordinates = dataframe[dataframe[categorical_column_name] == display_order_list[i]][value_column_name]
            y_coordinates = dataframe[dataframe[categorical_column_name] == display_order_list[i]][categorical_column_name]
            
            # Get the x and y coordinates for the lines
            x_line_coordinates = [x_coordinates.min(), x_coordinates.max()]
            y_line_coordinates = [y_coordinates.min(), y_coordinates.max()]
            
            # Plot the lines
            plt.plot(
                x_line_coordinates, 
                y_line_coordinates, 
                color=connect_line_color, 
                alpha=connect_line_alpha, 
                linestyle=connect_line_style, 
                linewidth=connect_line_width,
                zorder=1
            )
            
            # Plot the line label
            if dict_of_connect_line_labels is not None:
                # Ensure that the dict_of_connect_line_labels contains all of the categories in the dataframe
                if not set(dict_of_connect_line_labels.keys()).issubset(set(dataframe[categorical_column_name].unique())):
                    raise ValueError("dict_of_connect_line_labels must contain all of the categories in the dataframe.")
                
                # Iterate through the dict_of_connect_line_labels and plot the labels in the middle of the lines
                for key, value in dict_of_connect_line_labels.items():
                    # Get the x and y coordinates for the line of the current category
                    x_coordinates = dataframe[dataframe[categorical_column_name] == key][value_column_name]
                    y_coordinates = dataframe[dataframe[categorical_column_name] == key][categorical_column_name]
                    
                    # Get the y coordinates for the line label
                    y_line_label_coordinates = y_coordinates.min()
                    
                    # If show_connect_line_labels_in_margin is True, plot the data labels in the margin to the right of the plot
                    if show_connect_line_labels_in_margin == True:
                        # Get the range of the x-axis
                        value_range = dataframe[value_column_name].max() - dataframe[value_column_name].min()
                        # Calculate the bin range of the x-axis
                        bins = 6
                        bin_range = value_range / bins
                        # Add the bin range to the x_line_label_coordinates maximum
                        x_line_label_coordinates = dataframe[value_column_name].max() + (bin_range * connect_line_label_margin)
                    else:
                        # Get the x coordinates for the line label
                        x_line_label_coordinates = (x_coordinates.min() + x_coordinates.max()) / 2
                            
                    # Plot the line label
                    ax.text(
                        x=x_line_label_coordinates,
                        y=y_line_label_coordinates,
                        s=value,
                        # fontname="Arial",
                        fontsize=connect_line_label_fontsize,
                        fontweight=connect_line_label_fontsize_fontweight,
                        color=connect_line_label_color,
                        horizontalalignment='center',
                        verticalalignment='center',
                        bbox=dict(
                            facecolor='white',
                            edgecolor='white',
                            boxstyle='round'
                        ),
                        zorder=2,
                    )
    
    # Create pointplot for each group
    for group in dataframe[group_column_name].unique():
        sns.pointplot(
            data=dataframe[dataframe[group_column_name] == group],
            y=categorical_column_name,
            x=value_column_name,
            order=display_order_list,
            linestyles='none',
            color=group_1_color if group == dataframe[group_column_name].unique()[0] else group_2_color,
            markers='o',
            markersize=dot_size,
            alpha=dot_alpha,
            ax=ax,
            zorder=3 if group == dataframe[group_column_name].unique()[0] else 5,
        )
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Wrap y axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(categorical_column_name, 40, break_long_words=False)) 
    ax.set_ylabel(wrapped_variable_name)
    
    # Format and wrap y axis tick labels using textwrap
    y_tick_labels = ax.get_yticklabels()
    wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 40, break_long_words=False)) for label in y_tick_labels]
    ax.set_yticklabels(
        wrapped_y_tick_labels,
        fontsize=10, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Move x-axis to the top
    ax.xaxis.tick_top()
    if show_x_axis == False:
        ax.xaxis.set_visible(False)
        ax.spines['top'].set_color("white")
    else:
        ax.xaxis.set_visible(True)
        # Change x-axis colors to "#666666"
        ax.tick_params(axis='x', colors="#666666")
        ax.spines['top'].set_color("#666666")
    
    # Remove bottom, left, and right spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add data labels
    if show_data_labels:
        # Iterate through each category
        for category in dataframe[categorical_column_name].unique():
            df_temp = dataframe[dataframe[categorical_column_name] == category]
            # Iterate through each group
            for group in df_temp[group_column_name].unique():
                df_temp_2 = df_temp[df_temp[group_column_name] == group]
                
                # Get the x and y coordinates for the dots
                x_coordinates = df_temp_2[df_temp_2[group_column_name] == group][value_column_name]
                y_coordinates = df_temp_2[df_temp_2[group_column_name] == group][categorical_column_name]
                
                # Get the x and y coordinates for the data labels
                x_data_label_coordinates = x_coordinates
                y_data_label_coordinates = y_coordinates
                
                # If the current group is the zero_line_group, do not create labels for it
                if group == zero_line_group or group == dataframe[group_column_name].unique()[0]:
                    zorder = 4
                else:
                    zorder = 6
            
                # Get the x and y coordinates for the data labels of the higher values
                x_data_label_coordinates = x_coordinates.max()
                y_data_label_coordinates = y_coordinates.max()
            
                # Plot the data labels for the higher values
                ax.text(
                    x=x_data_label_coordinates,
                    y=y_data_label_coordinates,
                    s=round(df_temp_2[df_temp_2[group_column_name] == group][value_column_name + '- Original'].max(), decimal_places_for_data_label),
                    # fontname="Arial",
                    fontsize=data_label_fontsize,
                    fontweight=data_label_fontweight,
                    color=data_label_color,
                    horizontalalignment='center',
                    verticalalignment='center',
                    zorder=zorder
                )
            
                # Plot the data labels for the lower values
                x_data_label_coordinates = x_coordinates.min()
                y_data_label_coordinates = y_coordinates.min()
            
                # Plot the data labels for the lower values
                ax.text(
                    x=x_data_label_coordinates,
                    y=y_data_label_coordinates,
                    s=round(df_temp_2[df_temp_2[group_column_name] == group][value_column_name + '- Original'].min(), decimal_places_for_data_label),
                    # fontname="Arial",
                    fontsize=data_label_fontsize,
                    fontweight=data_label_fontweight,
                    color=data_label_color,
                    horizontalalignment='center',
                    verticalalignment='center',
                    zorder=zorder
                )
        
    # Set the x indent of the plot titles and captions
    # Get longest y tick label
    longest_y_tick_label = max(wrapped_y_tick_labels, key=len)
    if len(longest_y_tick_label) >= 30:
        x_indent = -0.3
    else:
        x_indent = -0.005 - (len(longest_y_tick_label) * 0.011)
        
    # Set the title with Arial font, size 14, and color #262626 at the top of the plot
    ax.text(
        x=x_indent,
        y=title_y_indent,
        s=title_for_plot,
        # fontname="Arial",
        fontsize=14,
        color="#262626",
        transform=ax.transAxes
    )
    
    # Set the subtitle with Arial font, size 11, and color #666666
    ax.text(
        x=x_indent,
        y=subtitle_y_indent,
        s=subtitle_for_plot,
        # fontname="Arial",
        fontsize=11,
        color="#666666",
        transform=ax.transAxes
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
            # fontname="Arial",
            fontsize=8,
            color="#666666",
            transform=ax.transAxes
        )
        
    # If filepath_to_save_plot is provided, save the plot
    if filepath_to_save_plot != None:
        # Ensure that the filepath ends with '.png' or '.jpg'
        if not filepath_to_save_plot.endswith('.png') and not filepath_to_save_plot.endswith('.jpg'):
            raise ValueError("The filepath to save the plot must end with '.png' or '.jpg'.")
        
        # Save plot
        plt.savefig(
            filepath_to_save_plot,
            bbox_inches="tight"
        )
        
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()

