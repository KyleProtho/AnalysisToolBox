# Load packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def ProbabilityOfAtLeastOne(probability_of_event,
                            number_of_events,
                            format_as_percent=False,
                            # Plotting arguments
                            show_plot=True,
                            # Line formatting arguments
                            line_color="#3269a8",
                            line_alpha=0.8,
                            risk_tolerance=None,
                            risk_tolerance_color="#cc453b",
                            # Text formatting arguments
                            number_of_x_axis_ticks=None,
                            x_axis_tick_rotation=None,
                            title_for_plot="Probability of at Least One Event",
                            subtitle_for_plot="Shows the probability of at least one event occurring, given the probability of an event and the number of events.",
                            caption_for_plot=None,
                            data_source_for_plot=None,
                            x_indent=-0.127,
                            title_y_indent=1.125,
                            subtitle_y_indent=1.05,
                            caption_y_indent=-0.3,
                            # Plot formatting arguments
                            figure_size=(7, 6)):
    """
    Calculates the probability of at least one event occurring, given the probability of an event and the number of events.

    Args:
        probability_of_event (float): The probability of an event occurring.
        number_of_events (int): The number of events.
        format_as_percent (bool, optional): Whether to format the probability as a percentage. Defaults to False.
        show_plot (bool, optional): Whether to show the plot of the probability of at least one event. Defaults to True.
        line_color (str, optional): The color of the line in the plot. Defaults to "#3269a8".
        line_alpha (float, optional): The transparency of the line in the plot. Defaults to 0.8.
        risk_tolerance (float, optional): The risk tolerance for the probability of at least one event. Defaults to None.
        risk_tolerance_color (str, optional): The color of the risk tolerance dot and lines in the plot. Defaults to "#cc453b".
        number_of_x_axis_ticks (int, optional): The number of ticks on the x-axis in the plot. Defaults to None.
        x_axis_tick_rotation (float, optional): The rotation angle of the x-axis tick labels in the plot. Defaults to None.
        title_for_plot (str, optional): The title of the plot. Defaults to "Probability of at Least One Event".
        subtitle_for_plot (str, optional): The subtitle of the plot. Defaults to "Shows the probability of at least one event occurring, given the probability of an event and the number of events.".
        caption_for_plot (str, optional): The caption of the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
        x_indent (float, optional): The x-coordinate indent for the plot elements. Defaults to -0.127.
        title_y_indent (float, optional): The y-coordinate indent for the title in the plot. Defaults to 1.125.
        subtitle_y_indent (float, optional): The y-coordinate indent for the subtitle in the plot. Defaults to 1.05.
        caption_y_indent (float, optional): The y-coordinate indent for the caption in the plot. Defaults to -0.3.
        figure_size (tuple, optional): The size of the plot figure. Defaults to (7, 6).

    Returns:
        float: The probability of at least one event occurring.
    """
    
    # Calculate the probability of at least one event
    probability_of_at_least_one_event = 1 - ((1 - probability_of_event) ** number_of_events)
    
    # If format_as_percent is True, format the probability as a percent
    if format_as_percent:
        probability_of_at_least_one_event = "{:.2%}".format(probability_of_at_least_one_event)
    
    # If show_plot is True, plot the probability of at least one event
    if show_plot:
        # Create dataframe of probability of at least one event
        dataframe = pd.DataFrame({
            "Number of Events": range(1, number_of_events + 1),
            "Probability of at Least One Event": [1 - ((1 - probability_of_event) ** x) for x in range(1, number_of_events + 1)]
        })
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Use Seaborn to create a line plot
        sns.lineplot(
            data=dataframe,
            x="Number of Events",
            y="Probability of at Least One Event",
            color=line_color,
            alpha=line_alpha,
            ax=ax
        )
        
        # If risk_tolerance is not None, plot the risk tolerance as a dot on the line plot
        if risk_tolerance is not None:
            # Calulate the number of events that meet the risk tolerance
            number_of_events_meeting_risk_tolerance = 0
            while 1 - ((1 - probability_of_event) ** number_of_events_meeting_risk_tolerance) < risk_tolerance:
                number_of_events_meeting_risk_tolerance += 1
            
            # Plot the risk tolerance as a dot on the line plot
            sns.scatterplot(
                x=[number_of_events_meeting_risk_tolerance],
                y=[risk_tolerance],
                color=risk_tolerance_color,
                ax=ax,
                marker="o",
                s=200
            )
            
            # Add a dashed line from the x-axis to the risk tolerance dot
            ax.plot(
                [number_of_events_meeting_risk_tolerance, number_of_events_meeting_risk_tolerance],
                [0, risk_tolerance],
                color=risk_tolerance_color,
                linestyle="dashed"
            )
            
            # Add a dashed line from the y-axis to the risk tolerance dot
            ax.plot(
                [0, number_of_events_meeting_risk_tolerance],
                [risk_tolerance, risk_tolerance],
                color=risk_tolerance_color,
                linestyle="dashed"
            )
            
            # Add the number of events that meet the risk tolerance to the plot
            ax.text(
                x=number_of_events_meeting_risk_tolerance + (max(dataframe["Number of Events"]) * .01),
                y=0,
                # Format the number of events that meet the risk tolerance with a comma
                s="Number of Events: {:,}".format(number_of_events_meeting_risk_tolerance),
                # fontname="Arial",
                fontsize=10,
                color=risk_tolerance_color
            )
            
            # Add risk tolerance label to the plot
            ax.text(
                x=0,
                y=risk_tolerance + (max(dataframe["Probability of at Least One Event"]) * .025),
                s="Risk Tolerance: " + "{:.1%}".format(risk_tolerance),
                # fontname="Arial",
                fontsize=10,
                color=risk_tolerance_color
            )
        
        # Remove top and right spines, and set bottom and left spines to gray
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        
        # Remove legend
        ax.legend().remove()
        
        # Set the number of ticks on the x-axis
        if number_of_x_axis_ticks is not None:
            ax.xaxis.set_major_locator(plt.MaxNLocator(number_of_x_axis_ticks))
            
        # Rotate x-axis tick labels
        if x_axis_tick_rotation is not None:
            plt.xticks(rotation=x_axis_tick_rotation)

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
            # fontname="Arial",
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )
        
        # Word wrap the subtitle without splitting words
        if subtitle_for_plot != None:   
            subtitle_for_plot = textwrap.fill(subtitle_for_plot, 80, break_long_words=False)
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
        
        # Move the y-axis label to the top of the y-axis, and set the font to Arial, size 9, and color #666666
        ax.yaxis.set_label_coords(-0.1, 0.84)
        ax.yaxis.set_label_text(
            textwrap.fill("Probability of at Least One Event", 20, break_long_words=False),
            # fontname="Arial",
            fontsize=10,
            color="#666666"
        )
        
        # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
        ax.xaxis.set_label_coords(0.9, -0.1)
        ax.xaxis.set_label_text(
            "Number of Events",
            # fontname="Arial",
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
                # fontname="Arial",
                fontsize=8,
                color="#666666",
                transform=ax.transAxes
            )

        # Show plot
        plt.show()
    
    # Return the probability of at least one event
    return probability_of_at_least_one_event

