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
    Calculate the cumulative probability of at least one event occurring over multiple trials.

    This function computes the probability $P(\text{at least one occurrence}) = 1 - (1 - p)^n$,
    where $p$ is the probability of a single event and $n$ is the number of trials or events.
    This is a fundamental calculation in risk assessment, reliability engineering, and
    epidemiology for understanding cumulative exposure or susceptibility over time.

    Calculating the probability of at least one event is essential for:
      * Epidemiology: Estimating the risk of a disease outbreak over multiple years.
      * Healthcare: Assessing the cumulative risk of drug side effects over a long-term prescription.
      * Intelligence Analysis: Determining the probability of an operation being detected across multiple phases.
      * Cybersecurity: Evaluating the likelihood of a successful system breach over numerous hacking attempts.
      * Engineering: Calculating the failure probability of a redundant system during its lifespan.
      * Finance: Estimating the probability of a market "black swan" event occurring within a specific decade.
      * Environmental Science: Assessing the risk of a 100-year flood occurring at least once in 30 years.
      * Quality Control: Predicting the likelihood of finding at least one defect in a batch of products.

    The function provides an optional visualization of how this probability grows as the
    number of events increases, with support for marking a specific "risk tolerance"
    threshold on the resulting curve.

    Parameters
    ----------
    probability_of_event : float
        The probability of the event occurring in a single trial (value between 0 and 1).
    number_of_events : int
        The total number of trials or events to consider.
    format_as_percent : bool, optional
        Whether to return the result as a formatted percentage string (e.g., "75.00%")
        instead of a raw float. Defaults to False.
    show_plot : bool, optional
        Whether to generate a line plot showing the cumulative probability growth.
        Defaults to True.
    line_color : str, optional
        The hex color code for the probability curve in the plot. Defaults to "#3269a8".
    line_alpha : float, optional
        The transparency level of the plotted line (0 to 1). Defaults to 0.8.
    risk_tolerance : float, optional
        A specific probability threshold (0 to 1) to Highlight on the plot. If provided,
        the function will mark the point where the cumulative risk exceeds this value.
        Defaults to None.
    risk_tolerance_color : str, optional
        The hex color code for the risk tolerance markers and dashed lines.
        Defaults to "#cc453b".
    number_of_x_axis_ticks : int, optional
        The maximum number of tick marks to display on the x-axis. Defaults to None.
    x_axis_tick_rotation : float, optional
        The rotation angle in degrees for the x-axis tick labels. Defaults to None.
    title_for_plot : str, optional
        The main title text displayed above the plot. Defaults to "Probability of at Least One Event".
    subtitle_for_plot : str, optional
        The descriptive text displayed below the main title. Defaults to a standard description.
    caption_for_plot : str, optional
        Additional summary or explanatory text displayed at the bottom of the plot.
        Defaults to None.
    data_source_for_plot : str, optional
        Text identifying the source of the data, appended to the caption. Defaults to None.
    x_indent : float, optional
        Horizontal offset for the plot title and labels. Defaults to -0.127.
    title_y_indent : float, optional
        Vertical offset for the plot title. Defaults to 1.125.
    subtitle_y_indent : float, optional
        Vertical offset for the plot subtitle. Defaults to 1.05.
    caption_y_indent : float, optional
        Vertical offset for the plot caption. Defaults to -0.3.
    figure_size : tuple, optional
        The width and height of the plot in inches. Defaults to (7, 6).

    Returns
    -------
    float or str
        The calculated probability that the event occurs at least once. If `format_as_percent`
        is True, this is returned as a string; otherwise, it is a float.

    Examples
    --------
    # Epidemiology: Probability of at least one infection in a group of 50 people
    # given a 2% transmission rate per contact
    p_inf = ProbabilityOfAtLeastOne(
        probability_of_event=0.02,
        number_of_events=50,
        risk_tolerance=0.5
    )

    # Intelligence: Likelihood of being detected across 10 discrete operation phases
    # with a 5% detection risk per phase
    p_det = ProbabilityOfAtLeastOne(
        probability_of_event=0.05,
        number_of_events=10,
        format_as_percent=True,
        title_for_plot="Operation Compromise Risk",
        line_color="#b0170c"
    )
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

