# Load packages
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def CreateAssociationRules(dataframe,
                           transaction_id_column,
                           items_column,
                           support_threshold=.01,
                           confidence_threshold=.05,
                           # Plotting parameters
                           plot_lift=True,
                           dot_fill_color="#999999",
                           upper_left_quadrant_fill_color="#32a852",
                           lower_right_quadrant_fill_color="#d14a41",
                           draw_life_equals_1_line=True,
                           # Plot text formatting arguments
                           title_for_plot="Association Rules",
                           subtitle_for_plot="Confidence vs. Consequent Support",
                           caption_for_plot="Lift is a measure of how much more likely it is for two items to occur together than expected by chance. A lift value of 1 means that the two items are independent of each other, while a lift value greater than 1 means that the two items are positively correlated, and a lift value less than 1 means that the two items are negatively correlated.",
                           data_source_for_plot=None,
                           x_indent=-0.128,
                           title_y_indent=1.125,
                           subtitle_y_indent=1.05,
                           caption_y_indent=-0.3,):
    """_summary_
    This function creates association rules from a dataset.
    
    Args:
        dataframe (Pandas dataframe): Pandas dataframe containing the data to be analyzed.
        transaction_id_column (str): The column name containing the transaction ID.
        items_column (str): The column name containing the items.
        support_threshold (float, optional): The threshold for the support for an association rule to be defined. Defaults to .20.
        confidence_threshold (float, optional): The threshold for the confidence for an association rule to be defined. Defaults to .50.
    
    Returns:
        Pandas dataframe: An updated Pandas dataframe with the association rules.
    """
    # Ensure that the support threshold is between 0 and 1
    if support_threshold < 0 or support_threshold > 1:
        raise ValueError("Support threshold must be between 0 and 1.")
    
    # Ensure that the confidence threshold is between 0 and 1
    if confidence_threshold < 0 or confidence_threshold > 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    
    # Group and summarize (as list) items to key column
    dataframe = dataframe.groupby(transaction_id_column)[items_column].apply(list)
    dataframe = dataframe.reset_index()

    # Create association rule mining-ready dataset of transactions
    transactions = TransactionEncoder()
    df_transactions = transactions.fit(dataframe[items_column]).transform(dataframe[items_column])
    df_transactions = pd.DataFrame(df_transactions,
                                columns=transactions.columns_)

    # Create apriori model
    if support_threshold != None:
        df_association_rules = apriori(
            df_transactions, 
            min_support = support_threshold,
            use_colnames = True, 
            verbose = 1
        )
    else:
        df_association_rules = apriori(
            df_transactions,
            use_colnames = True, 
            verbose = 1
        )

    # Create association rules based on apriori model
    df_association_rules = association_rules(
        df_association_rules, 
        metric = "confidence", 
        min_threshold = confidence_threshold
    )
    df_association_rules = df_association_rules.sort_values(
        by=['confidence', 'consequents'],
        ascending=[False, True]
    )
    
    # Create scatter plot of association rules with support on the x-axis and confidence on the y-axis
    if plot_lift:
        ax = sns.scatterplot(
            data=df_association_rules, 
            x="consequent support",
            y="confidence",
            color=dot_fill_color,
            alpha=0.5,
            linewidth=0.5,
            edgecolor=dot_fill_color
        )
        
        # Draw a line where support and confidence are equal
        if draw_life_equals_1_line:
            ax.plot(
                [0, 1], 
                [0, 1],
                linestyle='--',
                color='#666666',
                linewidth=0.5
            )
            
        # Shade area where lift is greater than 1
        ax.fill_between(
            [0, 1], 
            [0, 1], 
            1,
            facecolor=upper_left_quadrant_fill_color,
            alpha=0.15
        )
        
        # Show area where lift is less than 1
        ax.fill_between(
            [0, 1], 
            [0, 1], 
            0,
            facecolor=lower_right_quadrant_fill_color,
            alpha=0.15
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
        
        # Find the middle of the x-axis and y-axis
        x_start = ax.get_xlim()[0]
        x_end = ax.get_xlim()[1]
        y_start = ax.get_ylim()[0]
        y_end = ax.get_ylim()[1]
        x_middle = x_start + (x_end - x_start) / 2
        y_middle = y_start + (y_end - y_start) / 2
        
        # Get coordinates for upper left quadrant label
        upper_left_quadrant_label_x = x_start + (x_middle - x_start) / 2
        upper_left_quadrant_label_x = upper_left_quadrant_label_x + (x_start - upper_left_quadrant_label_x) / 2
        upper_left_quadrant_label_y = y_middle + (y_end - y_middle) / 2
        upper_left_quadrant_label_y = upper_left_quadrant_label_y + (y_end - upper_left_quadrant_label_y) / 2

        # Write text in upper left quadrant
        ax.text(
            upper_left_quadrant_label_x, 
            upper_left_quadrant_label_y, 
            "Higher Lift", 
            fontsize=10,
            fontweight='bold',
            color=upper_left_quadrant_fill_color,
            alpha=0.8,
            ha='center', 
            va='center'
        )
        
        # Get coordinates for lower right quadrant label
        lower_right_quadrant_label_x = x_middle + (x_end - x_middle) / 2
        lower_right_quadrant_label_x = lower_right_quadrant_label_x + (x_end - lower_right_quadrant_label_x) / 2
        lower_right_quadrant_label_y = y_start + (y_middle - y_start) / 2
        lower_right_quadrant_label_y = lower_right_quadrant_label_y - (y_middle - lower_right_quadrant_label_y) / 2
    
        # Write text in lower right quadrant
        ax.text(
            lower_right_quadrant_label_x, 
            lower_right_quadrant_label_y, 
            "Lower Lift", 
            fontsize=10,
            fontweight='bold',
            color=lower_right_quadrant_fill_color,
            alpha=0.8,
            ha='center', 
            va='center'
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
            "Confidence",
            fontname="Arial",
            fontsize=10,
            color="#666666"
        )
        
        # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
        ax.xaxis.set_label_coords(0.85, -0.1)
        ax.xaxis.set_label_text(
            "Consequent Support",
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
            
    # Reformat column names to title case
    df_association_rules.columns = df_association_rules.columns.str.title()

    # Return association rules
    return(df_association_rules)

