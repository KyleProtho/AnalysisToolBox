import pandas as pd
import plotly.express as px

# Create animated scatter plot using plotly express
def CreateAnimatedScatterPlot(dataframe,
                              x_axis_variable,
                              y_axis_variable,
                              subject_id_variable,
                              animation_frame_variable,
                              size_by=None,
                              color_by=None,
                              name_to_display=None,
                              range_of_x_axis=None,
                              range_of_y_axis=None):
    # Set hover name to display
    if name_to_display == None:
        name_to_display = subject_id_variable
        
    # Generate animated scatter plot
    p = px.scatter(
        data_frame=dataframe, 
        x=x_axis_variable, 
        y=y_axis_variable,
        animation_frame=animation_frame_variable, 
        animation_group=subject_id_variable,
        color=color_by,
        size=size_by,
        hover_name=name_to_display,
        range_x=range_of_x_axis, 
        range_y=range_of_y_axis
    )
    
    # Return plot
    return(p)
