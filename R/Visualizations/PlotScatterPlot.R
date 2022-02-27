# Load packages
library(ggplot2)
library(stringr)

# Defin function
PlotScatterPlot = function(dataframe,
                           x_axis_variable,
                           y_axis_variable,
                           point_color = "#2f7fff",
                           x_axis_label = NULL,
                           y_axis_label = NULL,
                           add_smoothing_line = TRUE,
                           smoothing_line_color = "#ffaf2f",
                           subtitle_text = NULL) {
  # Set labels if none specified
  if (is.null(x_axis_label)) {
    x_axis_label = x_axis_variable
  }
  if (is.null(y_axis_label)) {
    y_axis_label = y_axis_variable
  }
  
  # Set title
  title_for_plot = paste("Relationship between", x_axis_label, "and", y_axis_label)
  
  # Draw plot
  p = ggplot(data=dataframe, 
             aes(x=.data[[x_axis_variable]],
                 y=.data[[y_axis_variable]])) + 
    geom_point(color=point_color,
               fill=point_color,
               alpha=0.50,
               size=2) + 
    theme_minimal() +
    theme(panel.grid.major=element_blank(),
          panel.grid.minor=element_blank()) +
    labs(title=str_wrap(title_for_plot, width = 80),
         subtitle=str_wrap(subtitle_text, width = 110),
         x=x_axis_label,
         y=y_axis_label)
  
  # Add smoothing line if requested
  if (add_smoothing_line) {
    p = p + geom_smooth(color=smoothing_line_color,
                        alpha = 0.30,
                        fill=smoothing_line_color)
  }
  
  # Return plot
  plot(p)
  return(p)
}
