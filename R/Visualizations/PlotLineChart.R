# Load packages
library(ggplot2)
library(stringr)
library(dplyr)

# Define function
PlotLineChart = function(dataframe,
                         time_column_name,
                         quantitative_variable_column_name,
                         line_color="#2f7fff",
                         right_margin_in_cm=3,
                         quantitative_variable_label=NULL,
                         title_text=NULL,
                         subtitle_text=NULL) {
  # Set readable quantity label if none set
  if (is.null(quantitative_variable_label)) {
    quantitative_variable_label = quantitative_variable_column_name
  }
  
  # Set title for plot if none specified
  if (is.null(title_text)) {
    title_text = paste(quantitative_variable_label, "over time")
  }
  
  # Draw plot
  p = ggplot() +
    geom_line(data = dataframe,
              aes(x = .data[[time_column_name]],
                  y = .data[[quantitative_variable_column_name]]),
              color = line_color,
              size = 1.33) +
    geom_point(data = dataframe,
               aes(x = .data[[time_column_name]],
                   y = .data[[quantitative_variable_column_name]]),
               color = line_color,
               fill = line_color,
               size = 3) + 
    labs(title = str_wrap(title_text, width = 80),
         subtitle = str_wrap(subtitle_text, width = 110),
         x="Time",
         y="") +
    theme_minimal() +
    theme(legend.position = "none",
          panel.grid.minor=element_blank())
  plot(p)
  
  # Return plot
  return(p)
}
