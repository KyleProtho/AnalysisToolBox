# Load packages
library(ggplot2)
library(stringr)
library(dplyr)

# Define function
PlotTimeSeriesWithHighlightedLine <- function(dataframe,
                                              time_column_name,
                                              quantitative_variable_column_name,
                                              entity_column_name,
                                              entity_to_highlight,
                                              highlight_color="#2f7fff",
                                              right_margin_in_cm=3,
                                              entity_label_wrap_limit=25,
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
    geom_line(data=dataframe,
              aes(x=dataframe[[time_column_name]],
                  y=dataframe[[quantitative_variable_column_name]],
                  group=dataframe[[entity_column_name]]),
              color="#454545",
              alpha=0.20,
              size=0.75) + 
    labs(title = str_wrap(title_text, width = 80),
         subtitle = str_wrap(subtitle_text, width = 110),
         x="Time",
         y="") +
    scale_x_date() +
    coord_cartesian(clip = "off")
  # Add highlighted line
  df_highlight = filter(dataframe, !!sym(entity_column_name) == entity_to_highlight)
  p = p + geom_line(data=df_highlight,
                    aes(x=df_highlight[[time_column_name]],
                        y=df_highlight[[quantitative_variable_column_name]]),
                    color=highlight_color,
                    size=1.5) +
    geom_point(data=df_highlight,
               aes(x=df_highlight[[time_column_name]],
                   y=df_highlight[[quantitative_variable_column_name]]),
               color=highlight_color,
               size=3)
  # Add label to highlighted line
  latest_date = max(dataframe[[time_column_name]])
  latest_date_for_highlight = max(df_highlight[[time_column_name]])
  df_highlight_latest = filter(df_highlight, !!sym(time_column_name) == latest_date_for_highlight)
  p = p + geom_text(data=df_highlight_latest,
                    x=latest_date, 
                    y=df_highlight_latest[[quantitative_variable_column_name]],
                    label=str_wrap(entity_to_highlight, width = entity_label_wrap_limit),
                    color=highlight_color, 
                    fontface="bold",
                    hjust=-.1) +
    theme_minimal() +
    theme(legend.position = "none",
          plot.margin = margin(1,right_margin_in_cm,1,1, "cm"))
  plot(p)
  
  # Return plot
  return(p)
}

# Test
library(sweep)
df_bike_sales = as.data.frame(bike_sales)
df_bike_sales = df_bike_sales %>%
  group_by(
    order.date, 
    model, 
    category.primary, 
    category.secondary
  ) %>%
  summarize(
    total_quantity = sum(quantity)
  )
PlotTimeSeriesWithHighlightedLine(dataframe = df_bike_sales,
                                  time_column_name = "order.date",
                                  quantitative_variable_column_name = "total_quantity",
                                  entity_column_name = "model",
                                  entity_to_highlight = "Slice Hi-Mod Dura Ace D12",
                                  right_margin_in_cm=5)

