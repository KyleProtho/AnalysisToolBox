library(ggplot2)
library(dplyr)
library(stringr)
library(table1)

PlotSingleVariableHistogram=function(data,
                                     quantitative_variable,
                                     readable_variable_label=NULL, 
                                     fill_color="#4f92ff") {
  # Use Scott method to get number of bins
  max_value = max(data[[quantitative_variable]], na.rm=TRUE)
  min_value = min(data[[quantitative_variable]], na.rm=TRUE)
  sd_value = sd(data[[quantitative_variable]], na.rm=TRUE)
  obs_count = nrow(data)
  scott_number_bins_numer = (max_value - min_value) * obs_count ^ (1/3)
  scott_number_bins_denom = 3.5 * sd_value
  scott_number_bins = scott_number_bins_numer / scott_number_bins_denom
  scott_number_bins = round(scott_number_bins, 0)
  
  # Use Scott method to get bin width
  scott_bin_width_numer=3.5 * sd_value
  scott_bin_width_denom=obs_count ^ (1/3)
  scott_bin_width = scott_bin_width_numer / scott_bin_width_denom
  scott_bin_width = ifelse(scott_bin_width < 1,
                           1,
                           round(scott_bin_width, 0))
  
  # Set label
  if (is.null(readable_variable_label)) {
    readable_variable_label = quantitative_variable
  }
  
  # Generate title for plot
  title_for_plot = paste("Distribution")
  title_for_plot = str_wrap(title_for_plot, width=60)
  
  # Draw plot
  p = ggplot(data, aes(x=data[[quantitative_variable]])) + 
    geom_histogram(fill=fill_color,
                   alpha=0.8,
                   bins=scott_number_bins,
                   binwidth=scott_bin_width) +
    labs(title=title_for_plot,
         subtitle=readable_variable_label,
         y="Count") + 
    theme_minimal() + 
    theme(axis.line=element_line(size=1, colour="#d6d6d6"),
          axis.text=element_text(color="#3b3b3b", size=10),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          axis.title.x=element_blank()
    )
  plot(p)
}

