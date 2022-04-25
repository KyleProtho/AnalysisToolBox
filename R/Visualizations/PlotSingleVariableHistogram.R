library(ggplot2)
library(dplyr)
library(stringr)
library(table1)

PlotSingleVariableHistogram=function(dataframe,
                                     quantitative_variable,
                                     readable_variable_label=NULL, 
                                     fill_color="#4f92ff") {
  # Use Scott method to get number of bins
  max_value = max(dataframe[[quantitative_variable]], na.rm=TRUE)
  min_value = min(dataframe[[quantitative_variable]], na.rm=TRUE)
  sd_value = sd(dataframe[[quantitative_variable]], na.rm=TRUE)
  obs_count = nrow(dataframe)
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
  
  # Draw plot
  p = ggplot(data=dataframe, 
             aes(x=dataframe[[quantitative_variable]])) + 
    geom_histogram(fill=fill_color,
                   color="white",
                   alpha=0.8,
                   bins=scott_number_bins,
                   binwidth=scott_bin_width) +
    labs(title="Distribution",
         subtitle=readable_variable_label,
         y="Count") + 
    theme_minimal() + 
    theme(axis.line=element_line(size=1, colour="#d6d6d6"),
          axis.text=element_text(color="#3b3b3b"),
          axis.text.x=element_text(size=12),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          axis.title.x=element_blank()
    )
  plot(p)
}
