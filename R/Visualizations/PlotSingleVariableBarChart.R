
# Load packages
library(ggplot2)
library(RColorBrewer)

# Define function
PlotSingleVariableBarChart <- function(dataframe,
                                       categorical_variable,
                                       readable_variable_label=NULL,
                                       fill_color=NULL) {
  # Concert categorical variable to factor
  dataframe[[categorical_variable]] = as.factor(dataframe[[categorical_variable]])
  
  # Set label
  if (is.null(readable_variable_label)) {
    readable_variable_label = quantitative_variable
  }
  
  # Create bar chart
  if (is.null(fill_color)) {
    p = ggplot(data=dataframe, 
               aes(x=dataframe[[categorical_variable]], 
                   fill=dataframe[[categorical_variable]])) +
      geom_bar(stat="count",
               width=0.75,
               alpha=0.8) +
      theme_minimal() +
      scale_fill_brewer(palette="Set2")
  } else {
    p = ggplot(data=dataframe, 
               aes(x=dataframe[[categorical_variable]])) +
      geom_bar(stat="count", 
               width=0.75,
               alpha=0.8,
               fill=fill_color) +
      theme_minimal()
  }
  
  # Fix axis, add labels, and make bar chart horizontal
  p = p + coord_cartesian(clip = "off") + 
    theme(axis.line.y=element_line(size=1, colour="#d6d6d6"),
          axis.text.y=element_text(color="#3b3b3b", size=12),
          axis.line.x=element_blank(),
          axis.text.x=element_blank(),
          axis.title=element_blank(), 
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          legend.position = "none") +
    labs(title="Distribution",
         subtitle=readable_variable_label) + 
    geom_text(aes(label=paste0(..count.., " (", round(..count../sum(..count..)*100, 2), "%)")), 
              stat="count",
              color="#3b3b3b",
              hjust=1.1) + 
    coord_flip()
  
  plot(p)
}

