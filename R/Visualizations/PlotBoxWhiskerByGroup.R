library(ggplot2)

PlotBoxWhiskerByGroup = function(dataframe,
                                 quantitative_variable,
                                 grouping_variable_1,
                                 grouping_variable_2=NULL) {
  # Generate subtitle
  if (is.null(grouping_variable_2)) {
    subtitle_for_plot = paste(quantitative_variable, "by", grouping_variable_1)
  } else {
    subtitle_for_plot = paste(quantitative_variable, "by", grouping_variable_1, "and by", grouping_variable_2)
  }
  
  # Draw plot
  if (is.null(grouping_variable_2)) {
    p = ggplot(data=dataframe, 
               aes(x=dataframe[[grouping_variable_1]],
                   y=dataframe[[quantitative_variable]],
                   fill=dataframe[[grouping_variable_1]])) +
      geom_boxplot() +
      theme_minimal() + 
      theme(axis.line=element_line(size=1, colour="#d6d6d6"),
            axis.text=element_text(color="#3b3b3b"),
            axis.text.x=element_text(size=12),
            panel.grid.major=element_blank(),
            panel.grid.minor=element_blank(),
            legend.position = "none")
  } else {
    p = ggplot(data=dataframe, 
               aes(x=dataframe[[grouping_variable_1]],
                   y=dataframe[[quantitative_variable]],
                   fill=dataframe[[grouping_variable_2]])) + 
      geom_boxplot() +
      theme_minimal() + 
      theme(axis.line=element_line(size=1, colour="#d6d6d6"),
            axis.text=element_text(color="#3b3b3b"),
            axis.text.x=element_text(size=12),
            panel.grid.major=element_blank(),
            panel.grid.minor=element_blank())
  }
  p = p +
    labs(title="Distribution",
         subtitle=subtitle_for_plot,
         y=quantitative_variable,
         x=grouping_variable_1) + 
    scale_fill_brewer(grouping_variable_2, palette="Set2")
  plot(p)
}
