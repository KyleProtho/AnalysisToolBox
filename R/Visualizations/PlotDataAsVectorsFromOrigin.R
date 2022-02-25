library(ggplot2)
library(dplyr)

PlotDataAsVectorsFromOrigin = function(dataframe,
                                       x_coord_column_name,
                                       y_coord_column_name,
                                       main_x_coord,
                                       main_y_coord,
                                       main_vector_description=NULL,
                                       vector_color="#5ad1a9",
                                       sample_size=50,
                                       show_plot=TRUE) {
  # Randomly select subset of dataframe if larger than sample_size
  if (nrow(dataframe) > sample_size) {
    dataframe = sample_n(dataframe, sample_size)
  }
  
  # Get maximum of x and y for plot boundary
  plot_boundary_max = max(dataframe[[x_coord_column_name]], 
                          dataframe[[y_coord_column_name]],
                          main_x_coord, 
                          main_y_coord)
  if (plot_boundary_max < 0) {
    plot_boundary_max = 0
  }
  
  # Get minimum of x and y for plot boundary
  plot_boundary_min = min(dataframe[[x_coord_column_name]], 
                          dataframe[[y_coord_column_name]],
                          main_x_coord, 
                          main_y_coord)
  if (plot_boundary_min > 0) {
    plot_boundary_min = 0
  }
  
  # Create plot boundary
  plot_boundary = max(abs(plot_boundary_max), abs(plot_boundary_min))
  
  # Create plot
  p = ggplot() + 
    geom_point() + 
    scale_x_continuous(limits = c(plot_boundary*-1, plot_boundary)) +
    scale_y_continuous(limits = c(plot_boundary*-1, plot_boundary)) +
    # Add x and y axis lines
    geom_hline(yintercept=0,
               size=1, 
               color="#454545") + 
    geom_vline(xintercept=0,
               size=1,
               color="#454545")
  
  # Plot vectors for data sample
  p = p + geom_segment(data = dataframe,
                       aes(x=0,
                           y=0, 
                           xend=.data[[x_coord_column_name]],
                           yend=.data[[y_coord_column_name]]),
                       color=vector_color,
                       size=1,
                       alpha=0.2,
                       lineend="round",
                       linejoin="round",
                       arrow=arrow(length = unit(0.2, "inches")))
  
  # Draw main vector and finish plot styling
  p = p + geom_segment(aes(x=0,
                           y=0, 
                           xend=main_x_coord,
                           yend=main_y_coord),
                       color=vector_color,
                       size=3,
                       lineend="round",
                       linejoin="round",
                       arrow=arrow(length = unit(0.3, "inches"))) +
    theme_minimal() +
    theme(axis.title=element_blank(), 
          panel.grid.minor=element_blank(),
          legend.position="none") +
    # Add origin point dot
    geom_point(aes(x=0, y=0), 
               color="#454545",
               size=5) +
    # Add title
    labs(title = "Data as vectors from origin point",
         subtitle = main_vector_description)
  
  # Show plot if requested
  if (show_plot) {
    plot(p)
  }
  
  # Return plot
  return(p)
}
