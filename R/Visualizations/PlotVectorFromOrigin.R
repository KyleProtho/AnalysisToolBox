library(ggplot2)

PlotVectorFromOrigin = function(x_coord,
                                y_coord,
                                vector_color="#5ad1a9",
                                show_plot=TRUE) {
  # Get maximum of x and y for plot boundary
  plot_boundary_max = max(y_coord, x_coord)
  if (plot_boundary_max < 0) {
    plot_boundary_max = 0
  }
  # Get minimum of x and y for plot boundary
  plot_boundary_min = min(y_coord, x_coord)
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
               color="#454545") +
    # Draw vector
    geom_segment(aes(x=0,
                     y=0, 
                     xend=x_coord,
                     yend=y_coord),
                 color=vector_color,
                 size=3,
                 lineend = "round",
                 linejoin = "round",
                 arrow = arrow(length = unit(0.3, "inches"))) +
    theme_minimal() +
    theme(axis.title=element_blank(), 
          panel.grid.minor=element_blank(),
          legend.position="none") +
    # Add origin point dot
    geom_point(aes(x=0, y=0), 
               color="#454545",
               size=5) +
    # Add title
    labs(title = "Vector from origin point",
         subtitle = paste0("[", x_coord, ", ", y_coord, "]"))
  
  # Show plot if requested
  if (show_plot) {
    plot(p)
  }
  
  # Return plot
  return(p)
}

