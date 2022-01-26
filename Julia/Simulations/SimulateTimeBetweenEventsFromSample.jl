using DataFrames, Distributions, Plots, PlotThemes, KernelDensity
gr()
theme(:bright)

function SimulateTimeBetweenEventsFromSample(
    dataframe::DataFrame,
    column_name::String,
    trials::Integer=10000,
    show_plot::Bool=true
)
    # Get parameters for simulation
    Param_ExponentialDistSim = fit(Exponential, dataframe[:, column_name])
    println("The mean time between events (θ) used in simulation: ", Param_ExponentialDistSim.θ)
    # Conduct simulation
    Arr_ExponentialDistSim = rand(Param_ExponentialDistSim, trials, 1)
    DF_ExponentialDistSim = DataFrame(Arr_ExponentialDistSim, :auto)
    # Get kernel density estimates for sample and simulation
    kde_sample = kde(dataframe[:, column_name])
    kde_simulation = kde(DF_ExponentialDistSim[:, "x1"])
    # Generate plot if requested by user
    if show_plot
        title_for_plot = "Simulated Time Between Events (Exponential Distribution)"
        p = histogram(
            DF_ExponentialDistSim[:, "x1"], 
            bins=:scott, 
            fillalpha=0.4, 
            label="Simulated Time", 
            title=title_for_plot, 
            xlabel=column_name,
            linecolor=:transparent
        )
        plot!(
            kde_simulation.x, 
            kde_simulation.density .* length(DF_ExponentialDistSim[:, "x1"]), 
            linewidth=3, 
            color=1, 
            label="Density of Simulation"
        )
        plot!(
            kde_sample.x, kde_sample.density .* length(DF_ExponentialDistSim[:, "x1"]), 
            alpha=0.6,
            linewidth=3, 
            color=2, 
            label="Density of Sample"
        )
        display(p)
    end
    # Return simulation results
    return DF_ExponentialDistSim
end



# # For development: Import gehan dataset using RDatasets
# using RDatasets
# gehan_data = dataset("MASS", "gehan")
# DF_Sim = SimulateTimeBetweenEventsFromSample(
#     gehan_data, 
#     "Time",
#     10000,
#     true
# )
