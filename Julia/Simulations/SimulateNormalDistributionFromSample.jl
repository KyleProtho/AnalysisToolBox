# Load packages
using DataFrames, Distributions, Plots, PlotThemes, KernelDensity
gr()
theme(:bright)

# Define function
function SimulateNormalDistributionFromSample(
    dataframe::DataFrame,
    column_name::String,
    trials::Integer=10000,
    show_plot::Bool=true,
    kernal_density_scalar::Float16=0.5
)
    # Get parameters for simulation
    Param_NormalDistSim = fit(Normal, dataframe[:, column_name])
    println("Mean (μ) used in simulation: ", Param_NormalDistSim.μ)
    println("Standard deviation (σ) used in simulation: ", Param_NormalDistSim.σ)

    # Conduct simulation
    Arr_NormalDistSim = rand(Param_NormalDistSim, trials, 1)
    DF_NormalDistSim = DataFrame(Arr_NormalDistSim, :auto)

    # Get kernel density estimates for sample and simulation
    kde_sample = kde(dataframe[:, column_name])
    kde_simulation = kde(DF_NormalDistSim[:, "x1"])

    # Generate plot if requested by user
    if show_plot
        title_for_plot = "Simulated Outcomes (Normal Distribution)"
        p = histogram(
            DF_NormalDistSim[:, "x1"], 
            bins=:scott, 
            fillalpha=0.4, 
            label="Simulated Outcomes", 
            title=title_for_plot, 
            xlabel=column_name,
            linecolor=:transparent
        )
        plot!(
            kde_simulation.x, 
            kde_simulation.density .* length(DF_NormalDistSim[:, "x1"]) .* kernal_density_scalar, 
            linewidth=3, 
            color=1, 
            label="Density of Simulation"
        )
        plot!(
            kde_sample.x, 
            kde_sample.density .* length(DF_NormalDistSim[:, "x1"]) .* kernal_density_scalar, 
            alpha=0.6,
            linewidth=3, 
            color=2, 
            label="Density of Sample"
        )
        display(p)
    end
    
    # Return simulation results
    return DF_NormalDistSim
end



# # For development: Import iris dataset using RDatasets
# using RDatasets
# iris_data = dataset("datasets", "iris")
# DF_Sim = SimulateNormalDistributionFromSample(
#     iris_data, 
#     "SepalLength",
#     10000,
#     true,
#     0.2
# )
