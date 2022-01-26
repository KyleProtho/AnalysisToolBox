# Load packages
using DataFrames, Distributions, Plots, PlotThemes, KernelDensity
gr()
theme(:bright)

function SimulateBetaDistributionFromSample(
    dataframe::DataFrame,
    column_name::String,
    trials::Integer=10000,
    show_plot::Bool=true,
    kernel_density_scalar::Float64=0.05,
)
    # Get parameters for simulation
    Param_BetaDistSim = fit(Beta, dataframe[:, column_name])
    println("Alpha parameter (α) used in simulation: ", Param_BetaDistSim.α)
    println("Beta parameter (β) used in simulation: ", Param_BetaDistSim.β)
    # Conduct simulation
    Arr_BetaDistSim = rand(Param_BetaDistSim, trials, 1)
    DF_BetaDistSim = DataFrame(Arr_BetaDistSim, :auto)
    DF_BetaDistSim[:, "x1"] = round.(DF_BetaDistSim[:, "x1"], digits= 2)
    # Get kernel density estimates for sample and simulation
    kde_sample = kde(dataframe[:, column_name])
    kde_simulation = kde(DF_BetaDistSim[:, "x1"])
    # Generate plot if requested by user
    if show_plot
        title_for_plot = "Simulated Outcomes (Beta Distribution)"
        p = histogram(
            DF_BetaDistSim[:, "x1"],
            bins=20,
            fillalpha=0.4, 
            label="Simulated Outcomes", 
            title=title_for_plot, 
            xlabel=column_name,
            linecolor=:transparent
        )
        plot!(
            kde_simulation.x, 
            kde_simulation.density .* length(DF_BetaDistSim[:, "x1"]) .* kernel_density_scalar, 
            linewidth=3, 
            color=1, 
            label="Density of Simulation",
            xaxis=(0,1)
        )
        plot!(
            kde_sample.x, 
            kde_sample.density .* length(DF_BetaDistSim[:, "x1"]) .* kernel_density_scalar, 
            alpha=0.6, 
            linewidth=3, 
            color=2, 
            label="Density of Sample",
            xaxis=(0,1)
        )
        display(p)
    end
    # Return simulation results
    return DF_BetaDistSim
end


# For development: Test function based on random numbers between 0 and 1
Arr_RandomNumbers = rand(50, 1)
DF_RandomNumbers = DataFrame(Arr_RandomNumbers, :auto)
DF_Sim = SimulateBetaDistributionFromSample(
    DF_RandomNumbers, 
    "x1",
    10000,
    true,
    0.05
)
