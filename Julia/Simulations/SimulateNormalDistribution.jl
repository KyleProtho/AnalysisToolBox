# Load packages
using DataFrames, Distributions, Plots, PlotThemes, KernelDensity
gr()
theme(:bright)

# Define function
function SimulateNormalDistribution(
    observed_mean::Number,
    observed_sd::Number,
    outcome_name::String,
    trials::Integer=10000,
    show_plot::Bool=true,
    kernal_density_scalar::Float64=0.5
)
    # Get parameters for simulation
    Param_NormalDistSim = Normal(observed_mean, observed_sd)
    println("Mean (μ) used in simulation: ", Param_NormalDistSim.μ)
    println("Standard deviation (σ) used in simulation: ", Param_NormalDistSim.σ)

    # Conduct simulation
    Arr_NormalDistSim = rand(Param_NormalDistSim, trials, 1)
    DF_NormalDistSim = DataFrame(Arr_NormalDistSim, :auto)

    # Get kernel density estimates for sample and simulation
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
            xlabel=outcome_name,
            linecolor=:transparent
        )
        plot!(
            kde_simulation.x, 
            kde_simulation.density .* length(DF_NormalDistSim[:, "x1"]) .* kernal_density_scalar, 
            linewidth=3, 
            color=1, 
            label="Density of Simulation"
        )
        display(p)
    end
    
    # Return simulation results
    return DF_NormalDistSim
end


# # For development: Test normal distribution Monte Carlo
# DF_Sim = SimulateNormalDistribution(
#     10, 
#     5,
#     "Normally Distributed Outcome",
#     10000,
#     true,
#     1.0
# )
